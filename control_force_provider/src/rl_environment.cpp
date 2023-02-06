#include "control_force_provider/rl_environment.h"

#include <pybind11/stl.h>
#include <torch/extension.h>

#include <limits>

using namespace control_force_provider;
using namespace torch;

namespace control_force_provider::backend {
TorchRLEnvironment::TorchRLEnvironment(const std::string& config_file, std::array<double, 3> rcm, bool force_cpu)
    : ROSNode("cfp_rl_environment"), device_(force_cpu || !torch::hasCUDA() ? torch::kCPU : torch::kCUDA) {
  YAML::Node config = YAML::LoadFile(config_file);
  Time::setType<ManualTime>();
  time_ = boost::static_pointer_cast<ManualTime>(Time::getInstance());
  batch_size_ = utils::getConfigValue<int>(config["rl"], "robot_batch")[0];
  env_ = boost::make_shared<Environment>(config, batch_size_, device_);
  episode_context_ = boost::make_shared<EpisodeContext>(env_->getObstacles(), env_->getObstacleLoader(), config["rl"], batch_size_, device_);
  episode_context_->generateEpisode();
  episode_context_->startEpisode();
  state_provider_ = boost::make_shared<StateProvider>(*env_, utils::getConfigValue<std::string>(config["rl"], "state_pattern")[0]);
  interval_duration_ = utils::getConfigValue<double>(config["rl"], "interval_duration")[0];
  collision_threshold_distance_ = utils::getConfigValue<double>(config["rl"], "collision_threshold_distance")[0];
  timeout_ = int(utils::getConfigValue<double>(config["rl"], "episode_timeout")[0] * 1000 / interval_duration_);
  timeout_ = timeout_ <= 0 ? std::numeric_limits<int>::max() : timeout_;
  epoch_count_ = torch::zeros({batch_size_, 1}, utils::getTensorOptions(device_, torch::kInt32));
  goal_reached_threshold_distance_ = utils::getConfigValue<double>(config["rl"], "goal_reached_threshold_distance")[0];
  ee_positions_ = episode_context_->getStart();
  is_terminal_ = torch::zeros({batch_size_, 1}, utils::getTensorOptions(device_, torch::kBool));
  env_->setRCM(torch::from_blob(rcm.data(), {1, 3}, torch::kFloat64).clone());
  if (utils::getConfigValue<bool>(config["rl"], "rcm_origin")[0]) env_->setOffset(env_->getRCM().clone());
  env_->update(ee_positions_);
  if (utils::getConfigValue<bool>(config, "visualize")[0]) {
    visualizer_ = boost::make_shared<Visualizer>(node_handle_, env_, episode_context_, std::thread::hardware_concurrency());
  }
  spinner_.start();
}

std::map<std::string, torch::Tensor> TorchRLEnvironment::getStateDict() {
  std::map<std::string, torch::Tensor> out;
  out["state"] = state_provider_->createState().to(utils::getTensorOptions(device_, torch::kFloat32));
  out["robot_position"] = env_->getEePosition().to(utils::getTensorOptions(device_, torch::kFloat32));
  out["robot_velocity"] = env_->getEeVelocity().to(utils::getTensorOptions(device_, torch::kFloat32));
  out["robot_rcm"] = env_->getRCM().to(utils::getTensorOptions(device_, torch::kFloat32));
  out["obstacles_positions"] =
      torch::cat(TensorList(&env_->getObPositions()[0], env_->getObPositions().size()), -1).to(utils::getTensorOptions(device_, torch::kFloat32));
  out["obstacles_velocities"] =
      torch::cat(TensorList(&env_->getObVelocities()[0], env_->getObVelocities().size()), -1).to(utils::getTensorOptions(device_, torch::kFloat32));
  out["obstacles_rcms"] = torch::cat(TensorList(&env_->getObRCMs()[0], env_->getObRCMs().size()), -1).to(utils::getTensorOptions(device_, torch::kFloat32));
  out["points_on_l1"] =
      torch::cat(TensorList(&env_->getPointsOnL1()[0], env_->getPointsOnL1().size()), -1).to(utils::getTensorOptions(device_, torch::kFloat32));
  out["points_on_l2"] =
      torch::cat(TensorList(&env_->getPointsOnL2()[0], env_->getPointsOnL2().size()), -1).to(utils::getTensorOptions(device_, torch::kFloat32));
  out["collision_distances"] =
      torch::cat(TensorList(&env_->getCollisionDistances()[0], env_->getCollisionDistances().size()), -1).to(utils::getTensorOptions(device_, torch::kFloat32));
  out["goal"] = env_->getGoal().to(utils::getTensorOptions(device_, torch::kFloat32));
  out["is_terminal"] = is_terminal_.clone();
  out["reached_goal"] = reached_goal_.clone();
  out["collided"] = collided_.clone();
  out["is_timeout"] = is_timeout_.clone();
  out["workspace_bb_origin"] = env_->getWorkspaceBbOrigin().to(utils::getTensorOptions(device_, torch::kFloat32));
  out["workspace_bb_dims"] = env_->getWorkspaceBbDims().to(utils::getTensorOptions(device_, torch::kFloat32));
  return out;
}

std::map<std::string, torch::Tensor> TorchRLEnvironment::observe(const Tensor& actions) {
  torch::Tensor actions_copy = actions.to(device_);
  torch::Tensor actions_is_nan = actions_copy.isnan();
  if (actions_is_nan.any().item().toBool()) {
    ROS_WARN_STREAM_NAMED("RLEnvironment", "Some actions are NaN.");
    actions_copy = torch::where(actions_is_nan, 0, actions_copy);
  }
  // env_->clipForce(actions_copy);
  ee_positions_ += actions_copy * interval_duration_;
  ee_positions_ =
      ee_positions_.clamp(env_->getWorkspaceBbOrigin() + env_->getOffset(), env_->getWorkspaceBbOrigin() + env_->getOffset() + env_->getWorkspaceBbDims());
  // check if episode ended by reaching goal ...
  reached_goal_ = utils::norm((env_->getGoal() + env_->getOffset() - ee_positions_)) < goal_reached_threshold_distance_;
  // ... or by collision
  collided_ = torch::zeros_like(reached_goal_);
  for (auto& distance_to_obs : env_->getCollisionDistances()) {
    torch::Tensor not_nan = distance_to_obs.isnan().logical_not();
    collided_ = collided_.logical_or(not_nan.logical_and(distance_to_obs < collision_threshold_distance_));
  }
  // ... or by timeout
  is_timeout_ = epoch_count_ >= timeout_;
  episode_context_->generateEpisode(is_terminal_);
  episode_context_->startEpisode(is_terminal_);
  env_->setStartTime(torch::where(is_terminal_, Time::now(), env_->getStartTime()));
  ee_positions_ = torch::where(is_terminal_, episode_context_->getStart(), ee_positions_);
  is_terminal_ = reached_goal_.logical_or(collided_.logical_or(is_timeout_));
  epoch_count_ = torch::where(is_terminal_, 0, epoch_count_ + 1);
  env_->setGoal(episode_context_->getGoal());
  env_->update(ee_positions_);
  *time_ += interval_duration_ * 1e-3;
  return getStateDict();
}

void TorchRLEnvironment::setCustomMarker(const std::string& key, const torch::Tensor& marker) {
  if (visualizer_) visualizer_->setCustomMarker(key, marker);
}

PYBIND11_MODULE(native, module) {
  py::class_<TorchRLEnvironment>(module, "RLEnvironment")
      .def(py::init<std::string, std::array<double, 3>>())
      .def(py::init<std::string, std::array<double, 3>, bool>())
      .def("observe", &TorchRLEnvironment::observe, py::return_value_policy::move)
      .def("set_custom_marker", &TorchRLEnvironment::setCustomMarker);
}
}  // namespace control_force_provider::backend
