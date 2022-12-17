#include "control_force_provider/rl_environment.h"

#include <pybind11/stl.h>
#include <torch/extension.h>

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
  goal_reached_threshold_distance_ = utils::getConfigValue<double>(config["rl"], "goal_reached_threshold_distance")[0];
  ee_positions_ = episode_context_->getStart();
  env_->setRCM(torch::from_blob(rcm.data(), {1, 3}, torch::kFloat64).clone());
  if (utils::getConfigValue<bool>(config["rl"], "rcm_origin")[0]) env_->setOffset(env_->getRCM().clone());
  env_->update(ee_positions_);
  goal_delay_count_ = torch::zeros({batch_size_, 1}, torch::kInt);
  if (utils::getConfigValue<bool>(config, "visualize")[0])
    visualizer_ = boost::make_shared<Visualizer>(node_handle_, env_, episode_context_, std::thread::hardware_concurrency());
  spinner_.start();
}

std::map<std::string, torch::Tensor> TorchRLEnvironment::getStateDict() {
  std::map<std::string, torch::Tensor> out;
  out["state"] = state_provider_->createState().to(torch::kFloat32);
  out["robot_position"] = env_->getEePosition().to(torch::kFloat32, false, true);
  out["robot_velocity"] = env_->getEeVelocity().to(torch::kFloat32, false, true);
  out["robot_rcm"] = env_->getRCM().to(torch::kFloat32, false, true);
  out["obstacles_positions"] = torch::cat(TensorList(&env_->getObPositions()[0], env_->getObPositions().size()), -1).to(torch::kFloat32);
  out["obstacles_velocities"] = torch::cat(TensorList(&env_->getObVelocities()[0], env_->getObVelocities().size()), -1).to(torch::kFloat32);
  out["obstacles_rcms"] = torch::cat(TensorList(&env_->getObRCMs()[0], env_->getObRCMs().size()), -1).to(torch::kFloat32);
  out["points_on_l1"] = torch::cat(TensorList(&env_->getPointsOnL1()[0], env_->getPointsOnL1().size()), -1).to(torch::kFloat32);
  out["points_on_l2"] = torch::cat(TensorList(&env_->getPointsOnL2()[0], env_->getPointsOnL2().size()), -1).to(torch::kFloat32);
  out["goal"] = env_->getGoal().to(torch::kFloat32, false, true);
  out["workspace_bb_origin"] = env_->getWorkspaceBbOrigin().to(torch::kFloat32, false, true);
  out["workspace_bb_dims"] = env_->getWorkspaceBbDims().to(torch::kFloat32, false, true);
  return out;
}

std::map<std::string, torch::Tensor> TorchRLEnvironment::observe(const Tensor& actions) {
  // env_->clipForce(actions_copy);
  ee_positions_ += actions.to(device_) * interval_duration_;
  ee_positions_ =
      ee_positions_.clamp(env_->getWorkspaceBbOrigin() + env_->getOffset(), env_->getWorkspaceBbOrigin() + env_->getOffset() + env_->getWorkspaceBbDims());
  // check if episode ended
  torch::Tensor reached_goal = utils::norm((env_->getGoal() + env_->getOffset() - ee_positions_)) < goal_reached_threshold_distance_;
  goal_delay_count_ = torch::where(reached_goal, goal_delay_count_ + 1, goal_delay_count_);
  torch::Tensor episode_end = goal_delay_count_ >= goal_delay;
  episode_context_->generateEpisode(episode_end);
  episode_context_->startEpisode(episode_end);
  goal_delay_count_ = torch::where(episode_end, 0, goal_delay_count_);
  ee_positions_ = torch::where(episode_end, episode_context_->getStart(), ee_positions_);
  env_->setGoal(episode_context_->getGoal());
  env_->update(ee_positions_);
  *time_ += interval_duration_ * 1e-3;
  return getStateDict();
}

PYBIND11_MODULE(native, module) {
  py::class_<TorchRLEnvironment>(module, "RLEnvironment")
      .def(py::init<std::string, std::array<double, 3>>())
      .def(py::init<std::string, std::array<double, 3>, bool>())
      .def("observe", &TorchRLEnvironment::observe, py::return_value_policy::move);
}
}  // namespace control_force_provider::backend
