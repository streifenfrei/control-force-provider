#include "control_force_provider/rl_environment.h"

#include <pybind11/stl.h>
#include <torch/extension.h>

using namespace control_force_provider;
using namespace torch;

namespace control_force_provider::backend {
TorchRLEnvironment::TorchRLEnvironment(const std::string& config_file) : ROSNode("cfp_rl_environment") {
  YAML::Node config = YAML::LoadFile(config_file);
  Time::setType<ManualTime>();
  time_ = boost::static_pointer_cast<ManualTime>(Time::getInstance());
  batch_size_ = utils::getConfigValue<int>(config["rl"], "robot_batch")[0];
  env_ = boost::make_shared<Environment>(config, batch_size_);
  episode_context_ = boost::make_shared<EpisodeContext>(env_->obstacles, env_->obstacle_loader, config["rl"], batch_size_);
  episode_context_->generateEpisode();
  episode_context_->startEpisode();
  state_provider_ = boost::make_shared<StateProvider>(*env_, utils::getConfigValue<std::string>(config["rl"], "state_pattern")[0]);
  interval_duration_ = utils::getConfigValue<double>(config["rl"], "interval_duration")[0];
  goal_reached_threshold_distance_ = utils::getConfigValue<double>(config["rl"], "goal_reached_threshold_distance")[0];
  ee_positions_ = episode_context_->getStart();
  env_->update(ee_positions_);
  goal_delay_count_ = torch::zeros({batch_size_, 1}, torch::kInt);
  spinner_.start();
}

std::map<std::string, torch::Tensor> TorchRLEnvironment::getStateDict() {
  std::map<std::string, torch::Tensor> out;
  out["state"] = state_provider_->createState();
  out["robot_position"] = torch::empty_like(env_->ee_position).copy_(env_->ee_position);
  out["robot_velocity"] = torch::empty_like(env_->ee_velocity).copy_(env_->ee_velocity);
  out["robot_rcm"] = torch::empty_like(env_->rcm).copy_(env_->rcm);
  out["obstacles_positions"] = torch::cat(TensorList(&env_->ob_positions[0], env_->ob_positions.size()));
  out["obstacles_velocities"] = torch::cat(TensorList(&env_->ob_velocities[0], env_->ob_velocities.size()));
  out["obstacles_rcms"] = torch::cat(TensorList(&env_->ob_rcms[0], env_->ob_rcms.size()));
  out["points_on_l1"] = torch::cat(TensorList(&env_->points_on_l1_[0], env_->points_on_l1_.size()));
  out["points_on_l2"] = torch::cat(TensorList(&env_->points_on_l2_[0], env_->points_on_l2_.size()));
  out["goal"] = torch::empty_like(env_->goal).copy_(env_->goal);
  out["workspace_bb_origin"] = torch::empty_like(env_->workspace_bb_origin).copy_(env_->workspace_bb_origin);
  out["workspace_bb_dims"] = torch::empty_like(env_->workspace_bb_dims).copy_(env_->workspace_bb_dims);
  return out;
}

std::map<std::string, torch::Tensor> TorchRLEnvironment::observe(const Tensor& actions) {
  torch::Tensor actions_copy = actions;
  env_->clipForce(actions_copy);
  ee_positions_ += actions_copy * interval_duration_;
  // check if episode ended
  torch::Tensor reached_goal = utils::norm((env_->goal - ee_positions_)) < goal_reached_threshold_distance_;
  goal_delay_count_ = torch::where(reached_goal, goal_delay_count_ + 1, goal_delay_count_);
  torch::Tensor episode_end = goal_delay_count_ >= goal_delay;
  episode_context_->generateEpisode(episode_end);
  episode_context_->startEpisode(episode_end);
  goal_delay_count_ = torch::where(episode_end, 0, goal_delay_count_);
  ee_positions_ = torch::where(episode_end, episode_context_->getStart(), ee_positions_);
  env_->goal = episode_context_->getGoal();
  env_->update(ee_positions_);
  *time_ += interval_duration_ * 1e-3;
  return getStateDict();
}

PYBIND11_MODULE(native, module) {
  py::class_<TorchRLEnvironment>(module, "RLEnvironment")
      .def(py::init<std::string>())
      .def("observe", &TorchRLEnvironment::observe, py::return_value_policy::move);
}
}  // namespace control_force_provider::backend
