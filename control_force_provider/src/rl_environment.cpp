#include "control_force_provider/rl_environment.h"

#include <torch/extension.h>

using namespace control_force_provider;
using namespace torch;

namespace control_force_provider::backend {
TorchRLEnvironment::TorchRLEnvironment(const std::string& config_file) : ROSNode("cfp_rl_environment") {
  YAML::Node config = YAML::LoadFile(config_file);
  Time::setType<ManualTime>();
  boost::shared_ptr<Time> time = Time::getInstance();
  time_ = boost::dynamic_pointer_cast<ManualTime>(time);
  batch_size_ = utils::getConfigValue<int>(config["rl"], "batch_size_exploration")[0];
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

torch::Tensor TorchRLEnvironment::observe(const Tensor& actions) {
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
  return state_provider_->createState();
}

PYBIND11_MODULE(native, module) {
  py::class_<TorchRLEnvironment>(module, "RLEnvironment").def(py::init<std::string>()).def("observe", &TorchRLEnvironment::observe);
}
}  // namespace control_force_provider::backend
