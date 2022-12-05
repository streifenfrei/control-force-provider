#pragma once

#include <torch/torch.h>

#include "control_force_calculator.h"
#include "control_force_provider.h"

namespace control_force_provider::backend {
class TorchRLEnvironment : ROSNode {
 private:
  const inline static int goal_delay = 10;
  double goal_reached_threshold_distance_;
  torch::Tensor ee_positions_;
  torch::Tensor goal_delay_count_;
  double interval_duration_;
  unsigned int batch_size_;
  boost::shared_ptr<ManualTime> time_;
  boost::shared_ptr<Environment> env_;
  boost::shared_ptr<EpisodeContext> episode_context_;
  boost::shared_ptr<StateProvider> state_provider_;
  ros::NodeHandle node_handle_{};
  ros::AsyncSpinner spinner_{1};

 public:
  TorchRLEnvironment(const std::string& config_file);
  torch::Tensor observe(const torch::Tensor& actions);
};
}  // namespace control_force_provider::backend