#pragma once

#include <torch/torch.h>

#include <array>

#include "control_force_calculator.h"
#include "control_force_provider.h"

namespace control_force_provider::backend {
class TorchRLEnvironment : ROSNode {
 private:
  double goal_reached_threshold_distance_;
  double collision_threshold_distance_;
  double max_force_;
  int timeout_;
  torch::DeviceType device_;
  torch::Tensor ee_positions_;
  torch::Tensor epoch_count_;
  torch::Tensor is_terminal_;
  torch::Tensor reached_goal_;
  torch::Tensor collided_;
  torch::Tensor is_timeout_;
  double interval_duration_;
  unsigned int batch_size_;
  boost::shared_ptr<ManualTime> time_;
  boost::shared_ptr<Environment> env_;
  boost::shared_ptr<EpisodeContext> episode_context_;
  boost::shared_ptr<StateProvider> state_provider_;
  ros::NodeHandle node_handle_{};
  ros::AsyncSpinner spinner_{1};
  std::map<std::string, torch::Tensor> getStateDict();
  boost::shared_ptr<Visualizer> visualizer_;
  boost::shared_ptr<PotentialFieldMethod> pfm_;
  void copyToDevice(torch::DeviceType device);

 public:
  TorchRLEnvironment(const std::string& config_file, std::array<float, 3> rcm, double goal_distance, bool force_cpu = false, bool visualize = false);
  std::map<std::string, torch::Tensor> observe(const torch::Tensor& actions);
  void setCustomMarker(const std::string& key, const torch::Tensor& marker);
};
}  // namespace control_force_provider::backend