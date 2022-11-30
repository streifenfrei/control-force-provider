#pragma once

#include <torch/torch.h>

#include "control_force_calculator.h"
#include "control_force_provider.h"

namespace control_force_provider::backend {
class TorchRLEnvironment {
 private:
  torch::Tensor ee_positions_;
  double interval_duration_;
  unsigned int batch_size_;
  boost::shared_ptr<ManualTime> time_;
  boost::shared_ptr<Environment> env_;
  boost::shared_ptr<EpisodeContext> episode_context_;
  boost::shared_ptr<StateProvider> state_provider_;
  torch::Tensor state_;

 public:
  TorchRLEnvironment(const std::string& config_file);
  void update(const torch::Tensor& actions);
  torch::Tensor getState();
};
}  // namespace control_force_provider::backend