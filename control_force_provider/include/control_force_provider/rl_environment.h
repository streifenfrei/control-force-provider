#pragma once

#include <torch/torch.h>

namespace control_force_provider::backend {
class TorchRLEnvironment {
 private:
  torch::Tensor state_;

 public:
  TorchRLEnvironment();
  void update(const torch::Tensor& actions);
  const torch::Tensor& getState();
};
}  // namespace control_force_provider::backend