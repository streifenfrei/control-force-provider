#include "control_force_provider/rl_environment.h"

#include <torch/extension.h>

using namespace torch;

namespace control_force_provider::backend {
TorchRLEnvironment::TorchRLEnvironment() {
  auto options = TensorOptions().dtype(kFloat64);
  state_ = torch::ones(90, options);
}

void TorchRLEnvironment::update(const Tensor& actions) {}

const Tensor& TorchRLEnvironment::getState() { return state_; }

PYBIND11_MODULE(native, module) {
  py::class_<TorchRLEnvironment>(module, "RLEnvironment")
      .def(py::init<>())
      .def("update", &TorchRLEnvironment::update)
      .def("getState", &TorchRLEnvironment::getState);
}
}  // namespace control_force_provider::backend
