#include "control_force_provider/rl_environment.h"

#include <torch/extension.h>

using namespace control_force_provider;
using namespace torch;

namespace control_force_provider::backend {
TorchRLEnvironment::TorchRLEnvironment(const std::string& config_file) {
  YAML::Node config = YAML::LoadFile(config_file);
  Time::setType<ManualTime>();
  time_ = boost::static_pointer_cast<ManualTime>(Time::getInstance());
  env_ = boost::make_shared<Environment>(config);
  episode_context_ = boost::make_shared<EpisodeContext>(env_->obstacles, env_->obstacle_loader, config);
  state_provider_ = boost::make_shared<StateProvider>(*env_, utils::getConfigValue<std::string>(config["rl"], "state_pattern")[0]);
  interval_duration_ = utils::getConfigValue<double>(config["rl"], "interval_duration")[0];
  ee_positions_ = torch::zeros({batch_size_, 3}, utils::getTensorOptions());
}

void TorchRLEnvironment::update(const Tensor& actions) {
  torch::Tensor actions_copy = actions;
  env_->clipForce(actions_copy);
  ee_positions_ += actions_copy * interval_duration_;
  // TODO: get new episodes
  env_->update(ee_positions_);
  *time_ += interval_duration_ * 1e-3;
}

Tensor TorchRLEnvironment::getState() { return state_provider_->createState(); }

PYBIND11_MODULE(native, module) {
  py::class_<TorchRLEnvironment>(module, "RLEnvironment")
      .def(py::init<std::string>())
      .def("update", &TorchRLEnvironment::update)
      .def("getState", &TorchRLEnvironment::getState);
}
}  // namespace control_force_provider::backend
