#include "control_force_provider/control_force_provider.h"

using namespace std;
using namespace Eigen;
namespace control_force_provider {
YAML::Node ControlForceProvider::loadConfig() {
  string config_file;
  if (!node_handle_.getParam("control_force_provider/config", config_file))
    ROS_ERROR_STREAM_NAMED("control_force_provider", "Please provide a config file as ROS parameter: control_force_provider/config");
  return YAML::LoadFile(config_file);
}

ControlForceProvider::ControlForceProvider() : ROSNode("control_force_provider"), control_force_calculator_(nullptr), config(loadConfig()) { spinner_.start(); }

void ControlForceProvider::getForce(Vector4d& force, const Vector3d& ee_position) { control_force_calculator_->getForce(force, ee_position); }

ControlForceProvider::~ControlForceProvider() {
  spinner_.stop();
  node_handle_.shutdown();
}
}  // namespace control_force_provider