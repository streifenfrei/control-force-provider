#include "control_force_provider/control_force_provider.h"

#include <fstream>
#include <iostream>

#include "control_force_provider/utils.h"

using namespace std;
using namespace Eigen;
using namespace control_force_provider::utils;
using namespace control_force_provider::exceptions;
namespace control_force_provider {
ryml::Tree ControlForceProvider::loadConfig() {
  string config_file;
  if (!node_handle_.getParam("control_force_provider/config", config_file))
    ROS_ERROR_STREAM_NAMED("control_force_provider", "Please provide a config file as ROS parameter: control_force_provider/config");
  ifstream stream(config_file);
  stringstream buffer;
  buffer << stream.rdbuf();
  string content = buffer.str();
  return ryml::parse_in_arena(to_csubstr(content));
}

ControlForceProvider::ControlForceProvider() : ROSNode("control_force_provider"), control_force_calculator_(nullptr), config(loadConfig()) { spinner_.start(); }

void ControlForceProvider::getForce(Vector4d& force, const Vector3d& ee_position) { control_force_calculator_->getForce(force, ee_position); }

ControlForceProvider::~ControlForceProvider() {
  spinner_.stop();
  node_handle_.shutdown();
}
}  // namespace control_force_provider