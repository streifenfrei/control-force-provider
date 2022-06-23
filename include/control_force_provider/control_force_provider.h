#pragma once

#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>

#include "control_force_calculator.h"

using namespace control_force_provider::backend;
namespace control_force_provider {
namespace backend {
class ROSNode {
 public:
  explicit ROSNode(std::string node_name) {
    int argc = 0;
    ros::init(argc, nullptr, node_name);
  }

  ~ROSNode() { ros::shutdown(); }
};
}  // namespace backend
class ControlForceProvider : ROSNode {
 private:
  boost::shared_ptr<ControlForceCalculator> control_force_calculator_;
  YAML::Node config;
  ros::NodeHandle node_handle_{};
  ros::AsyncSpinner spinner_{1};

  YAML::Node loadConfig();

 public:
  ControlForceProvider();
  void getForce(Eigen::Vector4d& force, const Eigen::Vector3d& ee_position);
  ~ControlForceProvider();

  const Eigen::Vector3d& getRCM() const { return control_force_calculator_->getRCM(); }
  void setRCM(const Eigen::Vector3d& rcm) { control_force_calculator_->setRCM(rcm); }
};
}  // namespace control_force_provider
