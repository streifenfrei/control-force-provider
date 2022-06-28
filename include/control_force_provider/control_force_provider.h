#pragma once

#include <geometry_msgs/Point.h>
#include <ros/ros.h>

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <ryml.hpp>

#include "control_force_calculator.h"
#include "visualizer.h"

using namespace control_force_provider::backend;
namespace control_force_provider {
namespace backend {
class ROSNode {
 public:
  explicit ROSNode(const std::string& node_name) {
    int argc = 0;
    ros::init(argc, nullptr, node_name);
  }
  ~ROSNode() { ros::shutdown(); }
};
}  // namespace backend
class ControlForceProvider : ROSNode {
 private:
  boost::shared_ptr<ControlForceCalculator> control_force_calculator_;
  const ryml::Tree config_;
  ros::NodeHandle node_handle_{};
  ros::AsyncSpinner spinner_{1};
  ros::Subscriber goal_subscriber_;
  boost::shared_ptr<Visualizer> visualizer_;
  ryml::Tree loadConfig();

  void goalCallback(const geometry_msgs::Point& goal);

 public:
  ControlForceProvider();
  void getForce(Eigen::Vector4d& force, const Eigen::Vector3d& ee_position);
  ~ControlForceProvider();

  [[nodiscard]] const Eigen::Vector3d& getRCM() const { return control_force_calculator_->getRCM(); }
  void setRCM(const Eigen::Vector3d& rcm) { control_force_calculator_->setRCM(rcm); }
};
}  // namespace control_force_provider
