#pragma once

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>

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
  boost::shared_ptr<YAML::Node> config_;
  ros::NodeHandle node_handle_{};
  ros::AsyncSpinner spinner_{1};
  ros::Subscriber rcm_subscriber_;
  ros::Subscriber goal_subscriber_;
  boost::shared_ptr<Visualizer> visualizer_;
  void loadConfig();

  void goalCallback(const geometry_msgs::Point& goal);  // TODO: handle Vector4d goal (include roll)
  void rcmCallback(const geometry_msgs::PointStamped& rcm);

 public:
  ControlForceProvider();
  void getForce(Eigen::Vector4d& force, const Eigen::Vector4d& ee_position);
  ~ControlForceProvider();

  [[nodiscard]] const Eigen::Vector3d& getRCM() const { return control_force_calculator_->getRCM(); }
  void setRCM(const Eigen::Vector3d& rcm) { control_force_calculator_->setRCM(rcm); }
};

class SimulatedRobot {
 private:
  const static inline double max_force = 1e-4;
  ControlForceProvider& cfp_;
  Eigen::Vector3d rcm_;
  Eigen::Vector4d position_;
  Eigen::Vector4d velocity_;
  Eigen::Vector4d force_;

 public:
  SimulatedRobot(Eigen::Vector3d rcm, Eigen::Vector4d position, ControlForceProvider& cfp);
  void update();
};
}  // namespace control_force_provider
