#include "control_force_provider/control_force_provider.h"

#include <eigen_conversions/eigen_msg.h>

#include <boost/optional.hpp>
#include <iostream>

#include "control_force_provider/utils.h"

using namespace Eigen;
using namespace control_force_provider::utils;
using namespace control_force_provider::exceptions;
namespace control_force_provider {
void ControlForceProvider::loadConfig() {
  std::string config_file;
  if (!node_handle_.getParam("control_force_provider/config", config_file))
    ROS_ERROR_STREAM_NAMED("control_force_provider", "Please provide a config file as ROS parameter: control_force_provider/config");
  config_ = boost::make_shared<YAML::Node>(YAML::LoadFile(config_file));
}

ControlForceProvider::ControlForceProvider() : ROSNode("control_force_provider") {
  loadConfig();
  control_force_calculator_ = ControlForceCalculator::createFromConfig(*config_, node_handle_);
  // load visualizer
  if ((*config_)["visualize"].IsDefined() && getConfigValue<bool>(*config_, "visualize")[0]) {
    visualizer_ = boost::make_shared<Visualizer>(node_handle_, control_force_calculator_);
  }
  rcm_subscriber_ = node_handle_.subscribe(getConfigValue<std::string>(*config_, "rcm_topic")[0], 100, &ControlForceProvider::rcmCallback, this);
  goal_subscriber_ = node_handle_.subscribe("control_force_provider/goal", 100, &ControlForceProvider::goalCallback, this);
  spinner_.start();
}

void ControlForceProvider::getForce(torch::Tensor& force, const torch::Tensor& ee_position) { control_force_calculator_->getForce(force, ee_position); }

void ControlForceProvider::rcmCallback(const geometry_msgs::PointStamped& rcm) {
  Vector3d rcm_eigen;
  tf::pointMsgToEigen(rcm.point, rcm_eigen);
  control_force_calculator_->setRCM(utils::vectorToTensor(rcm_eigen));
}

void ControlForceProvider::goalCallback(const geometry_msgs::Point& goal) {
  Vector3d goal_eigen;
  tf::pointMsgToEigen(goal, goal_eigen);
  control_force_calculator_->setGoal(utils::vectorToTensor(goal_eigen));
}

ControlForceProvider::~ControlForceProvider() {
  spinner_.stop();
  node_handle_.shutdown();
}

SimulatedRobot::SimulatedRobot(torch::Tensor rcm, torch::Tensor position, ControlForceProvider& cfp)
    : rcm_(std::move(rcm)), position_(std::move(position)), cfp_(cfp), velocity_(torch::zeros(3, utils::getTensorOptions())) {
  cfp_.setRCM(rcm_);
}

void SimulatedRobot::update() {
  cfp_.getForce(force_, position_);
  double magnitude = utils::norm(force_).item().toDouble();
  if (magnitude > max_force) {
    force_ = force_ / magnitude * max_force;
  }
  position_ += force_;
}
}  // namespace control_force_provider