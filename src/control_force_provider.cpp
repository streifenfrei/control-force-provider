#include "control_force_provider/control_force_provider.h"

#include <eigen_conversions/eigen_msg.h>

#include <fstream>
#include <iostream>

#include "control_force_provider/utils.h"

using namespace Eigen;
using namespace control_force_provider::utils;
using namespace control_force_provider::exceptions;
namespace control_force_provider {
void ControlForceProvider::loadConfig() {
  std::string config_file;
  if (!node_handle_.getParam("control_force_provider/config", config_file))
    throw ConfigError("Please provide a config file as ROS parameter: control_force_provider/config");
  std::ifstream stream(config_file);
  std::stringstream buffer;
  buffer << stream.rdbuf();
  std::string content = buffer.str();
  config_ = boost::make_shared<ryml::Tree>(ryml::parse_in_arena(to_csubstr(content)));
}

ControlForceProvider::ControlForceProvider() : ROSNode("control_force_provider") {
  loadConfig();
  ryml::NodeRef config_root_node = config_->rootref();
  std::string obstacle_type = getConfigValue<std::string>(config_root_node, "obstacle_type")[0];
  boost::shared_ptr<Obstacle> obstacle;
  // load obstacle
  if (obstacle_type == "simulated") {
    ryml::NodeRef config_node = getConfigValue<ryml::NodeRef>(config_root_node, "simulated_obstacle")[0];
    obstacle = boost::static_pointer_cast<Obstacle>(boost::make_shared<SimulatedObstacle>(config_node));
  } else
    throw ConfigError("Unknown obstacle type '" + obstacle_type + "'");
  // load calculator
  std::string calculator_type = getConfigValue<std::string>(config_root_node, "algorithm")[0];
  if (calculator_type == "pfm") {
    ryml::NodeRef config_node = getConfigValue<ryml::NodeRef>(config_root_node, "pfm")[0];
    control_force_calculator_ = boost::static_pointer_cast<ControlForceCalculator>(boost::make_shared<PotentialFieldMethod>(obstacle, config_node));
  } else if (calculator_type == "rl") {
    ryml::NodeRef config_node = getConfigValue<ryml::NodeRef>(config_root_node, "rl")[0];
    std::string rl_type = getConfigValue<std::string>(config_node, "type")[0];
    if (rl_type == "dqn") {
      control_force_calculator_ = boost::static_pointer_cast<ControlForceCalculator>(boost::make_shared<DeepQNetworkAgent>(obstacle, config_node));
    } else
      throw ConfigError("Unknown RL type '" + rl_type + "'");
  } else
    throw ConfigError("Unknown calculator type '" + calculator_type + "'");
  // load visualizer
  if (config_root_node.has_child("visualize") && getConfigValue<bool>(config_root_node, "visualize")[0]) {
    visualizer_ = boost::make_shared<Visualizer>(node_handle_, control_force_calculator_);
  }
  rcm_subscriber_ = node_handle_.subscribe(getConfigValue<std::string>(config_root_node, "rcm_topic")[0], 100, &ControlForceProvider::rcmCallback, this);
  goal_subscriber_ = node_handle_.subscribe("control_force_provider/goal", 100, &ControlForceProvider::goalCallback, this);
  spinner_.start();
}

void ControlForceProvider::getForce(Vector4d& force, const Vector4d& ee_position) { control_force_calculator_->getForce(force, ee_position); }

void ControlForceProvider::rcmCallback(const geometry_msgs::PointStamped& rcm) {
  Vector3d rcm_eigen;
  tf::pointMsgToEigen(rcm.point, rcm_eigen);
  control_force_calculator_->setRCM(rcm_eigen);
}

void ControlForceProvider::goalCallback(const geometry_msgs::Point& goal) {
  Vector3d goal_eigen;
  tf::pointMsgToEigen(goal, goal_eigen);
  Vector4d goal_eigen4d = {goal_eigen[0], goal_eigen[1], goal_eigen[2], 0};
  control_force_calculator_->setGoal(goal_eigen4d);
}

ControlForceProvider::~ControlForceProvider() {
  spinner_.stop();
  node_handle_.shutdown();
}
}  // namespace control_force_provider