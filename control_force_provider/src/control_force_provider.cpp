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
  YAML::Node obstacle_configs = getConfigValue<YAML::Node>(*config_, "obstacles")[0];
  // load obstacles
  std::vector<boost::shared_ptr<Obstacle>> obstacles;
  std::vector<boost::shared_ptr<FramesObstacle>> frames_obstacles;
  boost::optional<ObstacleLoader> obstacle_loader;
  std::string data_path;
  for (YAML::const_iterator it = obstacle_configs.begin(); it != obstacle_configs.end(); it++) {
    std::string id = it->first.as<std::string>();
    if (id == "data") {
      data_path = it->second.as<std::string>();
    } else {
      const YAML::Node& ob_config = it->second;
      std::string ob_type = getConfigValue<std::string>(ob_config, "type")[0];
      if (ob_type == "waypoints") {
        obstacles.push_back(boost::static_pointer_cast<Obstacle>(boost::make_shared<WaypointsObstacle>(ob_config, id)));
      } else if (ob_type == "csv") {
        boost::shared_ptr<FramesObstacle> obstacle = boost::make_shared<FramesObstacle>(id);
        obstacles.push_back(boost::static_pointer_cast<Obstacle>(obstacle));
        frames_obstacles.push_back(obstacle);
      } else
        throw ConfigError("Unknown obstacle type '" + ob_type + "'");
    }
  }
  // load calculator
  std::string calculator_type = getConfigValue<std::string>(*config_, "algorithm")[0];
  if (calculator_type == "pfm") {
    control_force_calculator_ = boost::static_pointer_cast<ControlForceCalculator>(boost::make_shared<PotentialFieldMethod>(obstacles, *config_, data_path));
  } else if (calculator_type == "rl") {
    std::string rl_type = getConfigValue<std::string>((*config_)["rl"], "type")[0];
    if (rl_type == "dqn") {
      control_force_calculator_ =
          boost::static_pointer_cast<ControlForceCalculator>(boost::make_shared<DeepQNetworkAgent>(obstacles, *config_, node_handle_, data_path));
    } else
      throw ConfigError("Unknown RL type '" + rl_type + "'");
  } else
    throw ConfigError("Unknown calculator type '" + calculator_type + "'");
  // load visualizer
  if ((*config_)["visualize"].IsDefined() && getConfigValue<bool>(*config_, "visualize")[0]) {
    visualizer_ = boost::make_shared<Visualizer>(node_handle_, control_force_calculator_);
  }
  rcm_subscriber_ = node_handle_.subscribe(getConfigValue<std::string>(*config_, "rcm_topic")[0], 100, &ControlForceProvider::rcmCallback, this);
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