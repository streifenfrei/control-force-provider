#pragma once

#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <vector>

namespace control_force_provider::backend {
class Obstacle {
 private:
  ros::Time start_time;

 protected:
  const std::string id_;
  Eigen::Vector3d rcm_;
  virtual Eigen::Vector4d getPositionAt(const ros::Time& ros_time) = 0;

 public:
  Obstacle(const std::string& id) : start_time(ros::Time::now()), id_(id){};
  void reset(double offset = 0);
  Eigen::Vector4d getPosition();
  virtual ~Obstacle() = default;

  [[nodiscard]] const Eigen::Vector3d& getRCM() const { return rcm_; }
};

class WaypointsObstacle : public Obstacle {
 private:
  std::vector<Eigen::Vector3d> waypoints_;
  std::vector<Eigen::Vector3d> segments_;
  std::vector<Eigen::Vector3d> segments_normalized_;
  std::vector<double> segments_lengths_;
  std::vector<double> segments_durations_;
  double total_duration_;
  double speed_;  // m/s

 public:
  WaypointsObstacle(const YAML::Node& config, const std::string& id);
  Eigen::Vector4d getPositionAt(const ros::Time& ros_time) override;
  ~WaypointsObstacle() override = default;
};
}  // namespace control_force_provider::backend