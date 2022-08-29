#pragma once

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <vector>

namespace control_force_provider::backend {
class Obstacle {
 protected:
  const std::string id_;
  Eigen::Vector3d rcm_;

 public:
  Obstacle(const std::string& id) : id_(id){};
  virtual Eigen::Vector4d getPosition() = 0;
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
  Eigen::Vector4d getPosition() override;
  ~WaypointsObstacle() override = default;
};
}  // namespace control_force_provider::backend