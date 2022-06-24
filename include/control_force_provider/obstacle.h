#pragma once

#include <Eigen/Dense>
#include <ryml.hpp>
#include <vector>

namespace control_force_provider::backend {
class Obstacle {
 protected:
  Eigen::Vector3d rcm_;

 public:
  Obstacle() = default;
  virtual void getPosition(Eigen::Vector4d& position) = 0;
  virtual ~Obstacle() = default;

  [[nodiscard]] const Eigen::Vector3d& getRCM() const { return rcm_; }
};

class SimulatedObstacle : public Obstacle {
 private:
  std::vector<Eigen::Vector3d> waypoints_;
  std::vector<Eigen::Vector3d> segments_;
  std::vector<Eigen::Vector3d> segments_normalized_;
  std::vector<double> segments_lengths_;
  std::vector<double> segments_durations_;
  double total_duration_;
  double speed_;  // m/s

 public:
  explicit SimulatedObstacle(const ryml::NodeRef& config);
  void getPosition(Eigen::Vector4d& position) override;
  ~SimulatedObstacle() override = default;
};
}  // namespace control_force_provider::backend