#pragma once

#include <ros/ros.h>

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>

#include "obstacle.h"

namespace control_force_provider::backend {
class ControlForceCalculator {
 public:
  explicit ControlForceCalculator(boost::shared_ptr<Obstacle>& obstacle) : obstacle_(obstacle) {}
  virtual void getForce(Eigen::Vector4d& force, const Eigen::Vector3d& ee_position) = 0;
  virtual ~ControlForceCalculator() = default;
  [[nodiscard]] const Eigen::Vector3d& getRCM() const { return rcm_; }
  void setRCM(const Eigen::Vector3d& rcm) { rcm_ = rcm; };

 protected:
  Eigen::Vector3d rcm_;
  boost::shared_ptr<Obstacle> obstacle_;
  friend class Visualizer;
};

class PotentialFieldMethod : public ControlForceCalculator {
 public:
  PotentialFieldMethod(boost::shared_ptr<Obstacle>& obstacle, const ryml::NodeRef& config);
  void getForce(Eigen::Vector4d& force, const Eigen::Vector3d& ee_position) override;
  ~PotentialFieldMethod() override = default;
};
}  // namespace control_force_provider::backend