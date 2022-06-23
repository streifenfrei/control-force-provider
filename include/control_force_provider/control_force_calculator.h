#pragma once

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>

#include "obstacle.h"

namespace control_force_provider::backend {
class ControlForceCalculator {
 public:
  ControlForceCalculator() {}
  virtual void getForce(Eigen::Vector4d& force, const Eigen::Vector3d& ee_position);
  virtual ~ControlForceCalculator() = default;
  const Eigen::Vector3d& getRCM() const { return rcm_; }
  void setRCM(const Eigen::Vector3d& rcm) { rcm_ = rcm; };

 private:
  Eigen::Vector3d rcm_;
  boost::shared_ptr<Obstacle> obstacle_;
};
}  // namespace control_force_provider::backend