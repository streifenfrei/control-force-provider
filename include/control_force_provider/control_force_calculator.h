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
  const Eigen::Vector3d& getRCM() const { return rcm; }
  void setRCM(const Eigen::Vector3d& rcm) { this->rcm = rcm; };
 private:
  Eigen::Vector3d rcm;
  boost::shared_ptr<Obstacle> obstacle;
};
}