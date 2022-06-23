#pragma once

#include <Eigen/Dense>

namespace control_force_provider::backend {
class Obstacle {
 private:
  Eigen::Vector3d rcm;
 public:
  Obstacle() = default;
  virtual void getPosition(Eigen::Vector3d& position);
  virtual ~Obstacle() = default;

  const Eigen::Vector3d& getRCM() const { return rcm; }
};
}