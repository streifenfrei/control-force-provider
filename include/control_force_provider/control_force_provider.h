#pragma once

#include <Eigen/Dense>

namespace control_force_provider {
class ControlForceProvider {
 private:
  Eigen::Vector3d rcm;

 public:
  ControlForceProvider();
  void getForce(Eigen::Vector4d& force, const Eigen::Vector3d& ee_position);

  const Eigen::Vector3d& getRCM() const;
  void setRCM(const Eigen::Vector3d& rcm);
};
}  // namespace control_force_provider
