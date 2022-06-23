#pragma once

#include <Eigen/Dense>

#include <boost/shared_ptr.hpp>

#include "control_force_calculator.h"

using namespace control_force_provider::backend;
namespace control_force_provider {
class ControlForceProvider {
 private:
  boost::shared_ptr<ControlForceCalculator> control_force_calculator;

 public:
  ControlForceProvider();
  void getForce(Eigen::Vector4d& force, const Eigen::Vector3d& ee_position);

  const Eigen::Vector3d& getRCM() const { return control_force_calculator->getRCM(); }
  void setRCM(const Eigen::Vector3d& rcm) { control_force_calculator->setRCM(rcm); }
};
}  // namespace control_force_provider
