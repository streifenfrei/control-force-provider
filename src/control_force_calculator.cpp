#include "control_force_provider/control_force_calculator.h"

namespace control_force_provider::backend {

PotentialFieldMethod::PotentialFieldMethod(boost::shared_ptr<Obstacle>& obstacle, const ryml::NodeRef& config) : ControlForceCalculator(obstacle) {}

void PotentialFieldMethod::getForce(Eigen::Vector4d& force, const Eigen::Vector3d& ee_position) {}
}  // namespace control_force_provider::backend