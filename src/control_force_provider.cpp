#include "control_force_provider/control_force_provider.h"

using namespace Eigen;
namespace control_force_provider {
ControlForceProvider::ControlForceProvider(): control_force_calculator(nullptr) {}

void ControlForceProvider::getForce(Vector4d& force, const Vector3d& ee_position) { control_force_calculator->getForce(force, ee_position); }
}  // namespace control_force_provider