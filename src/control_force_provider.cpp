#include "control_force_provider/control_force_provider.h"

using namespace Eigen;
namespace control_force_provider {
ControlForceProvider::ControlForceProvider() : rcm() {}

void ControlForceProvider::getForce(Vector4d& force, const Vector3d& ee_position) { force = {0, 0.000005, 0.000003, 0}; }

const Vector3d& ControlForceProvider::getRCM() const { return rcm; }
void ControlForceProvider::setRCM(const Vector3d& rcm) { ControlForceProvider::rcm = rcm; }
}  // namespace control_force_provider