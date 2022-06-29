#include "control_force_provider/control_force_calculator.h"

#include <boost/algorithm/clamp.hpp>

#include "control_force_provider/utils.h"

using namespace Eigen;
namespace control_force_provider::backend {

PotentialFieldMethod::PotentialFieldMethod(boost::shared_ptr<Obstacle>& obstacle, const ryml::NodeRef& config) : ControlForceCalculator(obstacle) {
  attraction_strength_ = utils::getConfigValue<double>(config, "attraction_strength")[0];
  attraction_distance_ = utils::getConfigValue<double>(config, "attraction_distance")[0];
  repulsion_strength_ = utils::getConfigValue<double>(config, "repulsion_strength")[0];
  repulsion_distance_ = utils::getConfigValue<double>(config, "repulsion_distance")[0];
}

void PotentialFieldMethod::getForceImpl(Vector4d& force) {
  Vector4d obstacle_position4d;
  obstacle_->getPosition(obstacle_position4d);
  Vector3d obstacle_position = obstacle_position4d.head(3);
  const Vector3d& goal3d = goal_.head(3);
  const Vector3d& ee_position3d = ee_position_.head(3);
  // attractive vector
  Vector3d attractive_vector = goal3d - ee_position3d;
  double ee_to_goal_distance = attractive_vector.norm();
  double smoothing_factor = 1;
  if (ee_to_goal_distance > attraction_distance_) {
    attractive_vector.normalize();  // use linear potential when far away from goal
    smoothing_factor = 0.1;
  }
  attractive_vector *= attraction_strength_ * smoothing_factor;
  // repulsive vector
  //  for the vector between "robot" and "obstacle" we take the shortest line between both tools:
  //  tool1:                                                    l1 = a1 + t*b1
  //  tool2:                                                    l2 = a2 + s*b2
  const Vector3d& a1 = rcm_;
  Vector3d b1 = ee_position3d - a1;
  const Vector3d& a2 = obstacle_->getRCM();
  Vector3d b2 = obstacle_position - a2;
  //  general vector between l1 and l2:                         v = a2 - a1 + sb2 - t*b1 = a' + s*b2 - t*b1
  Vector3d a_diff = a2 - a1;
  //  the shortest line is perpendicular to both tools (v•b1 = v•b2 = 0). We want to solve this LEQ for t and s:
  //                b1•a' + s*b1•b2 - t*b1•b1 = 0
  //                b2•a' + s*b2•b2 - t*b2•b1 = 0
  //  substitute e1 = b1•b2, e2 = b2•b2 and e3 = b1•b1
  double e1 = b1.dot(b2);
  double e2 = b2.dot(b2);
  double e3 = b1.dot(b1);
  //                b1•a' + s*e1 - t*e3 = 0                                 |* e2
  //                b2•a' + s*e2 - t*e1 = 0                                 |* e1
  //                ————————————————————————————
  //                b1•a'*e2 + s*e1*e2 - t*e2*e3 = 0                        -
  //                b2•a'*e1 + s*e1*e2 - t*e1*e1 = 0
  //                ————————————————————————————————
  //                b1•a'*e2 - t*e2*e3 - b2•a'*e1 + t*e1*e1 = 0             |+ b2•a'*e1, - b1•a'*e2
  //                ———————————————————————————————————————————
  //                t*(e1*e1 - e2*e3) = a'•(b2*e1 - b1*e2)                  |: (e1*e1 - b1•b1*e2)
  //                ——————————————————————————————————————
  //                t = a'•(b2*e1 - b1*e2) / (e1*e1 - e2*e3)
  double t = a_diff.dot(b2 * e1 - b1 * e2) / (e1 * e1 - e2 * e3);

  Vector3d repulsive_vector = {0, 0, 0};
  if (t > 0) {  // both points lie past the RCMs (=inside the abdomen). If not we don't need to calculate the vector anyway
    //  use the first equation to get s:
    //                s*e1 = t*e3 - b1•a'                                     |: e1
    //                ———————————————————
    //                s = (t*e3 - b1•a') / e1
    double s = (t * e3 - b1.dot(a_diff)) / e1;
    t = boost::algorithm::clamp(t, 0, 1);
    s = boost::algorithm::clamp(s, 0, 1);
    point_on_l1_ = (a1 + t * b1);
    point_on_l2_ = (a2 + s * b2);
    Vector3d l2_to_l1 = point_on_l1_ - point_on_l2_;
    double l2_to_l1_distance = l2_to_l1.norm();
    // continue with PFM
    if (l2_to_l1_distance < repulsion_distance_) {
      l2_to_l1 *= t;  // reduce the magnitude relative to the distance of the point on l1 to the end effector
      repulsive_vector = (repulsion_strength_ / l2_to_l1_distance - repulsion_strength_ / repulsion_distance_) / (l2_to_l1_distance * l2_to_l1_distance) *
                         l2_to_l1.normalized();
    }
  }
  Vector3d force3d = attractive_vector + repulsive_vector;
  force = {force3d[0], force3d[1], force3d[2], 0};
}
}  // namespace control_force_provider::backend