#include "control_force_provider/visualizer.h"

#include <Eigen/Dense>
#include <boost/pointer_cast.hpp>

using namespace Eigen;
namespace control_force_provider::backend {
Visualizer::Visualizer(ros::NodeHandle &node_handle, boost::shared_ptr<ControlForceCalculator> control_force_calculator)
    : node_handle_(node_handle),
      control_force_calculator_(control_force_calculator),
      visual_tools_("world", "/rviz_visual_tools", node_handle),
      timer_(node_handle_.createTimer(ros::Duration(0.05), &Visualizer::callback, this)) {}

void Visualizer::callback(const ros::TimerEvent &event) {
  Eigen::Vector4d position;
  control_force_calculator_->obstacle_->getPosition(position);
  visual_tools_.deleteAllMarkers();
  Eigen::Vector3d obstacle_rcm = control_force_calculator_->obstacle_->getRCM();
  visual_tools_.publishSphere(obstacle_rcm, rviz_visual_tools::PURPLE);
  visual_tools_.publishLine(obstacle_rcm, {position[0], position[1], position[2]}, rviz_visual_tools::BROWN);
  boost::shared_ptr<PotentialFieldMethod> pfm = boost::dynamic_pointer_cast<PotentialFieldMethod>(control_force_calculator_);
  if (pfm) {
    double distance = (pfm->point_on_l2_ - pfm->point_on_l1_).norm();
    if (distance < pfm->repulsion_distance_) visual_tools_.publishLine(pfm->point_on_l1_, pfm->point_on_l2_, rviz_visual_tools::CYAN);
  }
  visual_tools_.trigger();
}
}  // namespace control_force_provider::backend