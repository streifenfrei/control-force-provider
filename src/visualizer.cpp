#include "control_force_provider/visualizer.h"

#include <Eigen/Dense>
#include <boost/pointer_cast.hpp>

using namespace Eigen;
namespace control_force_provider::backend {
Visualizer::Visualizer(ros::NodeHandle &node_handle, boost::shared_ptr<ControlForceCalculator> control_force_calculator)
    : node_handle_(node_handle),
      control_force_calculator_(control_force_calculator),
      visual_tools_("world", "/visualization_marker_array", node_handle),
      timer_(node_handle_.createTimer(ros::Duration(0.05), &Visualizer::callback, this)) {}

void Visualizer::callback(const ros::TimerEvent &event) {
  visual_tools_.deleteAllMarkers();
  // obstacle
  Eigen::Vector4d obstacle_position;
  control_force_calculator_->obstacle_->getPosition(obstacle_position);
  Eigen::Vector3d obstacle_rcm = control_force_calculator_->obstacle_->getRCM();
  visual_tools_.publishSphere(obstacle_rcm, rviz_visual_tools::BROWN);
  visual_tools_.publishLine(obstacle_rcm, obstacle_position.head(3), rviz_visual_tools::BROWN);
  // control_force_calculator
  const Eigen::Vector4d& ee_position = control_force_calculator_->ee_position_;
  const Eigen::Vector3d& robot_rcm = control_force_calculator_->rcm_;
  visual_tools_.publishLine(robot_rcm, ee_position.head(3), rviz_visual_tools::PURPLE);
  visual_tools_.publishSphere(control_force_calculator_->goal_.head(3), rviz_visual_tools::BLUE);
  boost::shared_ptr<PotentialFieldMethod> pfm = boost::dynamic_pointer_cast<PotentialFieldMethod>(control_force_calculator_);
  if (pfm) {
    double distance = (pfm->point_on_l2_ - pfm->point_on_l1_).norm();
    if (distance < pfm->repulsion_distance_) visual_tools_.publishLine(pfm->point_on_l1_, pfm->point_on_l2_, rviz_visual_tools::CYAN);
  }
  visual_tools_.trigger();
}
}  // namespace control_force_provider::backend