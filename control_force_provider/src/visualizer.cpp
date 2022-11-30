#include "control_force_provider/visualizer.h"

#include <eigen_conversions/eigen_msg.h>
#include <torch/torch.h>

#include <Eigen/Dense>
#include <boost/pointer_cast.hpp>

using namespace Eigen;
namespace control_force_provider::backend {
Visualizer::Visualizer(ros::NodeHandle& node_handle, boost::shared_ptr<ControlForceCalculator> control_force_calculator)
    : node_handle_(node_handle),
      control_force_calculator_(control_force_calculator),
      visual_tools_("world", "/visualization_marker_array", node_handle),
      timer_(node_handle_.createTimer(ros::Duration(0.05), &Visualizer::callback, this)) {}

void Visualizer::callback(const ros::TimerEvent& event) {
  // TODO: visualizer is not thread safe and currently crashes everything...
  visual_tools_.deleteAllMarkers();
  // obstacles
  const torch::Tensor& offset = control_force_calculator_->env.offset;
  for (size_t i = 0; i < control_force_calculator_->env.obstacles.size(); i++) {
    Vector3d obstacle_rcm = utils::tensorToVector(control_force_calculator_->env.ob_rcms[i] + offset);
    torch::Tensor obstacle_pos = control_force_calculator_->env.ob_positions[i];
    obstacle_pos += offset;
    visual_tools_.publishSphere(obstacle_rcm, rviz_visual_tools::BROWN);
    visual_tools_.publishLine(obstacle_rcm, utils::tensorToVector(obstacle_pos), rviz_visual_tools::BROWN);
  }
  // control_force_calculator
  Vector3d ee_position = utils::tensorToVector(control_force_calculator_->env.ee_position + offset);
  Vector3d robot_rcm = utils::tensorToVector(control_force_calculator_->env.rcm + offset);
  visual_tools_.publishLine(robot_rcm, ee_position, rviz_visual_tools::PURPLE);
  visual_tools_.publishSphere(utils::tensorToVector(control_force_calculator_->env.goal + offset), rviz_visual_tools::BLUE);
  geometry_msgs::Point ee_position_msg;
  tf::pointEigenToMsg(ee_position, ee_position_msg);
  geometry_msgs::Point target_msg;
  double scale =
      std::max(utils::tensorToVector(control_force_calculator_->env.ee_velocity).norm() / control_force_calculator_->env.max_force.to<double>() * 0.15, 0.05);
  tf::pointEigenToMsg(ee_position + utils::tensorToVector(control_force_calculator_->env.ee_velocity).normalized() * scale, target_msg);
  visual_tools_.publishArrow(ee_position_msg, target_msg, rviz_visual_tools::TRANSLUCENT_LIGHT, rviz_visual_tools::SMALL);

  auto acc1 = control_force_calculator_->env.workspace_bb_origin.accessor<double, 1>();
  Vector3d workspace_bb_origin(acc1[0], acc1[1], acc1[2]);
  auto acc2 = control_force_calculator_->env.workspace_bb_dims.accessor<double, 1>();
  Vector3d workspace_bb_dims = utils::tensorToVector(control_force_calculator_->env.workspace_bb_dims);
  visual_tools_.publishWireframeCuboid(
      Isometry3d(Translation3d(utils::tensorToVector(control_force_calculator_->env.workspace_bb_origin + offset) + 0.5 * workspace_bb_dims)),
      workspace_bb_dims[0], workspace_bb_dims[1], workspace_bb_dims[2], rviz_visual_tools::DARK_GREY);

  boost::shared_ptr<PotentialFieldMethod> pfm = boost::dynamic_pointer_cast<PotentialFieldMethod>(control_force_calculator_);
  if (pfm) {
    for (size_t i = 0; i < control_force_calculator_->env.obstacles.size(); i++) {
      double distance = utils::tensorToVector(pfm->env.points_on_l2_[i] - pfm->env.points_on_l1_[i]).norm();
      if (distance < pfm->repulsion_distance_)
        visual_tools_.publishLine(utils::tensorToVector(pfm->env.points_on_l1_[i]), utils::tensorToVector(pfm->env.points_on_l2_[i]), rviz_visual_tools::CYAN);
    }
  }
  boost::shared_ptr<ReinforcementLearningAgent> rl = boost::dynamic_pointer_cast<ReinforcementLearningAgent>(control_force_calculator_);
  if (rl) {
    const EpisodeContext& ec = rl->episode_context_;
    Vector3d start_bb_origin = utils::tensorToVector(ec.start_bb_origin);
    Vector3d start_bb_dims = utils::tensorToVector(ec.start_bb_dims);
    visual_tools_.publishWireframeCuboid(Isometry3d(Translation3d(start_bb_origin + 0.5 * start_bb_dims)), start_bb_dims[0], start_bb_dims[1], start_bb_dims[2],
                                         rviz_visual_tools::TRANSLUCENT);
    Vector3d goal_bb_origin = utils::tensorToVector(ec.goal_bb_origin);
    Vector3d goal_bb_dims = utils::tensorToVector(ec.goal_bb_dims);
    visual_tools_.publishWireframeCuboid(Isometry3d(Translation3d(goal_bb_origin + 0.5 * goal_bb_dims)), goal_bb_dims[0], goal_bb_dims[1], goal_bb_dims[2],
                                         rviz_visual_tools::TRANSLUCENT);
  }
  visual_tools_.trigger();
}
}  // namespace control_force_provider::backend