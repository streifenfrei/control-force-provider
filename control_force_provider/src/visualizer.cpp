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

  torch::Tensor offset;
  {
    auto lock = control_force_calculator_->env.getOffsetLock();
    offset = control_force_calculator_->env.getOffset().clone();
  }
  Vector3d workspace_bb_origin;
  Vector3d workspace_bb_dims = utils::tensorToVector(control_force_calculator_->env.getWorkspaceBbDims());
  visual_tools_.deleteAllMarkers();
  {
    auto lock = control_force_calculator_->env.getWorkspaceBbOriginLock();
    workspace_bb_origin = utils::tensorToVector(control_force_calculator_->env.getWorkspaceBbOrigin() + offset);
  }
  // obstacles
  {
    auto lock1 = control_force_calculator_->env.getObPositionsLock();
    auto lock2 = control_force_calculator_->env.getObRCMsLock();
    const std::vector<torch::Tensor>& ob_positions = control_force_calculator_->env.getObPositions();
    const std::vector<torch::Tensor>& ob_rcms = control_force_calculator_->env.getObRCMs();
    for (size_t i = 0; i < control_force_calculator_->env.getObstacles().size(); i++) {
      Vector3d obstacle_rcm = utils::tensorToVector(ob_rcms[i] + offset);
      Vector3d obstacle_pos = utils::tensorToVector(ob_positions[i] + offset);
      visual_tools_.publishSphere(obstacle_rcm, rviz_visual_tools::BROWN);
      visual_tools_.publishLine(obstacle_rcm, obstacle_pos, rviz_visual_tools::BROWN);
    }
  }
  // control_force_calculator
  torch::Tensor ee_position;
  {
    auto lock = control_force_calculator_->env.getEePositionLock();
    ee_position = control_force_calculator_->env.getEePosition() + offset;
  }
  torch::Tensor ee_velocity;
  {
    auto lock = control_force_calculator_->env.getEeVelocityLock();
    ee_velocity = control_force_calculator_->env.getEeVelocity();
  }
  torch::Tensor robot_rcm;
  {
    auto lock = control_force_calculator_->env.getRCMLock();
    robot_rcm = control_force_calculator_->env.getRCM() + offset;
  }
  torch::Tensor goal;
  {
    auto lock = control_force_calculator_->env.getGoalLock();
    goal = control_force_calculator_->getGoal() + offset;
  }
  Vector3d ee_position_eigen = utils::tensorToVector(ee_position);
  Vector3d ee_velocity_eigen = utils::tensorToVector(ee_velocity);
  Vector3d rcm_eigen = utils::tensorToVector(robot_rcm);
  Vector3d goal_eigen = utils::tensorToVector(goal);
  visual_tools_.publishLine(rcm_eigen, ee_position_eigen, rviz_visual_tools::PURPLE);
  visual_tools_.publishSphere(goal_eigen, rviz_visual_tools::BLUE);
  geometry_msgs::Point ee_position_msg;
  tf::pointEigenToMsg(ee_position_eigen, ee_position_msg);
  geometry_msgs::Point target_msg;
  double scale = std::max(ee_velocity_eigen.norm() / control_force_calculator_->env.getMaxForce() * 0.15, 0.05);
  tf::pointEigenToMsg(ee_position_eigen + ee_velocity_eigen.normalized() * scale, target_msg);
  visual_tools_.publishArrow(ee_position_msg, target_msg, rviz_visual_tools::TRANSLUCENT_LIGHT, rviz_visual_tools::SMALL);
  visual_tools_.publishWireframeCuboid(Isometry3d(Translation3d(workspace_bb_origin + 0.5 * workspace_bb_dims)), workspace_bb_dims[0], workspace_bb_dims[1],
                                       workspace_bb_dims[2], rviz_visual_tools::DARK_GREY);

  boost::shared_ptr<PotentialFieldMethod> pfm = boost::dynamic_pointer_cast<PotentialFieldMethod>(control_force_calculator_);
  if (pfm) {
    auto lock1 = pfm->env.getPointsOnL1Lock();
    auto lock2 = pfm->env.getPointsOnL2Lock();
    const std::vector<torch::Tensor>& points_on_l1 = pfm->env.getPointsOnL1();
    const std::vector<torch::Tensor>& points_on_l2 = pfm->env.getPointsOnL2();
    for (size_t i = 0; i < control_force_calculator_->env.getObstacles().size(); i++) {
      Vector3d point_on_l1 = utils::tensorToVector(points_on_l1[i]);
      Vector3d point_on_l2 = utils::tensorToVector(points_on_l2[i]);
      double distance = (point_on_l2 - point_on_l1).norm();
      if (distance < pfm->repulsion_distance_) visual_tools_.publishLine(point_on_l1, point_on_l2, rviz_visual_tools::CYAN);
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