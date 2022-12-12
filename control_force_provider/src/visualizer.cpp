#include "control_force_provider/visualizer.h"

#include <eigen_conversions/eigen_msg.h>
#include <torch/torch.h>

#include <Eigen/Dense>
#include <boost/pointer_cast.hpp>

using namespace Eigen;
namespace control_force_provider::backend {
Visualizer::Visualizer(ros::NodeHandle& node_handle, boost::shared_ptr<Environment> environment, boost::shared_ptr<EpisodeContext> episode_context)
    : node_handle_(node_handle),
      environment_(environment),
      episode_context_(episode_context),
      visual_tools_("world", "/visualization_marker_array", node_handle),
      timer_(node_handle_.createTimer(ros::Duration(0.05), &Visualizer::callback, this)) {}

void Visualizer::callback(const ros::TimerEvent& event) {
  torch::Tensor offset;
  {
    auto lock = environment_->getOffsetLock();
    offset = environment_->getOffset().clone();
  }
  Vector3d workspace_bb_origin;
  Vector3d workspace_bb_dims = utils::tensorToVector(environment_->getWorkspaceBbDims());
  visual_tools_.deleteAllMarkers();
  {
    auto lock = environment_->getWorkspaceBbOriginLock();
    workspace_bb_origin = utils::tensorToVector(environment_->getWorkspaceBbOrigin() + offset);
  }
  // obstacles
  {
    auto lock1 = environment_->getObPositionsLock();
    auto lock2 = environment_->getObRCMsLock();
    const std::vector<torch::Tensor>& ob_positions = environment_->getObPositions();
    const std::vector<torch::Tensor>& ob_rcms = environment_->getObRCMs();
    for (size_t i = 0; i < environment_->getObstacles().size(); i++) {
      Vector3d obstacle_rcm = utils::tensorToVector(ob_rcms[i] + offset);
      Vector3d obstacle_pos = utils::tensorToVector(ob_positions[i] + offset);
      visual_tools_.publishSphere(obstacle_rcm, rviz_visual_tools::BROWN);
      visual_tools_.publishLine(obstacle_rcm, obstacle_pos, rviz_visual_tools::BROWN);
    }
  }
  // control_force_calculator
  torch::Tensor ee_position;
  {
    auto lock = environment_->getEePositionLock();
    ee_position = environment_->getEePosition() + offset;
  }
  torch::Tensor ee_velocity;
  {
    auto lock = environment_->getEeVelocityLock();
    ee_velocity = environment_->getEeVelocity();
  }
  torch::Tensor robot_rcm;
  {
    auto lock = environment_->getRCMLock();
    robot_rcm = environment_->getRCM() + offset;
    std::cout << offset << std::endl;
    std::cout << robot_rcm << std::endl;
  }
  torch::Tensor goal;
  {
    auto lock = environment_->getGoalLock();
    goal = environment_->getGoal() + offset;
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
  double scale = std::max(ee_velocity_eigen.norm() / environment_->getMaxForce() * 0.15, 0.05);
  tf::pointEigenToMsg(ee_position_eigen + ee_velocity_eigen.normalized() * scale, target_msg);
  visual_tools_.publishArrow(ee_position_msg, target_msg, rviz_visual_tools::TRANSLUCENT_LIGHT, rviz_visual_tools::SMALL);
  visual_tools_.publishWireframeCuboid(Isometry3d(Translation3d(workspace_bb_origin + 0.5 * workspace_bb_dims)), workspace_bb_dims[0], workspace_bb_dims[1],
                                       workspace_bb_dims[2], rviz_visual_tools::DARK_GREY);

  {
    auto lock1 = environment_->getPointsOnL1Lock();
    auto lock2 = environment_->getPointsOnL2Lock();
    const std::vector<torch::Tensor>& points_on_l1 = environment_->getPointsOnL1();
    const std::vector<torch::Tensor>& points_on_l2 = environment_->getPointsOnL2();
    for (size_t i = 0; i < environment_->getObstacles().size(); i++) {
      Vector3d point_on_l1 = utils::tensorToVector(points_on_l1[i] + offset);
      Vector3d point_on_l2 = utils::tensorToVector(points_on_l2[i] + offset);
      double distance = (point_on_l2 - point_on_l1).norm();
      visual_tools_.publishLine(point_on_l1, point_on_l2, rviz_visual_tools::CYAN);
    }
  }
  if (episode_context_) {
    Vector3d start_bb_origin = utils::tensorToVector(episode_context_->start_bb_origin);
    Vector3d start_bb_dims = utils::tensorToVector(episode_context_->start_bb_dims);
    visual_tools_.publishWireframeCuboid(Isometry3d(Translation3d(start_bb_origin + 0.5 * start_bb_dims)), start_bb_dims[0], start_bb_dims[1], start_bb_dims[2],
                                         rviz_visual_tools::TRANSLUCENT);
    Vector3d goal_bb_origin = utils::tensorToVector(episode_context_->goal_bb_origin);
    Vector3d goal_bb_dims = utils::tensorToVector(episode_context_->goal_bb_dims);
    visual_tools_.publishWireframeCuboid(Isometry3d(Translation3d(goal_bb_origin + 0.5 * goal_bb_dims)), goal_bb_dims[0], goal_bb_dims[1], goal_bb_dims[2],
                                         rviz_visual_tools::TRANSLUCENT);
  }
  visual_tools_.trigger();
}
}  // namespace control_force_provider::backend