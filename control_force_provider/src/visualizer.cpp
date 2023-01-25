#include "control_force_provider/visualizer.h"

#include <eigen_conversions/eigen_msg.h>
#include <torch/torch.h>

#include <Eigen/Dense>
#include <boost/pointer_cast.hpp>
#include <thread>

using namespace Eigen;
namespace control_force_provider::backend {
Visualizer::Visualizer(ros::NodeHandle& node_handle, boost::shared_ptr<Environment> environment, boost::shared_ptr<EpisodeContext> episode_context,
                       unsigned int thread_count)
    : node_handle_(node_handle),
      environment_(environment),
      episode_context_(episode_context),
      visual_tools_("world", "/visualization_marker_array", node_handle),
      timer_(node_handle_.createTimer(ros::Duration(0.05), &Visualizer::callback, this)),
      thread_count_(thread_count) {
  visual_tools_.enableBatchPublishing();
  bb_dims = utils::tensorToVector(environment_->getWorkspaceBbDims());
  bb_width = bb_dims[0] + WS_SPACE;
  bb_length = bb_dims[1] + WS_SPACE;
  block_width = 100 * bb_width + BLOCK_SPACE - WS_SPACE;
}

Vector3d Visualizer::position(int index) const {
  double block_offset = (index / 1000) * block_width;
  int block_i = index % 1000;
  int row = block_i / 100;
  int col = block_i % 100;
  return Vector3d(block_offset + row * bb_width, col * bb_length, 0);
}

void Visualizer::callback(const ros::TimerEvent& event) {
  visual_tools_.deleteAllMarkers();
  torch::Tensor offset;
  {
    auto lock = environment_->getOffsetLock();
    offset = environment_->getOffset().clone();
  }
  Vector3d workspace_bb_origin;
  Vector3d workspace_bb_dims = utils::tensorToVector(environment_->getWorkspaceBbDims());
  {
    auto lock = environment_->getWorkspaceBbOriginLock();
    workspace_bb_origin = utils::tensorToVector(environment_->getWorkspaceBbOrigin() + offset);
  }
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
  }
  torch::Tensor goal;
  {
    auto lock = environment_->getGoalLock();
    goal = environment_->getGoal() + offset;
  }
  int ob_num = environment_->getObstacles().size();
  Vector3d start_bb_origin;
  Vector3d start_bb_dims;
  Vector3d goal_bb_origin;
  Vector3d goal_bb_dims;
  if (episode_context_) {
    start_bb_origin = utils::tensorToVector(episode_context_->start_bb_origin);
    start_bb_dims = utils::tensorToVector(episode_context_->start_bb_dims);
    goal_bb_origin = utils::tensorToVector(episode_context_->goal_bb_origin);
    goal_bb_dims = utils::tensorToVector(episode_context_->goal_bb_dims);
  }
  int batch_size = ee_position.size(0);
  auto runnable = [&](int index) {
    // obstacles
    {
      auto ob_pos_lock = environment_->getObPositionsLock();
      auto ob_rcm_lock = environment_->getObRCMsLock();
      const std::vector<torch::Tensor>& ob_positions = environment_->getObPositions();
      const std::vector<torch::Tensor>& ob_rcms = environment_->getObRCMs();
      for (size_t i = 0; i < ob_num; i++) {
        if (!boost::dynamic_pointer_cast<DummyObstacle>(environment_->getObstacles()[i])) {
          for (size_t j = index; j < batch_size; j += thread_count_) {
            Vector3d pos = position(j);
            Vector3d obstacle_rcm = utils::tensorToVector(ob_rcms[i] + offset) + pos;
            Vector3d obstacle_pos = utils::tensorToVector(ob_positions[i][j] + offset) + pos;
            boost::lock_guard<boost::mutex> lock(visualizer_mtx_);
            visual_tools_.publishSphere(obstacle_rcm, rviz_visual_tools::BROWN);
            visual_tools_.publishLine(obstacle_rcm, obstacle_pos, rviz_visual_tools::BROWN);
          }
        }
      }
    }
    // control_force_calculator
    for (size_t i = index; i < batch_size; i += thread_count_) {
      Vector3d pos = position(i);
      Vector3d ee_position_eigen = utils::tensorToVector(ee_position[i]) + pos;
      Vector3d ee_velocity_eigen = utils::tensorToVector(ee_velocity[i]);
      Vector3d rcm_eigen = utils::tensorToVector(robot_rcm) + pos;
      Vector3d goal_eigen = utils::tensorToVector(goal[i]) + pos;
      geometry_msgs::Point ee_position_msg;
      tf::pointEigenToMsg(ee_position_eigen, ee_position_msg);
      geometry_msgs::Point target_msg;
      double scale = std::max(ee_velocity_eigen.norm() / environment_->getMaxForce() * 0.15, 0.05);
      tf::pointEigenToMsg(ee_position_eigen + ee_velocity_eigen.normalized() * scale, target_msg);
      boost::lock_guard<boost::mutex> lock(visualizer_mtx_);
      visual_tools_.publishLine(rcm_eigen, ee_position_eigen, rviz_visual_tools::PURPLE);
      visual_tools_.publishSphere(goal_eigen, rviz_visual_tools::BLUE);
      visual_tools_.publishArrow(ee_position_msg, target_msg, rviz_visual_tools::TRANSLUCENT_LIGHT, rviz_visual_tools::SMALL);
      visual_tools_.publishWireframeCuboid(Isometry3d(Translation3d(workspace_bb_origin + pos + 0.5 * workspace_bb_dims)), workspace_bb_dims[0],
                                           workspace_bb_dims[1], workspace_bb_dims[2], rviz_visual_tools::DARK_GREY);
    }
    {
      auto l1_lock = environment_->getPointsOnL1Lock();
      auto l2_lock = environment_->getPointsOnL2Lock();
      const std::vector<torch::Tensor>& points_on_l1 = environment_->getPointsOnL1();
      const std::vector<torch::Tensor>& points_on_l2 = environment_->getPointsOnL2();
      for (size_t i = 0; i < ob_num; i++) {
        if (!boost::dynamic_pointer_cast<DummyObstacle>(environment_->getObstacles()[i])) {
          for (size_t j = index; j < batch_size; j += thread_count_) {
            Vector3d pos = position(j);
            Vector3d point_on_l1 = utils::tensorToVector(points_on_l1[i][j] + offset) + pos;
            Vector3d point_on_l2 = utils::tensorToVector(points_on_l2[i][j] + offset) + pos;
            double color_gradient = std::min((point_on_l2 - point_on_l1).norm() / OB_DISTANCE_BOUND, 1.);
            boost::lock_guard<boost::mutex> lock(visualizer_mtx_);
            visual_tools_.publishLine(point_on_l1, point_on_l2, visual_tools_.getColorScale(color_gradient));
          }
        }
      }
    }
    if (episode_context_) {
      for (size_t i = index; i < batch_size; i += thread_count_) {
        Vector3d pos = position(i);
        boost::lock_guard<boost::mutex> lock(visualizer_mtx_);
        visual_tools_.publishWireframeCuboid(Isometry3d(Translation3d(start_bb_origin + pos + 0.5 * start_bb_dims)), start_bb_dims[0], start_bb_dims[1],
                                             start_bb_dims[2], rviz_visual_tools::TRANSLUCENT);
        visual_tools_.publishWireframeCuboid(Isometry3d(Translation3d(goal_bb_origin + pos + 0.5 * goal_bb_dims)), goal_bb_dims[0], goal_bb_dims[1],
                                             goal_bb_dims[2], rviz_visual_tools::TRANSLUCENT);
      }
    }
    if (!custom_marker_.empty()) {
      boost::shared_lock_guard<boost::shared_mutex> cm_lock(custom_marker_mtx_);
      for (auto& tuple : custom_marker_) {
        torch::Tensor marker = tuple.second + offset;
        switch (marker.size(1)) {
          case 3:  // point
            for (size_t i = index; i < batch_size; i += thread_count_) {
              Vector3d pos = position(i);
              Vector3d point = utils::tensorToVector(marker[i]);
              boost::lock_guard<boost::mutex> lock(visualizer_mtx_);
              visual_tools_.publishSphere(point + pos, rviz_visual_tools::WHITE);
            }
            break;
          case 6:  // line
          {
            torch::Tensor points1 = marker.slice(1, 0, 3);
            torch::Tensor points2 = marker.slice(1, 3, 6);
            for (size_t i = index; i < batch_size; i += thread_count_) {
              Vector3d pos = position(i);
              Vector3d point1 = utils::tensorToVector(points1[i]);
              Vector3d point2 = utils::tensorToVector(points2[i]);
              boost::lock_guard<boost::mutex> lock(visualizer_mtx_);
              visual_tools_.publishLine(point1, point2, rviz_visual_tools::WHITE);
            }
          } break;
          default:
            break;
        }
      }
    }
  };
  std::vector<std::thread> threads;
  for (size_t i = 0; i < thread_count_; i++) threads.emplace_back(runnable, i);
  for (auto& thread : threads) thread.join();
  visual_tools_.trigger();
}

void Visualizer::setCustomMarker(const std::string& key, const torch::Tensor& marker) {
  boost::lock_guard<boost::shared_mutex> lock(custom_marker_mtx_);
  custom_marker_[key] = marker;
}
}  // namespace control_force_provider::backend