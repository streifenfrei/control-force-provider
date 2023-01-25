#pragma once

#include <ros/ros.h>
#include <rviz_visual_tools/rviz_visual_tools.h>

#include <boost/shared_ptr.hpp>

#include "control_force_calculator.h"
#include "obstacle.h"

namespace control_force_provider::backend {
class Visualizer {
 private:
  static constexpr double WS_SPACE = 0.1;
  static constexpr double BLOCK_SPACE = 1;
  const unsigned int thread_count_;
  Eigen::Vector3d bb_dims;
  double block_width;
  double bb_width;
  double bb_length;
  rviz_visual_tools::RvizVisualTools visual_tools_;
  ros::NodeHandle& node_handle_;
  const boost::shared_ptr<Environment> environment_;
  const boost::shared_ptr<EpisodeContext> episode_context_;
  std::map<std::string, torch::Tensor> custom_marker_;
  ros::Timer timer_;
  boost::mutex visualizer_mtx_;
  boost::shared_mutex custom_marker_mtx_;

  void callback(const ros::TimerEvent& event);
  Eigen::Vector3d position(int index) const;

 public:
  Visualizer(ros::NodeHandle& node_handle, boost::shared_ptr<Environment> environment, boost::shared_ptr<EpisodeContext> episode_context,
             unsigned int thread_count = 1);
  void setCustomMarker(const std::string& key, const torch::Tensor& marker);
  ~Visualizer() = default;
};
}  // namespace control_force_provider::backend