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
  Eigen::Vector3d bb_dims;
  double block_width;
  double bb_width;
  double bb_length;
  rviz_visual_tools::RvizVisualTools visual_tools_;
  ros::NodeHandle& node_handle_;
  const boost::shared_ptr<Environment> environment_;
  const boost::shared_ptr<EpisodeContext> episode_context_;
  ros::Timer timer_;
  void callback(const ros::TimerEvent& event);
  Eigen::Vector3d position(int index) const;

 public:
  Visualizer(ros::NodeHandle& node_handle, boost::shared_ptr<Environment> environment, boost::shared_ptr<EpisodeContext> episode_context);
  ~Visualizer() = default;
};
}  // namespace control_force_provider::backend