#pragma once

#include <ros/ros.h>
#include <rviz_visual_tools/rviz_visual_tools.h>

#include <boost/shared_ptr.hpp>

#include "control_force_calculator.h"
#include "obstacle.h"

namespace control_force_provider::backend {
class Visualizer {
 private:
  rviz_visual_tools::RvizVisualTools visual_tools_;
  ros::NodeHandle& node_handle_;
  const boost::shared_ptr<Environment> environment_;
  const boost::shared_ptr<EpisodeContext> episode_context_;
  ros::Timer timer_;
  void callback(const ros::TimerEvent& event);

 public:
  Visualizer(ros::NodeHandle& node_handle, boost::shared_ptr<Environment> environment, boost::shared_ptr<EpisodeContext> episode_context);
  ~Visualizer() = default;
};
}  // namespace control_force_provider::backend