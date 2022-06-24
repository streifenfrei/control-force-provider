#include "control_force_provider/obstacle.h"

#include <ros/ros.h>

#include <cmath>
#include <vector>

#include "control_force_provider/utils.h"

using namespace std;
using namespace Eigen;
using namespace control_force_provider::utils;
namespace control_force_provider::backend {
SimulatedObstacle::SimulatedObstacle(const ryml::NodeRef& config) : speed_(getConfigValue<double>(config, "speed")[0]) {
  vector<double> waypoints_raw = getConfigValue<double>(config, "waypoints");
  unsigned int waypoints_raw_length = waypoints_raw.size() - (waypoints_raw.size() % 3);
  for (size_t i = 0; i < waypoints_raw_length; i += 3) {
    waypoints_.emplace_back(waypoints_raw[i], waypoints_raw[i + 1], waypoints_raw[i + 2]);
  }
  total_duration_ = 0;
  for (size_t i = 0; i < waypoints_.size(); i++) {
    unsigned int j = i + 1 == waypoints_.size() ? 0 : i + 1;
    Vector3d segment = waypoints_[j] - waypoints_[i];
    segments_.push_back(segment);
    segments_normalized_.push_back(segment.normalized());
    double length = segments_[i].norm();
    segments_lengths_.push_back(length);
    double duration = speed_ / length;
    segments_durations_.push_back(duration);
    total_duration_ += duration;
  }
  vector<double> rcm_raw = getConfigValue<double>(config, "rcm");
  rcm_ = {rcm_raw[0], rcm_raw[1], rcm_raw[2]};
}

void SimulatedObstacle::getPosition(Vector4d& position) {
  double time = fmod(ros::Time::now().toSec(), total_duration_);
  unsigned int segment;
  double duration_sum = 0;
  for (size_t i = 0; i < segments_durations_.size(); i++) {
    duration_sum += segments_durations_[i];
    if (duration_sum > time) {
      segment = i;
      break;
    }
  }
  double position_on_segment = 1 - (duration_sum - time) / segments_durations_[segment];
  double length_on_segment = segments_lengths_[segment] * position_on_segment;
  Vector3d segment_part = segments_normalized_[segment] * length_on_segment;
  Vector3d position3d = waypoints_[segment] + segment_part;
  position = {position3d[0], position3d[1], position3d[2], 0};
}
}  // namespace control_force_provider::backend