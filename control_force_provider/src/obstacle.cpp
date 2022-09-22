#include "control_force_provider/obstacle.h"

#include <cmath>
#include <filesystem>
#include <sstream>
#include <vector>

#include "control_force_provider/utils.h"

using namespace Eigen;
using namespace control_force_provider::utils;
namespace control_force_provider::backend {
void Obstacle::reset(double offset) {
  ros::Time now = ros::Time::now();
  offset = std::min(offset, now.toSec());
  start_time = ros::Time::now() - ros::Duration(offset);
}

Vector4d Obstacle::getPosition() { return getPositionAt(ros::Time(0) + (ros::Time::now() - start_time)); }

WaypointsObstacle::WaypointsObstacle(const YAML::Node& config, const std::string& id) : Obstacle(id), speed_(getConfigValue<double>(config, "speed")[0]) {
  std::vector<double> waypoints_raw = getConfigValue<double>(config, "waypoints");
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
    double duration = length / speed_;
    segments_durations_.push_back(duration);
    total_duration_ += duration;
  }
  std::vector<double> rcm_raw = getConfigValue<double>(config, "rcm");
  setRCM({rcm_raw[0], rcm_raw[1], rcm_raw[2]});
}

Vector4d WaypointsObstacle::getPositionAt(const ros::Time& ros_time) {
  double time = fmod(ros_time.toSec(), total_duration_);
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
  return {position3d[0], position3d[1], position3d[2], 0};
}

FramesObstacle::FramesObstacle(const std::string& id) : Obstacle(id) { iter_ = frames_.begin(); }

void FramesObstacle::setFrames(std::map<double, Affine3d> frames) { frames_ = std::move(frames); }

Vector4d FramesObstacle::getPositionAt(const ros::Time& ros_time) {
  double seconds = ros_time.toSec();
  auto start_iter = iter_;
  do {
    auto current_iter = iter_++;
    if (iter_ == frames_.end()) {
      if (current_iter->first < seconds) {  // reached end
        iter_--;
        const Vector3d& out = current_iter->second.translation();
        return Vector4d(out[0], out[1], out[2], 0);
      }
      iter_ = frames_.begin();
      continue;
    }
    auto next_iter = iter_;
    const double& t1 = current_iter->first;
    const double& t2 = next_iter->first;
    if (t1 < seconds && t2 > seconds) {
      const Vector3d& p1 = current_iter->second.translation();
      const Vector3d& p2 = next_iter->second.translation();
      iter_--;
      // linear interpolation
      const Vector3d& out = p1 + (((seconds - t1) / (t2 - t1)) * (p2 - p1));
      return Vector4d(out[0], out[1], out[2], 0);
    }
  } while (iter_ != start_iter);
}

ObstacleLoader::ObstacleLoader(std::vector<boost::shared_ptr<FramesObstacle>> obstacles, const std::string& path, int reference_obstacle)
    : obstacles_(std::move(obstacles)), reference_obstacle_(reference_obstacle) {
  if (obstacles_.empty()) return;
  if (std::filesystem::is_regular_file(path)) csv_files_.push_back(path);
  if (csv_files_.empty()) throw CSVError("No .csv files found in " + path);
  std::vector<std::map<double, Affine3d>> frames = parseFile(csv_files_[0]);
  updateObstacles(std::move(frames));
}

std::vector<std::map<double, Affine3d>> ObstacleLoader::parseFile(const std::string& file) {
  std::vector<std::map<double, Affine3d>> out(obstacles_.size());
  std::string content = utils::readFile(file);
  std::istringstream line_stream(content);
  std::string buffer;
  double start_time;
  size_t line = 0;
  while (std::getline(line_stream, buffer)) {
    std::istringstream token_stream(buffer);
    std::vector<std::string> tokens;
    while (std::getline(token_stream, buffer, delimiter)) tokens.push_back(buffer);
    double time = std::stod(tokens[0]);
    if (line == 0) start_time = time;
    time = (time - start_time);
    for (size_t i = 0; i < obstacles_.size(); i++) {
      unsigned int j = 9 * i + 1;
      if (j + 8 >= tokens.size()) throw CSVError("Expected " + std::to_string(obstacles_.size()) + " obstacles in " + file);
      Quaterniond rotation(std::stod(tokens[j + 4]), std::stod(tokens[j + 1]), std::stod(tokens[j + 2]), std::stod(tokens[j + 3]));
      Vector3d position(std::stod(tokens[j + 5]), std::stod(tokens[j + 6]), std::stod(tokens[j + 7]));
      position *= 1e-3;
      out[i][time] = Translation3d(position) * rotation;
    }
    line++;
  }
  if (out[0].empty()) throw CSVError("Could not parse " + file);
  ROS_INFO_STREAM_NAMED("control_force_provider/obstacle_loader", "Loaded CSV: " << file);
  return out;
}

Vector3d ObstacleLoader::estimateRCM(const std::map<double, Eigen::Affine3d>& frames) {
  boost::random::uniform_int_distribution<> index_sampler(0, frames.size() - 1);
  std::vector<int> indices;
  for (size_t i = 0; i < rcm_estimation_sample_num; i++) indices.push_back(index_sampler(rng_));
  std::sort(indices.begin(), indices.end(), std::less<>());
  std::vector<Affine3d> poses;
  int index = 0;
  auto index_iter = indices.begin();
  for (auto& frame : frames) {
    if (*index_iter == index) poses.emplace_back(frame.second);
    while (*index_iter <= index) index_iter++;
    index++;
  }
  std::vector<Vector3d> points;
  for (auto poses_iter = poses.begin(); poses_iter != poses.end(); poses_iter++) {
    const Affine3d& p1 = *poses_iter;
    for (auto poses_iter2 = poses_iter; poses_iter2 != poses.end(); poses_iter2++) {
      if (poses_iter2 != poses_iter) {
        const Affine3d& p2 = *poses_iter2;
        double t, s = 0;
        const Vector3d& a1 = p1.translation();
        Vector3d b1 = (p1 * Translation3d(0, 0, 1)).translation() - a1;
        const Vector3d& a2 = p2.translation();
        Vector3d b2 = (p2 * Translation3d(0, 0, 1)).translation() - a2;
        utils::shortestLine(a1, b1, a2, b2, t, s);
        points.emplace_back(a1 + t * b1);
        points.emplace_back(a2 + s * b2);
      }
    }
  }
  Vector3d rcm = Vector3d::Zero();
  for (Vector3d& point : points) rcm += point;
  rcm /= points.size();
  Vector3d var = Vector3d::Zero();
  for (Vector3d& point : points) var += point - rcm;
  var /= points.size();
  if (std::abs(var[0]) > rcm_estimation_max_var || std::abs(var[1]) > rcm_estimation_max_var || std::abs(var[2]) > rcm_estimation_max_var)
    ROS_WARN_STREAM_NAMED("control_force_provider", "Point distribution for RCM estimation exceeds maximum variance.");
  return rcm;
}

void ObstacleLoader::updateObstacles(std::vector<std::map<double, Affine3d>> frames) {
  Vector3d reference_rcm, rcm_translation;
  bool translate = reference_obstacle_ >= 0;
  if (translate) {
    reference_rcm = estimateRCM(frames[reference_obstacle_]);
    rcm_translation = obstacles_[reference_obstacle_]->getRCM() - reference_rcm;
  }
  for (size_t i = 0; i < frames.size(); i++) {
    Vector3d rcm = i == reference_obstacle_ ? reference_rcm : estimateRCM(frames[i]);
    if (translate) {
      for (auto& frame : frames[i]) frame.second.pretranslate(rcm_translation);
      rcm += rcm_translation;
    }
    obstacles_[i]->setFrames(std::move(frames[i]));
    obstacles_[i]->setRCM(rcm);
  }
}
}  // namespace control_force_provider::backend