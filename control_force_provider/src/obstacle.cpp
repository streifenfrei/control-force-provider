#include "control_force_provider/obstacle.h"

#include <cmath>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <sstream>
#include <vector>

#include "control_force_provider/utils.h"

using namespace Eigen;
using namespace control_force_provider::utils;
namespace control_force_provider::backend {
std::vector<boost::shared_ptr<Obstacle>> Obstacle::createFromConfig(const YAML::Node& config, std::string& data_path, int batch_size) {
  YAML::Node obstacle_configs = getConfigValue<YAML::Node>(config, "obstacles")[0];
  // load obstacles
  std::vector<boost::shared_ptr<Obstacle>> obstacles;
  for (YAML::const_iterator it = obstacle_configs.begin(); it != obstacle_configs.end(); it++) {
    std::string id = it->first.as<std::string>();
    if (id == "data") {
      data_path = it->second.as<std::string>();
    } else {
      const YAML::Node& ob_config = it->second;
      std::string ob_type = getConfigValue<std::string>(ob_config, "type")[0];
      if (ob_type == "dummy") {
        obstacles.push_back(boost::static_pointer_cast<Obstacle>(boost::make_shared<DummyObstacle>(id, batch_size)));
      } else if (ob_type == "waypoints") {
        obstacles.push_back(boost::static_pointer_cast<Obstacle>(boost::make_shared<WaypointsObstacle>(ob_config, id, batch_size)));
      } else if (ob_type == "csv") {
        boost::shared_ptr<FramesObstacle> obstacle = boost::make_shared<FramesObstacle>(id, batch_size);
        if (ob_config["rcm"].IsDefined()) {
          obstacle->setRCM(utils::createTensor(utils::getConfigValue<double>(ob_config, "rcm"), 0, 3));
        };
        obstacles.push_back(boost::static_pointer_cast<Obstacle>(obstacle));
      } else
        throw ConfigError("Unknown obstacle type '" + ob_type + "'");
    }
  }
  return obstacles;
}

void Obstacle::reset(const torch::Tensor& mask, const torch::Tensor& offset) {
  torch::Tensor now = torch::full(1, Time::now(), utils::getTensorOptions());
  torch::Tensor offset_ = torch::min(offset, now);
  start_time = torch::where(mask, now - offset, start_time);
}

void Obstacle::reset(const torch::Tensor& offset) { this->reset(torch::ones_like(start_time), offset); }

torch::Tensor Obstacle::getPosition() { return getPositionAt((torch::full_like(start_time, Time::now(), utils::getTensorOptions()) - start_time)); }

WaypointsObstacle::WaypointsObstacle(const YAML::Node& config, const std::string& id, int batch_size)
    : Obstacle(id, batch_size), speed_(getConfigValue<double>(config, "speed")[0]) {
  std::vector<double> waypoints_raw = getConfigValue<double>(config, "waypoints");
  unsigned int waypoints_raw_length = waypoints_raw.size() - (waypoints_raw.size() % 3);
  for (size_t i = 0; i < waypoints_raw_length; i += 3) {
    waypoints_.push_back(utils::createTensor({waypoints_raw[i], waypoints_raw[i + 1], waypoints_raw[i + 2]}));
  }
  total_duration_ = 0;
  for (size_t i = 0; i < waypoints_.size(); i++) {
    unsigned int j = i + 1 == waypoints_.size() ? 0 : i + 1;
    torch::Tensor segment = waypoints_[j] - waypoints_[i];
    segments_.push_back(segment);
    torch::Scalar norm = utils::norm(segment).item();
    segments_normalized_.push_back(segment / norm);
    double length = norm.toDouble();
    segments_lengths_.push_back(length);
    double duration = length / speed_;
    segments_durations_.push_back(duration);
    total_duration_ += duration;
  }
  std::vector<double> rcm_raw = getConfigValue<double>(config, "rcm");
  setRCM(utils::createTensor({rcm_raw[0], rcm_raw[1], rcm_raw[2]}));
}

torch::Tensor WaypointsObstacle::getPositionAt(torch::Tensor time) {
  time = torch::fmod(time, total_duration_);
  torch::Tensor duration_sum = torch::zeros_like(time, utils::getTensorOptions());
  torch::Tensor found = torch::zeros_like(time, torch::kBool);
  torch::Tensor segments_durations = torch::empty_like(time, utils::getTensorOptions());
  torch::Tensor segments_lengths = torch::empty_like(time, utils::getTensorOptions());
  torch::Tensor segments_normalized = torch::empty({time.size(0), 3}, utils::getTensorOptions());
  torch::Tensor waypoints = torch::zeros({time.size(0), 3}, utils::getTensorOptions());
  for (size_t i = 0; i < segments_durations_.size(); i++) {
    duration_sum = torch::where(found, duration_sum, duration_sum + segments_durations_[i]);
    torch::Tensor mask = torch::logical_and(torch::logical_not(found), duration_sum > time);
    found = torch::logical_or(mask, found);
    segments_durations = segments_durations.masked_fill(mask, segments_durations_[i]);
    segments_lengths = torch::where(mask, segments_lengths_[i], segments_lengths);
    segments_normalized = torch::where(mask, segments_normalized_[i], segments_normalized);
    waypoints = torch::where(mask, waypoints_[i], waypoints);
  }
  torch::Tensor position_on_segment = 1 - (duration_sum - time) / segments_durations;
  torch::Tensor length_on_segment = segments_lengths * position_on_segment;
  torch::Tensor segment_part = segments_normalized * length_on_segment;
  return waypoints + segment_part;
}

FramesObstacle::FramesObstacle(const std::string& id, int batch_size) : Obstacle(id, batch_size) { iter_ = frames_.begin(); }

void FramesObstacle::setFrames(std::map<double, Affine3d> frames) {
  frames_ = std::move(frames);
  iter_ = frames_.begin();
}

torch::Tensor FramesObstacle::getPositionAt(torch::Tensor time) {
  // TODO: parallelize
  auto start_iter = iter_;
  do {
    auto current_iter = iter_++;
    if (iter_ == frames_.end()) {
      if (current_iter->first < time.item().toDouble()) {  // reached end
        iter_--;
        return utils::vectorToTensor(current_iter->second.translation());
      }
      iter_ = frames_.begin();
      continue;
    }
    auto next_iter = iter_;
    const double& t1 = current_iter->first;
    const double& t2 = next_iter->first;
    if (t1 < time.item().toDouble() && t2 > time.item().toDouble()) {
      const Vector3d& p1 = current_iter->second.translation();
      const Vector3d& p2 = next_iter->second.translation();
      iter_--;
      // linear interpolation
      return utils::vectorToTensor(p1 + (((time.item().toDouble() - t1) / (t2 - t1)) * (p2 - p1)));
    }
  } while (iter_ != start_iter);
}

ObstacleLoader::ObstacleLoader(std::vector<boost::shared_ptr<FramesObstacle>> obstacles, const std::string& path, int reference_obstacle)
    : obstacles_(std::move(obstacles)), reference_obstacle_(reference_obstacle) {
  if (obstacles_.empty()) return;
  if (isValidFile(path))
    csv_files_.push_back(path);
  else if (fs::is_directory(path)) {
    for (auto& file : fs::recursive_directory_iterator(path))
      if (isValidFile(file.path().string())) csv_files_.push_back(file.path().string());
  }
  if (csv_files_.empty()) throw CSVError("No .csv files found in " + path);
  files_iter_ = csv_files_.begin();
  loadNext();
}

bool ObstacleLoader::isValidFile(const std::string& file) { return fs::is_regular_file(file) && file.substr(file.size() - 4, 4) == ".csv"; }

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

torch::Tensor ObstacleLoader::estimateRCM(const std::map<double, Eigen::Affine3d>& frames) {
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
        torch::Tensor t, s = torch::zeros(1, utils::getTensorOptions());
        const Vector3d& a1 = p1.translation();
        Vector3d b1 = (p1 * Translation3d(0, 0, 1)).translation() - a1;
        const Vector3d& a2 = p2.translation();
        Vector3d b2 = (p2 * Translation3d(0, 0, 1)).translation() - a2;
        utils::shortestLine(utils::vectorToTensor(a1), utils::vectorToTensor(b1), utils::vectorToTensor(a2), utils::vectorToTensor(b2), t, s);
        if (!std::isnan(t.item().toDouble())) points.emplace_back(a1 + t.item().toDouble() * b1);
        if (!std::isnan(s.item().toDouble())) points.emplace_back(a2 + s.item().toDouble() * b2);
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
  return utils::vectorToTensor(rcm);
}

void ObstacleLoader::updateObstacles(std::vector<std::map<double, Affine3d>> frames) {
  torch::Tensor reference_rcm, rcm_translation;
  bool translate = reference_obstacle_ >= 0;
  if (translate) {
    reference_rcm = estimateRCM(frames[reference_obstacle_]);
    rcm_translation = obstacles_[reference_obstacle_]->getRCM() - reference_rcm;
  }
  for (size_t i = 0; i < frames.size(); i++) {
    torch::Tensor rcm = i == reference_obstacle_ ? reference_rcm : estimateRCM(frames[i]);
    if (translate) {
      for (auto& frame : frames[i]) frame.second.pretranslate((Vector3d)utils::tensorToVector(rcm_translation));
      rcm += rcm_translation;
    }
    obstacles_[i]->setFrames(std::move(frames[i]));
    obstacles_[i]->setRCM(rcm);
  }
}

void ObstacleLoader::loadNext() {
  if (obstacles_.empty() || csv_files_.empty()) return;
  std::vector<std::map<double, Affine3d>> frames = parseFile(*files_iter_);
  updateObstacles(std::move(frames));
  files_iter_++;
  if (files_iter_ == csv_files_.end()) files_iter_ = csv_files_.begin();
}
}  // namespace control_force_provider::backend