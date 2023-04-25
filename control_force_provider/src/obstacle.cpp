#include "control_force_provider/obstacle.h"

#include <boost/filesystem.hpp>
#include <cmath>
namespace fs = boost::filesystem;
#include <sstream>
#include <vector>

#include "control_force_provider/utils.h"

using namespace Eigen;
using namespace control_force_provider::utils;
namespace control_force_provider::backend {
std::vector<boost::shared_ptr<Obstacle>> Obstacle::createFromConfig(const YAML::Node& config, std::string& data_path, int batch_size,
                                                                    torch::DeviceType device) {
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
        obstacles.push_back(boost::static_pointer_cast<Obstacle>(boost::make_shared<DummyObstacle>(id, batch_size, device)));
      } else if (ob_type == "waypoints") {
        obstacles.push_back(boost::static_pointer_cast<Obstacle>(boost::make_shared<WaypointsObstacle>(ob_config, id, batch_size, device)));
      } else if (ob_type == "csv") {
        boost::shared_ptr<FramesObstacle> obstacle = boost::make_shared<FramesObstacle>(id, batch_size, device);
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

void Obstacle::setDevice(torch::DeviceType device) {
  start_time = start_time.to(device);
  rcm_ = rcm_.to(device);
  copyToDevice(device);
  device_ = device;
}

void Obstacle::reset(const torch::Tensor& mask, const torch::Tensor& offset) {
  torch::Tensor now = torch::full(1, Time::now(), utils::getTensorOptions(device_));
  torch::Tensor offset_ = torch::min(offset.to(device_), now);
  start_time = torch::where(mask, now - offset_, start_time);
}

void Obstacle::reset(const torch::Tensor& offset) { this->reset(torch::ones_like(start_time, utils::getTensorOptions(device_)), offset); }

torch::Tensor Obstacle::getPosition() { return getPositionAt((torch::full_like(start_time, Time::now(), utils::getTensorOptions(device_)) - start_time)); }

WaypointsObstacle::WaypointsObstacle(const YAML::Node& config, const std::string& id, int batch_size, torch::DeviceType device)
    : Obstacle(id, batch_size, device), speed_(getConfigValue<double>(config, "speed")[0]) {
  std::vector<double> waypoints_raw = getConfigValue<double>(config, "waypoints");
  unsigned int waypoints_raw_length = waypoints_raw.size() - (waypoints_raw.size() % 3);
  for (size_t i = 0; i < waypoints_raw_length; i += 3) {
    waypoints_.push_back(utils::createTensor({waypoints_raw[i], waypoints_raw[i + 1], waypoints_raw[i + 2]}, 0, -1, device_));
  }
  total_duration_ = 0;
  for (size_t i = 0; i < waypoints_.size(); i++) {
    unsigned int j = i + 1 == waypoints_.size() ? 0 : i + 1;
    torch::Tensor segment = waypoints_[j] - waypoints_[i];
    segments_.push_back(segment);
    torch::Scalar norm = utils::norm(segment).item();
    segments_normalized_.push_back(segment / norm);
    double length = norm.toFloat();
    segments_lengths_.push_back(length);
    double duration = length / speed_;
    segments_durations_.push_back(duration);
    total_duration_ += duration;
  }
  std::vector<double> rcm_raw = getConfigValue<double>(config, "rcm");
  setRCM(utils::createTensor({rcm_raw[0], rcm_raw[1], rcm_raw[2]}, 0, -1, device_));
}

void WaypointsObstacle::copyToDevice(torch::DeviceType device) {
  for (size_t i = 0; i < waypoints_.size(); i++) {
    waypoints_[i] = waypoints_[i].to(device);
    segments_[i] = segments_[i].to(device);
    segments_normalized_[i] = segments_normalized_[i].to(device);
  }
}

torch::Tensor WaypointsObstacle::getPositionAt(torch::Tensor time) {
  if (waypoints_.size() == 1) return torch::expand_copy(waypoints_[0], {time.size(0), 3});
  time = torch::fmod(time.to(device_), total_duration_);
  torch::Tensor duration_sum = torch::zeros_like(time, utils::getTensorOptions(device_));
  torch::Tensor found = torch::zeros_like(time, utils::getTensorOptions(device_, torch::kBool));
  torch::Tensor segments_durations = torch::empty_like(time, utils::getTensorOptions(device_));
  torch::Tensor segments_lengths = torch::empty_like(time, utils::getTensorOptions(device_));
  torch::Tensor segments_normalized = torch::empty({time.size(0), 3}, utils::getTensorOptions(device_));
  torch::Tensor waypoints = torch::zeros({time.size(0), 3}, utils::getTensorOptions(device_));
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

FramesObstacle::FramesObstacle(const std::string& id, int batch_size, torch::DeviceType device)
    : Obstacle(id, batch_size, device), frame_distance_(0), ob_ids_(torch::zeros({batch_size, 1}, torch::kInt64)) {}

void FramesObstacle::copyToDevice(torch::DeviceType device) {}

void FramesObstacle::setFrames(torch::Tensor frames, torch::Tensor lengths, torch::Tensor rcms, double frame_distance) {
  frames_ = std::move(frames);
  lengths_ = std::move(lengths);
  all_rcms_ = std::move(rcms);
  frame_distance_ = frame_distance;
}

void FramesObstacle::setObIDs(const torch::Tensor& ob_ids) { setObIDs(ob_ids, torch::ones_like(ob_ids_)); }

void FramesObstacle::setObIDs(const torch::Tensor& ob_ids, const torch::Tensor& mask) {
  ob_ids_ = torch::where(mask, ob_ids, ob_ids_).to(torch::kInt64);
  rcm_ = torch::gather(all_rcms_, 0, ob_ids_.expand({ob_ids_.size(0), 3}));
}

torch::Tensor FramesObstacle::getPositionAt(torch::Tensor time) {
  torch::Tensor robot_specific_lengths = torch::gather(lengths_, 0, ob_ids_);
  torch::Tensor frame_index = (torch::round(time / frame_distance_) % robot_specific_lengths).to(torch::kInt64);
  torch::Tensor robot_specific_frames = torch::gather(frames_, 0, ob_ids_.unsqueeze(-1).expand({ob_ids_.size(0), frames_.size(1), frames_.size(2)}));
  return torch::gather(robot_specific_frames, 1, frame_index.unsqueeze(1).expand({frame_index.size(0), 1, 3})).squeeze();
}

bool FramesObstacle::isValidFile(const std::string& file) { return fs::is_regular_file(file) && file.substr(file.size() - 4, 4) == ".csv"; }

std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> FramesObstacle::loadDataset(const std::string& path, unsigned int num,
                                                                                                 double frame_distance, unsigned int reference_ob,
                                                                                                 const torch::Tensor& reference_rcm,
                                                                                                 const torch::Tensor& workspace_dims) {
  // get all .csv files
  std::vector<std::string> csv_files;
  if (isValidFile(path))
    csv_files.push_back(path);
  else if (fs::is_directory(path)) {
    for (auto& file : fs::recursive_directory_iterator(path))
      if (isValidFile(file.path().string())) csv_files.push_back(file.path().string());
  }
  if (csv_files.empty()) throw CSVError("No .csv files found in " + path);
  // parse files
  std::vector<torch::Tensor> out_frames(num);
  std::vector<torch::Tensor> out_rcms(num);
  std::vector<torch::Tensor> out_lengths(num);
  unsigned int file_i = 0;
  boost::random::mt19937 rng;
  torch::Tensor max_dev = (0.5 * workspace_dims).squeeze();
  for (auto& file : csv_files) {
    // parse raw geometry and time
    std::vector<std::map<double, Affine3d>> affines(num);
    std::string content = utils::readFile(file);
    std::istringstream line_stream(content);
    std::string buffer;
    double start_time;
    size_t line = 0;
    bool all_visible = true;
    double last_time;
    Vector3d first_mid = Vector3d::Zero();
    while (std::getline(line_stream, buffer)) {
      std::istringstream token_stream(buffer);
      std::vector<std::string> tokens;
      while (std::getline(token_stream, buffer, delimiter)) tokens.push_back(buffer);
      double time = std::stod(tokens[0]);
      if (line == 0) start_time = time;
      time = (time - start_time);
      last_time = time;
      Vector3d mean = Vector3d::Zero();
      for (size_t i = 0; i < num; i++) {
        unsigned int j = 9 * i + 1;
        if (j + 8 >= tokens.size()) throw CSVError("Expected " + std::to_string(num) + " obstacles in " + file);
        Quaterniond rotation(std::stod(tokens[j + 4]), std::stod(tokens[j + 1]), std::stod(tokens[j + 2]), std::stod(tokens[j + 3]));
        Vector3d position(std::stod(tokens[j + 5]), std::stod(tokens[j + 6]), std::stod(tokens[j + 7]));
        all_visible = all_visible && (bool)std::stoi(tokens[j + 8]);
        position *= 1e-3;
        mean += position;
        affines[i][time] = Translation3d(position) * rotation;
      }
      mean /= num;
      if (line == 0) first_mid = Vector3d(mean);
      torch::Tensor dev = utils::vectorToTensor(mean - first_mid).squeeze();
      bool critical_dev = (dev < -max_dev).logical_or(dev > max_dev).any().item<bool>();
      double distance = (affines[0][time].translation() - affines[1][time].translation()).norm();
      if (!all_visible || distance > 0.3 || critical_dev) break;
      line++;
    }
    if (last_time < 60) {
      ROS_WARN_STREAM_NAMED("control_force_provider", "" << file << " is only valid for " << last_time << " seconds. skipping...");
      continue;
    }
    if (affines[0].empty()) throw CSVError("Could not parse " + file);
    // estimate RCM and create interpolated frame tensor
    unsigned int ob_i = 0;
    for (auto& ob : affines) {
      torch::Tensor frames_tensor;
      boost::random::uniform_int_distribution<> index_sampler(0, affines.size() - 1);
      std::vector<int> indices;
      for (size_t i = 0; i < rcm_estimation_sample_num; i++) indices.push_back(index_sampler(rng));
      std::sort(indices.begin(), indices.end(), std::less<>());
      std::vector<Affine3d> poses;
      int index = 0;
      auto index_iter = indices.begin();
      last_time = 0;
      Vector3d last_position;
      double target_time = 0;
      for (auto& frame : ob) {
        Affine3d& pose = frame.second;
        Vector3d position = pose.translation();
        double time = frame.first;
        if (target_time <= time) {
          if (target_time == 0) {
            frames_tensor = torch::from_blob(position.data(), {1, 3}, torch::kFloat64).clone();
          } else {
            double time_between_frames = last_time - time;
            double rel_pos = (target_time - last_time) / time_between_frames;
            Vector3d target_position = last_position + rel_pos * (last_position - position);
            frames_tensor = torch::cat({frames_tensor, torch::from_blob(target_position.data(), {1, 3}, torch::kFloat64)}, 0);
          }
          target_time += frame_distance;
        }
        last_position = position;
        last_time = time;
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
            if (!p1.translation().isApprox(p2.translation(), 1e-5)) {
              torch::Tensor t, s = torch::zeros(1, utils::getTensorOptions());
              const Vector3d& a1 = p1.translation();
              Vector3d b1 = (p1 * Translation3d(0, 0, 1)).translation() - a1;
              const Vector3d& a2 = p2.translation();
              Vector3d b2 = (p2 * Translation3d(0, 0, 1)).translation() - a2;
              utils::shortestLine(utils::vectorToTensor(a1), utils::vectorToTensor(b1), utils::vectorToTensor(a2), utils::vectorToTensor(b2), t, s);
              if (!std::isnan(t.item().toFloat())) points.emplace_back(a1 + t.item().toFloat() * b1);
              if (!std::isnan(s.item().toFloat())) points.emplace_back(a2 + s.item().toFloat() * b2);
            }
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
      torch::Tensor rcm_t = torch::from_blob(rcm.data(), {1, 3}, torch::kFloat64).to(torch::kFloat32);
      out_rcms[ob_i] = file_i == 0 ? rcm_t : torch::cat({out_rcms[ob_i], rcm_t}, 0);
      int length = frames_tensor.size(0);
      torch::Tensor length_t = torch::full({1, 1}, length);
      out_lengths[ob_i] = file_i == 0 ? length_t : torch::cat({out_lengths[ob_i], length_t}, 0);
      if (file_i != 0) {
        int out_length = out_frames[ob_i].size(1);
        if (length < out_length) {
          frames_tensor = torch::cat({frames_tensor, torch::zeros({out_length - length, 3})}, 0);
        } else if (length > out_length) {
          out_frames[ob_i] = torch::cat({out_frames[ob_i], torch::zeros({out_frames[ob_i].size(0), length - out_length, 3})}, 1);
        }
      }
      frames_tensor = frames_tensor.unsqueeze(0).to(torch::kFloat32);
      out_frames[ob_i] = file_i == 0 ? frames_tensor : torch::cat({out_frames[ob_i], frames_tensor}, 0);
      ob_i++;
    }
    file_i++;
    ROS_INFO_STREAM_NAMED("control_force_provider", "Loaded CSV file " << file_i << "/" << csv_files.size() << " (" << file << ")");
  }
  std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> out;
  torch::Tensor translation;
  for (size_t i = 0; i < num; i++) {
    unsigned int ob_i = (reference_ob + i) % num;
    if (i == 0) {
      translation = reference_rcm.expand({1, 3}) - out_rcms[ob_i];
    }
    out_rcms[ob_i] += translation;
    torch::Tensor zero_mask = out_frames[ob_i] == 0;
    out_frames[ob_i] += translation.unsqueeze(1);
    out_frames[ob_i] = torch::where(zero_mask, 0, out_frames[ob_i]);
    out.emplace_back(out_frames[ob_i], out_lengths[ob_i], out_rcms[ob_i]);
  }
  return out;
}
}  // namespace control_force_provider::backend