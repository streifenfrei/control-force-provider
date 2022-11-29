#pragma once

#include <ros/ros.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <boost/random.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

#include "time.h"
#include "utils.h"

namespace control_force_provider::backend {
class Obstacle {
 private:
  double start_time;

 protected:
  const std::string id_;
  torch::Tensor rcm_;
  virtual torch::Tensor getPositionAt(double time) = 0;

 public:
  Obstacle(const std::string& id) : start_time(Time::now()), id_(id), rcm_(torch::zeros(3, utils::getTensorOptions())){};
  void reset(double offset = 0);
  torch::Tensor getPosition();
  virtual ~Obstacle() = default;

  [[nodiscard]] const torch::Tensor& getRCM() const { return rcm_; }
  void setRCM(torch::Tensor rcm) { rcm_ = rcm; }
};

class DummyObstacle : public Obstacle {
 protected:
  torch::Tensor getPositionAt(double time) override { return torch::zeros(3, utils::getTensorOptions()); };

 public:
  explicit DummyObstacle(const std::string& id) : Obstacle(id){};
};

class WaypointsObstacle : public Obstacle {
 private:
  std::vector<torch::Tensor> waypoints_;
  std::vector<torch::Tensor> segments_;
  std::vector<torch::Tensor> segments_normalized_;
  std::vector<double> segments_lengths_;
  std::vector<double> segments_durations_;
  double total_duration_;
  double speed_;  // m/s

 protected:
  torch::Tensor getPositionAt(double time) override;

 public:
  WaypointsObstacle(const YAML::Node& config, const std::string& id);
  ~WaypointsObstacle() override = default;
};

class FramesObstacle : public Obstacle {
 private:
  std::map<double, Eigen::Affine3d> frames_;
  std::map<double, Eigen::Affine3d>::iterator iter_;

 protected:
  torch::Tensor getPositionAt(double time) override;

 public:
  explicit FramesObstacle(const std::string& id);
  void setFrames(std::map<double, Eigen::Affine3d> frames);
  ~FramesObstacle() override = default;
};

class ObstacleLoader {
 private:
  const static inline char delimiter = ',';
  const static inline unsigned int rcm_estimation_sample_num = 100;
  const static inline double rcm_estimation_max_var = 1e-6;
  std::vector<boost::shared_ptr<FramesObstacle>> obstacles_;
  std::vector<std::string> csv_files_;
  std::vector<std::string>::iterator files_iter_;
  std::vector<std::map<double, Eigen::Affine3d>> cached_frames_;
  int reference_obstacle_;
  boost::random::mt19937 rng_;
  static bool isValidFile(const std::string& file);
  std::vector<std::map<double, Eigen::Affine3d>> parseFile(const std::string& file);
  torch::Tensor estimateRCM(const std::map<double, Eigen::Affine3d>& frames);
  void updateObstacles(std::vector<std::map<double, Eigen::Affine3d>> frames);

 public:
  ObstacleLoader(std::vector<boost::shared_ptr<FramesObstacle>> obstacles, const std::string& path, int reference_obstacle = -1);
  void loadNext();
};
}  // namespace control_force_provider::backend