#pragma once

#include <ros/ros.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <boost/optional.hpp>
#include <boost/random.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

#include "time.h"
#include "utils.h"

namespace control_force_provider::backend {
class Obstacle {
 private:
  torch::Tensor start_time;

 protected:
  const std::string id_;
  torch::DeviceType device_;
  torch::Tensor rcm_;
  virtual void copyToDevice(torch::DeviceType device) = 0;
  virtual torch::Tensor getPositionAt(torch::Tensor time) = 0;

 public:
  Obstacle(const std::string& id, unsigned int batch_size = 1, torch::DeviceType device = torch::kCPU)
      : start_time(torch::full({batch_size, 1}, Time::now(), utils::getTensorOptions(device))),
        id_(id),
        rcm_(torch::zeros(3, utils::getTensorOptions(device))),
        device_(device){};
  void setDevice(torch::DeviceType device);
  void reset(const torch::Tensor& mask, const torch::Tensor& offset);
  void reset(const torch::Tensor& offset);
  torch::Tensor getPosition();
  virtual ~Obstacle() = default;

  [[nodiscard]] const torch::Tensor& getRCM() const { return rcm_; }
  void setRCM(torch::Tensor rcm) { rcm_ = rcm.to(device_); }
  static std::vector<boost::shared_ptr<Obstacle>> createFromConfig(const YAML::Node& config, std::string& data_path, int batch_size = 1,
                                                                   torch::DeviceType device = torch::kCPU);
};

class DummyObstacle : public Obstacle {
 protected:
  void copyToDevice(torch::DeviceType device) override{};
  torch::Tensor getPositionAt(torch::Tensor time) override { return torch::zeros({time.size(0), 3}, utils::getTensorOptions(device_)); };

 public:
  DummyObstacle(const std::string& id, int batch_size = 1, torch::DeviceType device = torch::kCPU) : Obstacle(id, batch_size, device){};
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
  void copyToDevice(torch::DeviceType device) override;
  torch::Tensor getPositionAt(torch::Tensor time) override;

 public:
  WaypointsObstacle(const YAML::Node& config, const std::string& id, int batch_size = 1, torch::DeviceType device = torch::kCPU);
  ~WaypointsObstacle() override = default;
};

class FramesObstacle : public Obstacle {
 private:
  torch::Tensor frames_;
  torch::Tensor all_rcms_;
  torch::Tensor ob_ids_;
  torch::Tensor lengths_;
  double frame_distance_;
  static bool isValidFile(const std::string& file);
  const static inline char delimiter = ',';
  const static inline unsigned int rcm_estimation_sample_num = 100;
  const static inline double rcm_estimation_max_var = 1e-6;

 protected:
  void copyToDevice(torch::DeviceType device) override;
  torch::Tensor getPositionAt(torch::Tensor time) override;

 public:
  explicit FramesObstacle(const std::string& id, int batch_size = 1, torch::DeviceType device = torch::kCPU);
  unsigned int getObsAmount() { return frames_.size(0); }
  void setFrames(torch::Tensor frames, torch::Tensor lengths, torch::Tensor rcms, double frame_distance);
  void setObIDs(const torch::Tensor& ob_ids, const torch::Tensor& mask);
  void setObIDs(const torch::Tensor& ob_ids);
  torch::Tensor getDurations();
  ~FramesObstacle() override = default;
  static std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> loadDataset(const std::string& path, unsigned int num, double frame_distance,
                                                                                          unsigned int reference_ob, const torch::Tensor& reference_rcm,
                                                                                          const torch::Tensor& workspace_dims);
};
}  // namespace control_force_provider::backend