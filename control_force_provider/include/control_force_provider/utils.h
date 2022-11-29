#pragma once

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>

namespace control_force_provider {
namespace exceptions {
class Error : public std::runtime_error {
 public:
  explicit Error(const std::string &message, const std::string &error_type);
};
class ConfigError : Error {
 public:
  explicit ConfigError(const std::string &message);
};
class CSVError : Error {
 public:
  explicit CSVError(const std::string &message);
};
class PythonError : Error {
 public:
  explicit PythonError(const std::string &message);
};
}  // namespace exceptions

namespace utils {
using namespace exceptions;

torch::TensorOptions getTensorOptions();

Eigen::VectorXd tensorToVector(const torch::Tensor &tensor);

torch::Tensor vectorToTensor(const Eigen::VectorXd &vector);

torch::Tensor createTensor(const std::vector<double> &args);

torch::Tensor norm(const torch::Tensor &tensor);

template <typename T>
std::vector<T> getConfigValue(const YAML::Node &config, const std::string &key) {
  YAML::Node node = config[key];
  if (!node.IsDefined()) {
    throw ConfigError("Missing key: " + key);
  }
  try {
    std::vector<T> value;
    switch (node.Type()) {
      case YAML::NodeType::Sequence:
        for (auto &&i : node) {
          value.push_back(i.as<T>());
        }
        break;
      default:
        value.push_back(node.as<T>());
    }
    return value;
  } catch (const YAML::BadConversion &exception) {
    throw ConfigError("Bad data type for key '" + key + "'. Expected: " + typeid(T).name());
  }
}

std::vector<std::string> regexFindAll(const std::string &regex, const std::string &str);

torch::Tensor tensorFromList(const std::vector<double> &list, unsigned int start_index);

std::string readFile(const std::string &file);

void shortestLine(const Eigen::Vector3d &a1, const Eigen::Vector3d &b1, const Eigen::Vector3d &a2, const Eigen::Vector3d &b2, double &t, double &s);

Eigen::Quaterniond zRotation(const torch::Tensor &p1, const torch::Tensor &p2);
}  // namespace utils
}  // namespace control_force_provider