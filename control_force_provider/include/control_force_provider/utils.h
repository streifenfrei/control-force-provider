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

torch::Tensor createTensor(const std::vector<double> &args, unsigned int start = 0, unsigned int end = -1);

torch::Tensor norm(const torch::Tensor &tensor);

void normalize(torch::Tensor &tensor);

torch::Tensor dot(const torch::Tensor &tensor1, const torch::Tensor &tensor2);

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

std::string readFile(const std::string &file);

void shortestLine(const torch::Tensor &a1, const torch::Tensor &b1, const torch::Tensor &a2, const torch::Tensor &b2, torch::Tensor &t, torch::Tensor &s);

torch::Tensor zRotation(const torch::Tensor &p1, const torch::Tensor &p2);
}  // namespace utils
}  // namespace control_force_provider