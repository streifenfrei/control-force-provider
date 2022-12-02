#include "control_force_provider/utils.h"

#include <ros/ros.h>

#include <boost/regex.hpp>
#include <fstream>
#include <iostream>
#include <sstream>

namespace control_force_provider {
namespace exceptions {
Error::Error(const std::string &message, const std::string &error_type) : runtime_error(message) {
  ROS_ERROR_STREAM_NAMED("control_force_provider", error_type << ": " << message);
}
ConfigError::ConfigError(const std::string &message) : Error(message, "ConfigError") {}
CSVError::CSVError(const std::string &message) : Error(message, "CSVError") {}
PythonError::PythonError(const std::string &message) : Error(message, "PythonError") {}
}  // namespace exceptions

namespace utils {
using namespace exceptions;
using namespace Eigen;

torch::TensorOptions getTensorOptions() { return torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU).requires_grad(false); }

VectorXd tensorToVector(const torch::Tensor &tensor) {
  if (tensor.dim() > 1) {
    auto acc = tensor.accessor<double, 2>();
    VectorXd out(tensor.size(1));
    for (size_t i = 0; i < tensor.size(1); i++) out[i] = acc[0][i];
    return out;
  } else {
    auto acc = tensor.accessor<double, 1>();
    VectorXd out(tensor.size(0));
    for (size_t i = 0; i < tensor.size(0); i++) out[i] = acc[i];
    return out;
  }
}

torch::Tensor vectorToTensor(const VectorXd &vector) {
  torch::Tensor out = torch::empty(vector.size(), getTensorOptions());
  auto acc = out.accessor<double, 1>();
  for (size_t i = 0; i < vector.size(); i++) {
    acc[i] = vector[i];
  }
  return out.unsqueeze(0);
}

torch::Tensor createTensor(const std::vector<double> &values, unsigned int start, unsigned int end) {
  if (end == -1) end = values.size();
  torch::Tensor out = torch::empty(end - start, getTensorOptions());
  auto acc = out.accessor<double, 1>();
  for (size_t i = 0; i < end - start; i++) {
    acc[i] = values[start + i];
  }
  return out.unsqueeze(0);
}

torch::Tensor norm(const torch::Tensor &tensor) { return torch::linalg::vector_norm(tensor, 2, -1, true, torch::kFloat64); }

void normalize(torch::Tensor &tensor) { tensor /= utils::norm(tensor); }

torch::Tensor dot(const torch::Tensor &tensor1, const torch::Tensor &tensor2) {
  return torch::bmm(tensor1.view({tensor1.size(0), 1, tensor1.size(1)}), tensor2.view({tensor2.size(0), tensor2.size(1), 1})).squeeze(-1);
}

std::vector<std::string> regexFindAll(const std::string &regex, const std::string &str) {
  boost::sregex_token_iterator iter(str.begin(), str.end(), boost::regex(regex), 0);
  boost::sregex_token_iterator end;
  std::vector<std::string> result;
  for (; iter != end; iter++) result.emplace_back(*iter);
  return result;
}

std::string readFile(const std::string &file) {
  std::ifstream file_stream(file);
  std::ostringstream string_stream{};
  string_stream << file_stream.rdbuf();
  std::string content = string_stream.str();
  file_stream.close();
  return content;
}

void shortestLine(const torch::Tensor &a1, const torch::Tensor &b1, const torch::Tensor &a2, const torch::Tensor &b2, torch::Tensor &t, torch::Tensor &s) {
  //  tool1:                                                    l1 = a1 + t*b1
  //  tool2:                                                    l2 = a2 + s*b2
  //  general vector between l1 and l2:                         v = a2 - a1 + sb2 - t*b1 = a' + s*b2 - t*b1
  torch::Tensor a_diff = a2 - a1;
  //  the shortest line is perpendicular to both tools (v•b1 = v•b2 = 0). We want to solve this LEQ for t and s:
  //                b1•a' + s*b1•b2 - t*b1•b1 = 0
  //                b2•a' + s*b2•b2 - t*b2•b1 = 0
  //  substitute e1 = b1•b2, e2 = b2•b2 and e3 = b1•b1
  torch::Tensor e1 = utils::dot(b1, b2);
  torch::Tensor e2 = utils::dot(b2, b2);
  torch::Tensor e3 = utils::dot(b1, b1);
  //                b1•a' + s*e1 - t*e3 = 0                                 |* e2
  //                b2•a' + s*e2 - t*e1 = 0                                 |* e1
  //                ————————————————————————————
  //                b1•a'*e2 + s*e1*e2 - t*e2*e3 = 0                        -
  //                b2•a'*e1 + s*e1*e2 - t*e1*e1 = 0
  //                ————————————————————————————————
  //                b1•a'*e2 - t*e2*e3 - b2•a'*e1 + t*e1*e1 = 0             |+ b2•a'*e1, - b1•a'*e2
  //                ———————————————————————————————————————————
  //                t*(e1*e1 - e2*e3) = a'•(b2*e1 - b1*e2)                  |: (e1*e1 - b1•b1*e2)
  //                ——————————————————————————————————————
  //                t = a'•(b2*e1 - b1*e2) / (e1*e1 - e2*e3)
  t = utils::dot(a_diff, b2 * e1 - b1 * e2) / (e1 * e1 - e2 * e3);
  //  use the first equation to get s:
  //                s*e1 = t*e3 - b1•a'                                     |: e1
  //                ———————————————————
  //                s = (t*e3 - b1•a') / e1
  s = (t * e3 - utils::dot(b1, a_diff)) / e1;
}

torch::Tensor zRotation(const torch::Tensor &p1, const torch::Tensor &p2) {
  torch::Tensor vec = p2 - p1;
  normalize(vec);
  torch::Tensor z = createTensor({0, 0, 1});
  torch::Tensor d = utils::dot(vec, z);
  torch::Tensor w = torch::cross(vec, z, -1);
  torch::Tensor out = torch::cat({d + torch::sqrt(d * d + utils::dot(w, w)), w}, -1);
  utils::normalize(out);
  return out;
}
}  // namespace utils
}  // namespace control_force_provider