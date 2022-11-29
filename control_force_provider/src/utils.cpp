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
  return out;
}

torch::Tensor createTensor(const std::vector<double> &values) {
  torch::Tensor out = torch::empty(values.size(), getTensorOptions());
  auto acc = out.accessor<double, 1>();
  for (size_t i = 0; i < values.size(); i++) {
    acc[i] = values[i];
  }
  return out;
}

torch::Tensor norm(const torch::Tensor &tensor) { return torch::linalg::vector_norm(tensor, 2, {}, false, torch::kFloat64); }

std::vector<std::string> regexFindAll(const std::string &regex, const std::string &str) {
  boost::sregex_token_iterator iter(str.begin(), str.end(), boost::regex(regex), 0);
  boost::sregex_token_iterator end;
  std::vector<std::string> result;
  for (; iter != end; iter++) result.emplace_back(*iter);
  return result;
}

torch::Tensor tensorFromList(const std::vector<double> &list, unsigned int start_index) {
  torch::Tensor out = torch::empty(3, getTensorOptions());
  auto accessor = out.accessor<double, 1>();
  accessor[0] = list[start_index];
  accessor[1] = list[start_index + 1];
  accessor[2] = list[start_index + 2];
  return out;
}

std::string readFile(const std::string &file) {
  std::ifstream file_stream(file);
  std::ostringstream string_stream{};
  string_stream << file_stream.rdbuf();
  std::string content = string_stream.str();
  file_stream.close();
  return content;
}

void shortestLine(const Vector3d &a1, const Vector3d &b1, const Vector3d &a2, const Vector3d &b2, double &t, double &s) {
  //  tool1:                                                    l1 = a1 + t*b1
  //  tool2:                                                    l2 = a2 + s*b2
  //  general vector between l1 and l2:                         v = a2 - a1 + sb2 - t*b1 = a' + s*b2 - t*b1
  Vector3d a_diff = a2 - a1;
  //  the shortest line is perpendicular to both tools (v•b1 = v•b2 = 0). We want to solve this LEQ for t and s:
  //                b1•a' + s*b1•b2 - t*b1•b1 = 0
  //                b2•a' + s*b2•b2 - t*b2•b1 = 0
  //  substitute e1 = b1•b2, e2 = b2•b2 and e3 = b1•b1
  double e1 = b1.dot(b2);
  double e2 = b2.dot(b2);
  double e3 = b1.dot(b1);
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
  t = a_diff.dot(b2 * e1 - b1 * e2) / (e1 * e1 - e2 * e3);
  //  use the first equation to get s:
  //                s*e1 = t*e3 - b1•a'                                     |: e1
  //                ———————————————————
  //                s = (t*e3 - b1•a') / e1
  s = (t * e3 - b1.dot(a_diff)) / e1;
}

Quaterniond zRotation(const torch::Tensor &p1, const torch::Tensor &p2) {
  // TODO: witcalc with tensor
  Vector3d vec = (tensorToVector(p2) - tensorToVector(p1)).normalized();
  Vector3d z = Vector3d::UnitZ();
  AngleAxis rot(std::acos(z.dot(vec)), z.cross(vec).normalized());
  return Quaterniond(rot.toRotationMatrix());
}

}  // namespace utils
}  // namespace control_force_provider