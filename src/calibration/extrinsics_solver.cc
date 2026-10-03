#include "calibration/extrinsics_solver.h"

#include <array>
#include <cmath>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <unordered_set>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <nlohmann/json.hpp>

namespace calibration {
namespace {
constexpr double kTranslationScale = 0.10;
constexpr double kRotationScale = 10 * std::numbers::pi / 180;
constexpr double kRotationPriorScale = 5 * std::numbers::pi / 180;

auto CameraPose(const Observation& observation) -> ExtrinsicPose {
  const auto matrix = observation.field_to_camera.ToMatrix();
  return {Eigen::Quaterniond{matrix.topLeftCorner<3, 3>()},
          matrix.topRightCorner<3, 1>()};
}

template <typename T>
void Quaternion(const Eigen::Quaterniond& q, T* output) {
  output[0] = T(q.w()); output[1] = T(q.x());
  output[2] = T(q.y()); output[3] = T(q.z());
}

template <typename T>
void Predict(const ExtrinsicPose& y, const T* parameters, T* q, T* p) {
  T yq[4], eq[4];
  Quaternion(y.rotation, yq);
  ceres::AngleAxisToQuaternion(parameters, eq);
  ceres::QuaternionProduct(yq, eq, q);
  ceres::QuaternionRotatePoint(yq, parameters + 3, p);
  for (int i = 0; i < 3; ++i) p[i] += T(y.translation[i]);
}

template <typename T>
void RotationLog(const T* a, const T* b, T* output) {
  const T inverse[4] = {a[0], -a[1], -a[2], -a[3]};
  T delta[4];
  ceres::QuaternionProduct(inverse, b, delta);
  // Ceres uses the shortest rotation and has a finite derivative at identity.
  ceres::QuaternionToAngleAxis(delta, output);
}

struct PairResidual {
  ExtrinsicPose yi, yj;
  double normalization;
  template <typename T>
  auto operator()(const T* ei, const T* ej, T* residual) const -> bool {
    T qi[4], qj[4], pi[3], pj[3];
    Predict(yi, ei, qi, pi);
    Predict(yj, ej, qj, pj);
    RotationLog(qi, qj, residual + 3);
    for (int i = 0; i < 3; ++i) {
      residual[i] = (pi[i] - pj[i]) * T(normalization / kTranslationScale);
      residual[i + 3] *= T(normalization / kRotationScale);
    }
    return true;
  }
};

struct RotationPrior {
  Eigen::Quaterniond initial;
  template <typename T>
  auto operator()(const T* parameters, T* residual) const -> bool {
    T q0[4], q[4];
    Quaternion(initial, q0);
    ceres::AngleAxisToQuaternion(parameters, q);
    RotationLog(q0, q, residual);
    for (int i = 0; i < 3; ++i) residual[i] /= T(kRotationPriorScale);
    return true;
  }
};

auto Parameters(const ExtrinsicPose& pose) -> std::array<double, 6> {
  std::array<double, 6> parameters{};
  double q[4];
  Quaternion(pose.rotation, q);
  ceres::QuaternionToAngleAxis(q, parameters.data());
  for (int i = 0; i < 3; ++i) parameters[i + 3] = pose.translation[i];
  return parameters;
}

void Validate(const std::vector<ObservationGroup>& groups,
              const std::vector<ExtrinsicPose>& extrinsics) {
  for (const auto& e : extrinsics) {
    if (!e.translation.allFinite() || !e.rotation.coeffs().allFinite() ||
        std::abs(e.rotation.norm() - 1) > 1e-8)
      throw std::invalid_argument("Extrinsics must be finite with unit quaternions");
  }
  for (const auto& group : groups) {
    if (group.size() < 2) throw std::invalid_argument("An observation group needs two cameras");
    std::unordered_set<size_t> cameras;
    for (const auto& o : group) {
      if (o.camera >= extrinsics.size() || !cameras.insert(o.camera).second ||
          !o.field_to_camera.ToMatrix().allFinite() || o.capture_ns < 0)
        throw std::invalid_argument("Invalid observation");
    }
  }
}
}  // namespace

auto LoadExtrinsics(const std::vector<ReplayCamera>& cameras)
    -> std::vector<ExtrinsicPose> {
  std::vector<ExtrinsicPose> result;
  for (const auto& camera : cameras) {
    const auto matrix = camera::Extrinsics{camera.config_path}
                            .ToCameraToRobot<frc::Transform3d>().ToMatrix();
    result.push_back({Eigen::Quaterniond{matrix.topLeftCorner<3, 3>()},
                      matrix.topRightCorner<3, 1>()});
  }
  return result;
}

auto EvaluatePairErrors(const std::vector<ObservationGroup>& groups,
                        const std::vector<ExtrinsicPose>& extrinsics) -> PairErrors {
  Validate(groups, extrinsics);
  PairErrors errors;
  double translation_squared = 0, rotation_squared = 0;
  for (const auto& group : groups) {
    for (size_t i = 0; i < group.size(); ++i) {
      for (size_t j = i + 1; j < group.size(); ++j) {
        const auto a = Parameters(extrinsics[group[i].camera]);
        const auto b = Parameters(extrinsics[group[j].camera]);
        double residual[6];
        PairResidual{CameraPose(group[i]), CameraPose(group[j]), 1}(a.data(), b.data(), residual);
        for (int d = 0; d < 3; ++d) {
          translation_squared += std::pow(residual[d] * kTranslationScale, 2);
          rotation_squared += std::pow(residual[d + 3] * kRotationScale, 2);
        }
        ++errors.pairs;
      }
    }
  }
  if (errors.pairs) {
    errors.translation_rms_m = std::sqrt(translation_squared / errors.pairs);
    errors.rotation_rms_deg = std::sqrt(rotation_squared / errors.pairs) * 180 / std::numbers::pi;
    errors.normalized_half_mean_squared = 0.5 / errors.pairs *
        (translation_squared / std::pow(kTranslationScale, 2) +
         rotation_squared / std::pow(kRotationScale, 2));
  }
  return errors;
}

auto SolveExtrinsics(const std::vector<ObservationGroup>& groups,
                     const std::vector<ExtrinsicPose>& initial, size_t anchor)
    -> CalibrationResult {
  if (anchor >= initial.size()) throw std::invalid_argument("Invalid anchor camera");
  CalibrationResult result;
  result.initial_errors = EvaluatePairErrors(groups, initial);
  if (!result.initial_errors.pairs) throw std::invalid_argument("No training pairs");
  // Every optimized camera needs a path to the anchor in the observation graph.
  std::vector<bool> connected(initial.size(), false);
  connected[anchor] = true;
  for (size_t pass = 0; pass < initial.size(); ++pass) {
    for (const auto& group : groups) {
      bool touches = false;
      for (const auto& o : group) touches |= connected[o.camera];
      if (touches) for (const auto& o : group) connected[o.camera] = true;
    }
  }
  for (bool c : connected) if (!c)
    throw std::invalid_argument("Training cameras are not connected to the anchor");

  std::vector<std::array<double, 6>> parameters;
  for (const auto& pose : initial) parameters.push_back(Parameters(pose));
  ceres::Problem problem;
  const double normalization = 1 / std::sqrt(static_cast<double>(result.initial_errors.pairs));
  for (const auto& group : groups) {
    for (size_t i = 0; i < group.size(); ++i) {
      for (size_t j = i + 1; j < group.size(); ++j) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PairResidual, 6, 6, 6>(
                new PairResidual{CameraPose(group[i]), CameraPose(group[j]), normalization}),
            nullptr, parameters[group[i].camera].data(), parameters[group[j].camera].data());
      }
    }
  }
  for (size_t i = 0; i < initial.size(); ++i) {
    if (i == anchor) {
      problem.SetParameterBlockConstant(parameters[i].data());
    } else {
      problem.AddResidualBlock(new ceres::AutoDiffCostFunction<RotationPrior, 3, 6>(
          new RotationPrior{initial[i].rotation}), nullptr, parameters[i].data());
    }
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 200;
  options.function_tolerance = 1e-12;
  options.gradient_tolerance = 1e-12;
  options.parameter_tolerance = 1e-12;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  result.usable = summary.IsSolutionUsable();
  result.solver_report = summary.BriefReport();
  for (size_t i = 0; i < initial.size(); ++i) {
    if (i == anchor) {
      result.extrinsics.push_back(initial[i]);  // Preserve the anchor exactly.
    } else {
      double q[4];
      ceres::AngleAxisToQuaternion(parameters[i].data(), q);
      result.extrinsics.push_back({Eigen::Quaterniond{q[0], q[1], q[2], q[3]},
          Eigen::Vector3d{parameters[i][3], parameters[i][4], parameters[i][5]}});
    }
  }
  result.final_errors = EvaluatePairErrors(groups, result.extrinsics);
  return result;
}

void WriteCandidateConfigs(const std::vector<ReplayCamera>& cameras,
                           const std::vector<ExtrinsicPose>& extrinsics,
                           const std::filesystem::path& output_directory) {
  if (cameras.size() != extrinsics.size()) throw std::invalid_argument("Camera count mismatch");
  Validate({}, extrinsics);
  std::unordered_set<std::string> filenames;
  std::vector<nlohmann::json> configs;
  for (size_t i = 0; i < cameras.size(); ++i) {
    const auto filename = cameras[i].config_path.filename();
    if (!filenames.insert(filename.string()).second ||
        std::filesystem::exists(output_directory / filename))
      throw std::invalid_argument("Refusing to overwrite candidate configuration: " + filename.string());
    std::ifstream input(cameras[i].config_path);
    auto json = nlohmann::json::parse(input);
    // JSON stores the inverse pose: camera position and Euler angles in robot.
    const auto& e = extrinsics[i];
    const Eigen::Vector3d translation = -(e.rotation.conjugate() * e.translation);
    const auto q = e.rotation.conjugate();
    const frc::Rotation3d rotation{frc::Quaternion{q.w(), q.x(), q.y(), q.z()}};
    const auto original = camera::Extrinsics{cameras[i].config_path}
                              .ToRobotToCamera<frc::Transform3d>().ToMatrix();
    const frc::Pose3d candidate{units::meter_t{translation.x()},
        units::meter_t{translation.y()}, units::meter_t{translation.z()}, rotation};
    // Preserve the anchor's original JSON numbers as well as its transform.
    if (!candidate.ToMatrix().isApprox(original, 1e-12)) {
      auto& values = json["extrinsics"];
      values["translation_x"] = translation.x();
      values["translation_y"] = translation.y();
      values["translation_z"] = translation.z();
      values["rotation_x"] = units::degree_t{rotation.X()}.value();
      values["rotation_y"] = units::degree_t{rotation.Y()}.value();
      values["rotation_z"] = units::degree_t{rotation.Z()}.value();
    }
    configs.push_back(std::move(json));
  }
  std::filesystem::create_directories(output_directory);
  for (size_t i = 0; i < cameras.size(); ++i) {
    std::ofstream output(output_directory / cameras[i].config_path.filename());
    output << configs[i].dump(2) << '\n';
    if (!output) throw std::runtime_error("Cannot write candidate configuration");
  }
}

}  // namespace calibration
