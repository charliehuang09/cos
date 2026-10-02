#include "calibration/extrinsics_solver.h"
#include "utils/cv_geometry.h"

#include <atomic>
#include <fstream>
#include <numbers>
#include <unordered_set>
#include <unistd.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <opencv2/calib3d.hpp>

namespace {
using calibration::ExtrinsicPose;
using calibration::Observation;
using calibration::ObservationGroup;

auto Obs(size_t frame, size_t camera, int64_t timestamp) -> Observation {
  return {frame, camera, timestamp, frc::Pose3d{}, frc::Pose3d{}};
}

auto Matrix(const ExtrinsicPose& e) -> Eigen::Matrix4d {
  Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
  result.topLeftCorner<3, 3>() = e.rotation.toRotationMatrix();
  result.topRightCorner<3, 1>() = e.translation;
  return result;
}

auto Synthetic(const std::vector<ExtrinsicPose>& truth, size_t count = 30)
    -> std::vector<ObservationGroup> {
  std::vector<ObservationGroup> groups;
  for (size_t k = 0; k < count; ++k) {
    const frc::Pose3d robot{units::meter_t{2 + 0.1 * k}, 3_m, 0_m,
        frc::Rotation3d{0_rad, 0_rad, units::radian_t{0.1 * k}}};
    ObservationGroup group;
    for (size_t i = 0; i < truth.size(); ++i) {
      group.push_back({k * truth.size() + i, i, static_cast<int64_t>(k * 20'000'000 + i),
          frc::Pose3d{robot.ToMatrix() * Matrix(truth[i]).inverse()}, robot});
    }
    groups.push_back(std::move(group));
  }
  return groups;
}

auto Truth() -> std::vector<ExtrinsicPose> {
  return {{Eigen::Quaterniond{Eigen::AngleAxisd{0.2, Eigen::Vector3d::UnitY()}}, {0.1, 0, -0.3}},
      {Eigen::Quaterniond{Eigen::AngleAxisd{1.3, Eigen::Vector3d::UnitZ()}}, {-0.2, 0.4, -0.3}},
      {Eigen::Quaterniond{Eigen::AngleAxisd{-1.4, Eigen::Vector3d::UnitZ()}}, {0.3, -0.2, -0.1}}};
}

TEST(CaptureTimestamp, DecimalSecondsAndExactBoundary) {
  EXPECT_EQ(calibration::ParseCaptureTimestamp("173.036780"), 173'036'780'000);
  EXPECT_EQ(calibration::ParseCaptureTimestamp("0001.000000001"), 1'000'000'001);
  EXPECT_EQ(calibration::ParseCaptureTimestamp("1.010") - calibration::ParseCaptureTimestamp("1"),
            calibration::kMatchWindowNs);
  for (const auto* bad : {"", "nan", "1e3", "1.0000000001", "1.", "-1", ".1", "1.2.3"})
    EXPECT_THROW(calibration::ParseCaptureTimestamp(bad), std::invalid_argument);
  EXPECT_THROW(calibration::ParseCaptureTimestamp("99999999999999999999"), std::out_of_range);
}

TEST(Matching, StrictBoundaryAndDifferentCameras) {
  EXPECT_TRUE(calibration::MatchObservations({Obs(0, 0, 0), Obs(1, 1, 10'000'000)}).empty());
  EXPECT_EQ(calibration::MatchObservations({Obs(0, 0, 0), Obs(1, 1, 9'999'999)}).size(), 1u);
  EXPECT_TRUE(calibration::MatchObservations({Obs(0, 0, 0), Obs(1, 0, 1)}).empty());
  const auto groups = calibration::MatchObservations(
      {Obs(0, 0, 0), Obs(1, 1, 6'000'000), Obs(2, 2, 12'000'000)});
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(groups[0].size(), 2u);  // Adjacent gaps cannot extend the full window.
}

TEST(Matching, NonReuseByDefaultAndExplicitReuse) {
  const std::vector<Observation> observations{
      Obs(0, 0, 0), Obs(1, 0, 1), Obs(2, 1, 2), Obs(3, 2, 3)};
  const auto groups = calibration::MatchObservations(observations);
  ASSERT_EQ(groups.size(), 1u);
  ASSERT_EQ(groups[0].size(), 3u);
  EXPECT_EQ(groups[0][0].frame_id, 0u);
  const auto reused = calibration::MatchObservations(observations, true);
  EXPECT_EQ(reused.size(), 3u);
  EXPECT_EQ(reused[1][0].frame_id, 1u);
  EXPECT_EQ(reused[1][1].frame_id, 2u);
  EXPECT_THROW(calibration::MatchObservations({Obs(0, 0, 0), Obs(0, 1, 1)}), std::invalid_argument);
}

TEST(Matching, DeterministicHeldOutWithoutFrameLeakage) {
  std::vector<ObservationGroup> groups;
  for (size_t i = 0; i < 10; ++i) groups.push_back({Obs(2 * i, 0, i * 20'000'000),
      Obs(2 * i + 1, 1, i * 20'000'000 + 1)});
  groups.push_back({groups[4][0], groups[4][1]});
  const auto split = calibration::SplitGroups(groups);
  EXPECT_EQ(split.training.size(), 8u);
  EXPECT_EQ(split.held_out.size(), 3u);
  std::unordered_set<size_t> training_frames;
  for (const auto& group : split.training)
    for (const auto& o : group) training_frames.insert(o.frame_id);
  for (const auto& group : split.held_out)
    for (const auto& o : group) EXPECT_FALSE(training_frames.contains(o.frame_id));
}

TEST(BranchSelection, RetainsSelectedSecondBranchAndInputIdentity) {
  localization::UnambiguousSolverNode selector{""};
  localization::ambiguous_estimate_t previous;
  previous.pos1.pose = frc::Pose3d{5_m, 3_m, 0_m, frc::Rotation3d{}};
  previous.pos1.variance = 1;
  ASSERT_TRUE(selector.Solve({&previous}, false));
  localization::ambiguous_estimate_t candidates;
  candidates.pos1 = previous.pos1;
  candidates.pos1.pose = frc::Pose3d{9_m, 3_m, 0_m, frc::Rotation3d{}};
  candidates.pos1.field_to_camera = frc::Pose3d{10_m, 3_m, 1_m, frc::Rotation3d{}};
  candidates.pos2 = previous.pos1;
  candidates.pos2->field_to_camera = frc::Pose3d{6_m, 3_m, 1_m, frc::Rotation3d{}};
  const auto result = selector.SolveSelected({nullptr, &candidates}, false);
  ASSERT_TRUE(result);
  ASSERT_EQ(result->selected.size(), 1u);
  EXPECT_EQ(result->selected[0].input_index, 1u);
  EXPECT_EQ(result->combined.pose.X(), 5_m);
  ASSERT_TRUE(result->selected[0].estimate.field_to_camera);
  EXPECT_EQ(result->selected[0].estimate.field_to_camera->X(), 6_m);
}

TEST(BranchSelection, FilteringKeepsPoseAndCameraBranchTogether) {
  localization::UnambiguousSolverNode selector{""};
  localization::ambiguous_estimate_t candidates;
  candidates.pos1.pose = frc::Pose3d{-20_m, 3_m, 0_m, frc::Rotation3d{}};
  candidates.pos1.variance = 1;
  candidates.pos1.field_to_camera = frc::Pose3d{-19_m, 3_m, 1_m, frc::Rotation3d{}};
  candidates.pos2 = candidates.pos1;
  candidates.pos2->pose = frc::Pose3d{5_m, 3_m, 0_m, frc::Rotation3d{}};
  candidates.pos2->field_to_camera = frc::Pose3d{6_m, 3_m, 1_m, frc::Rotation3d{}};
  const auto result = selector.SolveSelected({&candidates}, true);
  ASSERT_TRUE(result);
  EXPECT_EQ(result->selected[0].estimate.field_to_camera->X(), 6_m);
}

TEST(JointSolver, RecoversRelativeTranslationsWithExactFrontAnchor) {
  const auto truth = Truth();
  auto initial = truth;
  initial[1].translation += Eigen::Vector3d{0.2, -0.15, 0.1};
  initial[2].translation += Eigen::Vector3d{-0.1, 0.25, -0.15};
  const auto result = calibration::SolveExtrinsics(Synthetic(truth), initial, 0);
  ASSERT_TRUE(result.usable) << result.solver_report;
  EXPECT_EQ(result.extrinsics[0].translation, initial[0].translation);
  EXPECT_EQ(result.extrinsics[0].rotation.coeffs(), initial[0].rotation.coeffs());
  EXPECT_EQ(result.initial_errors.pairs, 90u);  // All three pairs in 30 groups.
  EXPECT_LT(result.final_errors.translation_rms_m, 1e-7);
  for (size_t i = 1; i < truth.size(); ++i) {
    EXPECT_LT((result.extrinsics[i].translation - truth[i].translation).norm(), 1e-7);
    EXPECT_LT(result.extrinsics[i].rotation.angularDistance(truth[i].rotation), 1e-7);
  }
  const auto heldout = Synthetic(truth, 7);
  EXPECT_LT(calibration::EvaluatePairErrors(heldout, result.extrinsics).normalized_half_mean_squared, 1e-12);
}

TEST(JointSolver, AlternateAnchorAndDisconnectedCamera) {
  const auto truth = Truth();
  auto initial = truth;
  initial[0].translation.x() += 0.2;
  const auto result = calibration::SolveExtrinsics(Synthetic(truth), initial, 1);
  ASSERT_TRUE(result.usable);
  EXPECT_EQ(result.extrinsics[1].translation, truth[1].translation);
  EXPECT_EQ(result.extrinsics[1].rotation.coeffs(), truth[1].rotation.coeffs());
  auto groups = Synthetic(truth);
  for (auto& group : groups) group.pop_back();
  EXPECT_THROW(calibration::SolveExtrinsics(groups, initial, 0), std::invalid_argument);
  EXPECT_THROW(calibration::SolveExtrinsics({}, initial, 0), std::invalid_argument);
}

TEST(JointSolver, RotationPriorAndPairNormalizationIndependentOfReplayLength) {
  const auto truth = Truth();
  auto initial = truth;
  initial[1].rotation = initial[1].rotation * Eigen::AngleAxisd{0.12, Eigen::Vector3d::UnitX()};
  initial[2].rotation = initial[2].rotation * Eigen::AngleAxisd{-0.08, Eigen::Vector3d::UnitY()};
  const auto groups = Synthetic(truth, 5);
  const auto a = calibration::SolveExtrinsics(groups, initial, 0);
  auto repeated = groups;
  for (int i = 0; i < 3; ++i) repeated.insert(repeated.end(), groups.begin(), groups.end());
  const auto b = calibration::SolveExtrinsics(repeated, initial, 0);
  ASSERT_TRUE(a.usable);
  ASSERT_TRUE(b.usable);
  EXPECT_LT(a.final_errors.normalized_half_mean_squared, a.initial_errors.normalized_half_mean_squared);
  for (size_t i = 0; i < truth.size(); ++i) {
    EXPECT_LT(a.extrinsics[i].rotation.angularDistance(b.extrinsics[i].rotation), 1e-7);
    EXPECT_LT((a.extrinsics[i].translation - b.extrinsics[i].translation).norm(), 1e-7);
  }
  // The specified rotation penalty allows a compromise, rather than forcing
  // an exact rotation fit to the observations.
  EXPECT_GT(a.extrinsics[1].rotation.angularDistance(truth[1].rotation), 0.01);
  EXPECT_LT(a.extrinsics[1].rotation.angularDistance(truth[1].rotation),
            initial[1].rotation.angularDistance(truth[1].rotation));
}

class ReplayTest : public ::testing::Test {
 protected:
  std::filesystem::path root;
  std::vector<calibration::ReplayCamera> cameras;
  void SetUp() override {
    static std::atomic<int> serial{0};
    root = std::filesystem::temp_directory_path() /
        ("cos-extrinsics-" + std::to_string(getpid()) + "-" + std::to_string(serial++));
    std::filesystem::create_directories(root);
    for (size_t i = 0; i < 3; ++i) {
      const auto name = "camera" + std::to_string(i);
      const auto log = root / name;
      std::filesystem::create_directory(log);
      const auto config = root / (name + ".json");
      nlohmann::json json = {{"name", name}, {"width", 1280}, {"height", 800},
          {"intrinsics", {{"fx", 900}, {"fy", 900}, {"cx", 640}, {"cy", 400}}},
          {"extrinsics", {{"translation_x", 0.2}, {"translation_y", 0.1 * i},
              {"translation_z", 0.3}, {"rotation_x", 0}, {"rotation_y", -15}, {"rotation_z", 30 * i}}}};
      std::ofstream(config) << json;
      cameras.push_back({name, config, log});
    }
  }
  void TearDown() override { std::filesystem::remove_all(root); }
};

TEST_F(ReplayTest, EnumeratesEveryJpegInTimestampOrderAndRejectsInvalidJpegNames) {
  std::ofstream(cameras[0].log_directory / "1.010.jpg");
  std::ofstream(cameras[1].log_directory / "1.000.JPEG");
  std::ofstream(cameras[2].log_directory / "1.000.jpg");
  std::ofstream(cameras[0].log_directory / "notes.txt");
  const auto frames = calibration::EnumerateReplay(cameras);
  ASSERT_EQ(frames.size(), 3u);
  EXPECT_EQ(frames[0].camera, 1u);
  EXPECT_EQ(frames[1].camera, 2u);
  EXPECT_EQ(frames[2].camera, 0u);
  EXPECT_EQ(frames[2].capture_ns - frames[0].capture_ns, calibration::kMatchWindowNs);
  std::ofstream(cameras[0].log_directory / "bad.jpg");
  EXPECT_THROW(calibration::EnumerateReplay(cameras), std::invalid_argument);
}

TEST_F(ReplayTest, CollectsWholeReplayIncludingUnmatchedFramesAndRetainsPnPTransforms) {
  const frc::AprilTagFieldLayout layout{
      {{10, frc::Pose3d{0_m, 0_m, 0_m, frc::Rotation3d{0_rad, 0_rad, -90_deg}}}}, 20_m, 10_m};
  const camera::Intrinsics intrinsics{cameras[0].config_path};
  const cv::Mat rvec = (cv::Mat_<double>(3, 1) << 0.05, 0.4, 0);
  const cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0.03, 0, 0.6);
  std::vector<cv::Point2d> corners;
  cv::projectPoints(localization::kApriltagCorners, rvec, tvec,
      intrinsics.ToMatrix(), intrinsics.ToDistortionCoefficients(), corners);
  localization::tag_detection_t detection;
  detection.tag_id = 10;
  std::copy(corners.begin(), corners.end(), detection.corners.begin());
  const std::vector<calibration::ReplayFrame> frames{
      {0, 0, "0.jpg"}, {1, 1, "1.jpg"}, {2, 100'000'000, "2.jpg"},
      {0, 200'000'000, "3.jpg"}};
  std::vector<int64_t> visited;
  const auto replay = calibration::CollectObservations(cameras, frames,
      [&](const auto& frame) {
        visited.push_back(frame.capture_ns);
        return frame.capture_ns == 200'000'000 ? std::vector<localization::tag_detection_t>{} :
            std::vector<localization::tag_detection_t>{detection};
      }, false, layout);
  EXPECT_EQ(replay.jpegs_processed, 4u);
  EXPECT_EQ(visited, (std::vector<int64_t>{0, 1, 100'000'000, 200'000'000}));
  ASSERT_EQ(replay.observations.size(), 3u);
  EXPECT_EQ(calibration::MatchObservations(replay.observations).size(), 1u);
  const auto extrinsics = calibration::LoadExtrinsics(cameras);
  for (const auto& o : replay.observations)
    EXPECT_TRUE((o.field_to_camera.ToMatrix() * Matrix(extrinsics[o.camera]))
                    .isApprox(o.field_to_robot.ToMatrix(), 1e-8));
}

TEST_F(ReplayTest, MultiTagPoseRetainsCameraTransform) {
  const frc::AprilTagFieldLayout layout{
      {{10, frc::Pose3d{0_m, 0_m, 0_m, frc::Rotation3d{}}},
       {11, frc::Pose3d{0_m, 1_m, 0_m, frc::Rotation3d{}}}}, 20_m, 10_m};
  const camera::Intrinsics intrinsics{cameras[0].config_path};
  localization::MultiTagSolverNode pnp{"", "", intrinsics,
      camera::Extrinsics{cameras[0].config_path}, layout};
  const cv::Mat rvec = (cv::Mat_<double>(3, 1) << 0.1, 0.2, 0);
  const cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0.1, 0.1, 3);
  const cv::Mat flip = utils::MakeTransform(
      (cv::Mat_<double>(3, 1) << 0, std::numbers::pi, 0), cv::Mat::zeros(3, 1, CV_64F));
  std::vector<localization::tag_detection_t> detections;
  for (const auto& tag : layout.GetTags()) {
    const auto tag_matrix = utils::Pose3dToCvMat(tag.pose);
    std::vector<cv::Point3d> object_points;
    for (const auto& corner : localization::kApriltagCorners)
      object_points.push_back(utils::CvMatToPoint3d(tag_matrix * flip * utils::HomogenizePoint3d(corner)));
    std::vector<cv::Point2d> corners;
    cv::projectPoints(object_points, rvec, tvec, intrinsics.ToMatrix(), intrinsics.ToDistortionCoefficients(), corners);
    localization::tag_detection_t d;
    d.tag_id = tag.ID;
    std::copy(corners.begin(), corners.end(), d.corners.begin());
    detections.push_back(d);
  }
  const auto estimate = pnp.AmbiguousSolve(detections, false);
  ASSERT_TRUE(estimate);
  ASSERT_TRUE(estimate->pos1.field_to_camera);
  const auto expected = utils::ConvertOpencvTransformationMatrixToWpilibPose(utils::MakeTransform(rvec, tvec).inv());
  EXPECT_TRUE(expected.ToMatrix().isApprox(estimate->pos1.field_to_camera->ToMatrix(), 1e-7));
  const auto e = calibration::LoadExtrinsics(cameras);
  EXPECT_TRUE((expected.ToMatrix() * Matrix(e[0])).isApprox(estimate->pos1.pose.ToMatrix(), 1e-7));
}

TEST_F(ReplayTest, JsonRoundTripPreservesIntrinsicsAndProtectsInputs) {
  const auto original = calibration::LoadExtrinsics(cameras);
  auto candidate = original;
  candidate[1].translation += Eigen::Vector3d{0.15, -0.1, 0.05};
  candidate[1].rotation = candidate[1].rotation * Eigen::AngleAxisd{0.04, Eigen::Vector3d::UnitX()};
  const auto output = root / "output";
  calibration::WriteCandidateConfigs(cameras, candidate, output);
  auto written_cameras = cameras;
  for (auto& camera : written_cameras) camera.config_path = output / camera.config_path.filename();
  const auto written = calibration::LoadExtrinsics(written_cameras);
  for (size_t i = 0; i < cameras.size(); ++i) {
    EXPECT_LT((written[i].translation - candidate[i].translation).norm(), 1e-10);
    EXPECT_LT(written[i].rotation.angularDistance(candidate[i].rotation), 1e-10);
    std::ifstream a(cameras[i].config_path), b(written_cameras[i].config_path);
    const auto ja = nlohmann::json::parse(a), jb = nlohmann::json::parse(b);
    EXPECT_EQ(ja["intrinsics"], jb["intrinsics"]);
    EXPECT_EQ(ja["name"], jb["name"]);
    if (i == 0) {
      EXPECT_EQ(ja["extrinsics"], jb["extrinsics"]);
    }
  }
  EXPECT_THROW(calibration::WriteCandidateConfigs(cameras, candidate, root), std::invalid_argument);
  EXPECT_THROW(calibration::WriteCandidateConfigs(cameras, candidate, output), std::invalid_argument);
  const auto unchanged = calibration::LoadExtrinsics(cameras);
  EXPECT_EQ(unchanged[1].translation, original[1].translation);
}
}  // namespace
