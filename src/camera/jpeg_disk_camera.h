#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <queue>
#include <random>

#include "control_loop/node.h"

namespace camera {

class JpegDiskCamera final : public control_loop::INode {
 public:
  // A zero frequency disables the corresponding fault. Otherwise, each source
  // frame has a 1/N chance of being skipped, or every Nth frame is emitted as
  // an empty JpegBuffer.
  JpegDiskCamera(std::string_view folder_path, std::string_view output_channel,
                 bool stop_when_empty = true, bool replay_all_frames = false,
                 std::size_t skip_frame_frequency = 0,
                 std::size_t empty_frame_frequency = 0);
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;

 private:
  void Callback(const control_loop::Context& context);
  auto GetTimestamp(const std::filesystem::path& path) -> std::optional<double>;

 private:
  std::string output_channel_;
  bool stop_when_empty_;
  bool replay_all_frames_;
  std::size_t skip_frame_frequency_;
  std::size_t empty_frame_frequency_;
  std::size_t frame_count_ = 0;
  static constexpr unsigned int kRandomSeed = 0xC0FFEEU;
  std::mt19937 random_engine_{kRandomSeed};
  std::queue<std::pair<std::filesystem::path, double>> file_paths_;
  std::optional<double> replay_start_time_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

}  // namespace camera
