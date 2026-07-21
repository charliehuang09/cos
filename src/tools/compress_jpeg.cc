#include <filesystem>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"

namespace fs = std::filesystem;

ABSL_FLAG(std::string, input_folder, "",           // NOLINT
          "Input folder filled with .jpg files");  // NOLINT

ABSL_FLAG(std::string, output_folder, "",                          // NOLINT
          "Output folder to be filled with encoded jpeg images");  // NOLINT

ABSL_FLAG(int, compression, 90,           // NOLINT
          "How much compression 0-100");  // NOLINT

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);

  CHECK(absl::GetFlag(FLAGS_input_folder) != "");
  CHECK(absl::GetFlag(FLAGS_output_folder) != "");

  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY,
                             absl::GetFlag(FLAGS_compression)};

  fs::path output_path = absl::GetFlag(FLAGS_output_folder);
  for (const auto& entry :
       fs::directory_iterator(absl::GetFlag(FLAGS_input_folder))) {
    if (!entry.is_regular_file()) {
      continue;
    }

    cv::Mat image = cv::imread(entry.path().string());
    if (image.empty()) {
      std::cerr << "Skipping " << entry.path() << '\n';
      continue;
    }

    std::vector<uchar> jpeg_data;
    if (!cv::imencode(".jpg", image, jpeg_data, params)) {
      std::cerr << "Failed to encode " << entry.path() << '\n';
      continue;
    }

    fs::path output_file_path =
        output_path / (entry.path().stem().string() + ".jpg");

    std::ofstream out(output_file_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(jpeg_data.data()),
              jpeg_data.size());
  }
}
