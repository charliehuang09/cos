#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "control_loop/node.h"

namespace localization {

class VarianceCalculatorNode final : public control_loop::INode {
 public:
  VarianceCalculatorNode(std::string_view input_channel,
                         std::string_view output_channel,
                         double min_variance = 1.0,
                         double variance_scalar = 0.7);

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;

 private:
  std::string input_channel_;
  std::string output_channel_;
  double min_variance_;
  double variance_scalar_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

}  // namespace localization
