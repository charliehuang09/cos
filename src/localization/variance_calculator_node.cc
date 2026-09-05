#include "localization/variance_calculator_node.h"

#include <memory>
#include <numeric>
#include <utility>

#include "localization/position.h"
#include "localization/solver_common.h"

namespace localization {

VarianceCalculatorNode::VarianceCalculatorNode(std::string_view input_channel,
                                               std::string_view output_channel,
                                               double min_variance,
                                               double variance_scalar)
    : input_channel_(input_channel),
      output_channel_(output_channel),
      min_variance_(min_variance),
      variance_scalar_(variance_scalar),
      dependencies_({{input_channel_, typeid(PositionEstimateMessage)}}),
      publications_({{output_channel_, typeid(PositionEstimateMessage)}}) {}

auto VarianceCalculatorNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) {
    auto notify_callbacks = [this, &context]() -> void {
      for (const auto& callback : callbacks_) {
        callback(context);
      }
    };

    const auto input =
        context->GetMessage<PositionEstimateMessage>(input_channel_);
    if (input == nullptr || input->distances.empty() ||
        input->tag_ids.size() != input->distances.size()) {
      context->SetMessage(output_channel_, nullptr);
      notify_callbacks();
      return;
    }

    auto output = std::make_unique<PositionEstimateMessage>(*input);
    const double distance_sum =
        std::accumulate(input->distances.begin(), input->distances.end(), 0.0);
    const int num_tags = static_cast<int>(input->distances.size());
    output->variance =
        Variance(num_tags, distance_sum / num_tags, min_variance_,
                 variance_scalar_);
    context->SetMessage(output_channel_, std::move(output));
    notify_callbacks();
  };
}

auto VarianceCalculatorNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto VarianceCalculatorNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

void VarianceCalculatorNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

}  // namespace localization
