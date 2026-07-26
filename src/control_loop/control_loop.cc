#include "control_loop/control_loop.h"

#include <chrono>
#include <unordered_set>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "control_loop/timer.h"

using namespace std::chrono_literals;

namespace control_loop {

ContextInternal::ContextInternal(std::chrono::steady_clock::time_point start,
                                 ControlLoop* control_loop,
                                 std::stop_token stop_token,
                                 std::atomic<bool>* destructed)
    : start(start),
      control_loop(control_loop),
      stop_token(std::move(stop_token)),
      destructed(destructed) {}

ContextInternal::~ContextInternal() {
  destructed->store(true);
  destructed->notify_all();
}

ControlLoop::ControlLoop(std::chrono::milliseconds period) : period_(period) {}

void ControlLoop::Start() {
  ValidateNodeGraph();
  RegisterNodeCallbacks();

  thread_ = std::jthread([this](const std::stop_token& stop_token) -> void {
    while (!stop_token.stop_requested()) {
      Timer loop_timer;
      std::stop_source stop_source;
      std::atomic destructed = false;

      Context context(new ContextInternal(std::chrono::steady_clock::now(),
                                          this, stop_source.get_token(),
                                          &destructed));

      for (const auto& dependancy : dependencies_) {
        dependancy(context);
      }

      for (const auto& callback : callbacks_) {
        callback(context);
      }

      Timer timer;
      std::this_thread::sleep_for(period_.value_or(0ms));
      context.reset();

      if (!destructed) {
        std::ignore = stop_source.request_stop();
        destructed.wait(false);
        if (period_.has_value()) {
          LOG(WARNING) << "Command loop overrun! " << timer.Stop().count()
                       << "s loop";
        }
      }

      auto time = loop_timer.Stop();
      if (log_latency_ && time.count() > kMinLoopSeconds) {
        timestamp_queue_.push(loop_timer.GetStart());
        if (timestamp_queue_.size() > kTimestampQueueMaxSize) {
          timestamp_queue_.pop();
          const std::chrono::duration<double> elapsed =
              loop_timer.GetStart() - timestamp_queue_.front();
          loops_per_second_ = static_cast<double>(timestamp_queue_.size() - 1) /
                              elapsed.count();
          LOG(INFO) << "Average loops per second: " << loops_per_second_;
        }
        LOG(INFO) << "Control loop took " << time.count() << "s";
      }
    }
  });
}

void ControlLoop::Stop() {
  thread_.request_stop();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void ControlLoop::RegisterCallback(
    const std::function<void(const Context&)>& callback) {
  callbacks_.emplace_back(callback);
}

void ControlLoop::RegisterDependancy(
    const std::function<void(const Context&)>& dependancy) {
  dependencies_.emplace_back(dependancy);
}

void ControlLoop::RegisterNode(const std::shared_ptr<INode>& node) {
  nodes_.emplace_back(node);
}
void ControlLoop::RegisterDependancyNode(const std::shared_ptr<INode>& node) {
  dependancy_nodes_.emplace_back(node);
  dependencies_.emplace_back(node->CreateCallback());
}

void ControlLoop::EnableLatencyLog() {
  log_latency_ = true;
}

void ControlLoop::ValidateNodeGraph() {
  std::unordered_map<std::string, std::type_index> publishers;
  for (const auto& node : dependancy_nodes_) {
    for (const auto& message_descriptor : node->GetPublications()) {
      PCHECK(!publishers.contains(message_descriptor.GetChannel()))
          << "Multiple publishers to the same channel. Channel is: "
          << message_descriptor.GetChannel();
      PCHECK(message_descriptor.GetTypes().size() == 1)
          << "Publisher message descriptor has multiple types. Channel is: "
          << message_descriptor.GetChannel();
      publishers.insert({message_descriptor.GetChannel(),
                         *message_descriptor.GetTypes().begin()});
    }
  }
  for (const auto& node : nodes_) {
    for (const auto& message_descriptor : node->GetPublications()) {
      PCHECK(!publishers.contains(message_descriptor.GetChannel()))
          << "Multiple publishers to the same channel. Channel is: "
          << message_descriptor.GetChannel();
      PCHECK(message_descriptor.GetTypes().size() == 1)
          << "Publisher message descriptor has multiple types. Channel is: "
          << message_descriptor.GetChannel();
      publishers.insert({message_descriptor.GetChannel(),
                         *message_descriptor.GetTypes().begin()});
    }
  }
  for (const auto& node : nodes_) {
    for (const auto& message_descriptor : node->GetDependencies()) {
      PCHECK(publishers.contains(message_descriptor.GetChannel()))
          << "Node channel dependancy does has not been registered. Channel "
             "is: "
          << message_descriptor.GetChannel();
      PCHECK(message_descriptor.GetTypes().contains(
          publishers.at(message_descriptor.GetChannel())))
          << "Publisher and subscriber channel type does not match. Channel "
             "is: "
          << message_descriptor.GetChannel();
    }
  }
}

void ControlLoop::RegisterNodeCallbacks() {
  std::unordered_map<std::string, INode*> publishers;
  for (const auto& node : dependancy_nodes_) {
    for (const auto& message_descriptor : node->GetPublications()) {
      publishers.insert({message_descriptor.GetChannel(), node.get()});
    }
  }
  for (const auto& node : nodes_) {
    for (const auto& message_descriptor : node->GetPublications()) {
      publishers.insert({message_descriptor.GetChannel(), node.get()});
    }
  }

  for (const auto& node : nodes_) {
    for (const auto& message_descriptor : node->GetDependencies()) {
      publishers.at(message_descriptor.GetChannel())
          ->RegisterCallback(node->CreateCallback());
    }
  }
}

auto ControlLoop::GetLoopsPerSecond() const -> double {
  return loops_per_second_;
}

}  // namespace control_loop
