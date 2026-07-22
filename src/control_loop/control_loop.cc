#include "control_loop/control_loop.h"

#include <chrono>
#include <unordered_set>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"

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
  thread_ = std::jthread([this](const std::stop_token& stop_token) -> void {
    while (!stop_token.stop_requested()) {
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

      std::this_thread::sleep_for(period_);
      context.reset();

      if (!destructed) {
        LOG(WARNING) << "Command loop overun";
        stop_source.request_stop();
        destructed.wait(false);
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
    std::function<void(const Context&)> callback) {
  callbacks_.emplace_back(callback);
}

void ControlLoop::RegisterDependancy(
    std::function<void(const Context&)> dependancy) {
  dependencies_.emplace_back(dependancy);
}

void ControlLoop::RegisterNode(const std::shared_ptr<INode>& node) {
  nodes_.emplace_back(node);
}

void ControlLoop::ValidateNodeGraph() {
  std::unordered_map<std::string, std::type_index> publishers;
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

}  // namespace control_loop
