#include "logging/wpilog_writer.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <typeindex>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace logging {
namespace {

template <typename T>
auto RegisterValue(wpi::log::DataLogWriter& log, FieldRegistrar& registrar)
    -> index_t {
  using Message = control_loop::ValueMessage<T>;
  if constexpr (std::is_same_v<T, bool>) {
    return registrar.Add<Message>(log, "", [](const Message& message) {
      return message.value;
    });
  } else if constexpr (std::is_integral_v<T>) {
    return registrar.Add<Message>(log, "", [](const Message& message) {
      if constexpr (std::is_unsigned_v<T> &&
                    sizeof(T) >= sizeof(std::int64_t)) {
        if (message.value >
            static_cast<T>(std::numeric_limits<std::int64_t>::max())) {
          throw std::out_of_range("Unsigned WPILog integer exceeds int64_t");
        }
      }
      return static_cast<std::int64_t>(message.value);
    });
  } else if constexpr (std::is_same_v<T, long double>) {
    return registrar.Add<Message>(log, "", [](const Message& message) {
      return static_cast<double>(message.value);
    });
  } else {
    return registrar.Add<Message>(log, "", [](const Message& message) {
      return message.value;
    });
  }
}

}  // namespace

WPILogWriter::WPILogWriter(
    std::string_view filename,
    const std::vector<control_loop::MessageDescriptor>& publications) {
  std::error_code error;
  log_ = std::make_unique<wpi::log::DataLogWriter>(filename, error);
  if (error) {
    throw std::system_error(error, "Cannot open WPILog file");
  }

  std::unordered_set<std::string> channels;
  // The file has one DataLogWriter, but every publication gets its own field
  // registrations. The channel is a runtime value from MessageDescriptor.
  for (const auto& publication : publications) {
    const auto& info = publication.GetPublicationInfo();
    if (!info.has_value() || publication.GetTypes().size() != 1 ||
        !publication.GetTypes().contains(info->message_type)) {
      throw std::invalid_argument("Publication lacks concrete type: " +
                                  publication.GetChannel());
    }
    if (!channels.insert(publication.GetChannel()).second) {
      throw std::invalid_argument("Duplicate publication: " +
                                  publication.GetChannel());
    }

    // The generated RegisterWPILog method calls RegisterFields<T>, which calls
    // registrar.Add once per leaf. Add creates entries such as
    // "master/sub1/variance" or "master/sub2/variance" for the same T.
    FieldRegistrar registrar(
        publication.GetChannel(),
        [this](std::string path, std::string channel,
               std::unique_ptr<ILogField> field) -> index_t {
          return AddField(std::move(path), std::move(channel),
                          std::move(field));
        });
    if (!info->is_class || info->value_type == typeid(std::string)) {
      RegisterPrimitive(publication, *log_);
      continue;
    }
    if (!info->register_logs.has_value()) {
      throw std::invalid_argument("Missing WPILog registration: " +
                                  publication.GetChannel());
    }
    const auto first_index = slots_.size();
    const auto indices = (*info->register_logs)(*log_, registrar);
    if (indices.size() != slots_.size() - first_index) {
      throw std::invalid_argument("Incomplete WPILog registration: " +
                                  publication.GetChannel());
    }
    for (std::size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] != first_index + i) {
        throw std::invalid_argument("Invalid WPILog registration index: " +
                                    publication.GetChannel());
      }
    }
  }
}

WPILogWriter::~WPILogWriter() {
  slots_.clear();
  log_->Flush();
  log_->Stop();
}

auto WPILogWriter::AddField(std::string path, std::string channel,
                            std::unique_ptr<ILogField> field) -> index_t {
  for (const auto& [existing_path, index] : log_paths_) {
    if (existing_path == path) {
      throw std::invalid_argument("Duplicate WPILog path: " + path);
    }
  }
  const index_t index = slots_.size();
  // Keep the publication channel with the actual WPILib entry object. At log
  // time this selects the matching message from ContextInternal.
  slots_.push_back({std::move(channel), std::move(field)});
  log_paths_.emplace_back(std::move(path), index);
  return index;
}

void WPILogWriter::RegisterPrimitive(
    const control_loop::MessageDescriptor& publication,
    wpi::log::DataLogWriter& log) {
  const auto type = publication.GetPublicationInfo()->value_type;
  FieldRegistrar registrar(
      publication.GetChannel(),
      [this](std::string path, std::string channel,
             std::unique_ptr<ILogField> field) -> index_t {
        return AddField(std::move(path), std::move(channel), std::move(field));
      });
  if (type == typeid(bool)) {
    RegisterValue<bool>(log, registrar);
  } else if (type == typeid(char)) {
    RegisterValue<char>(log, registrar);
  } else if (type == typeid(signed char)) {
    RegisterValue<signed char>(log, registrar);
  } else if (type == typeid(unsigned char)) {
    RegisterValue<unsigned char>(log, registrar);
  } else if (type == typeid(char8_t)) {
    RegisterValue<char8_t>(log, registrar);
  } else if (type == typeid(char16_t)) {
    RegisterValue<char16_t>(log, registrar);
  } else if (type == typeid(char32_t)) {
    RegisterValue<char32_t>(log, registrar);
  } else if (type == typeid(wchar_t)) {
    RegisterValue<wchar_t>(log, registrar);
  } else if (type == typeid(short)) {
    RegisterValue<short>(log, registrar);
  } else if (type == typeid(unsigned short)) {
    RegisterValue<unsigned short>(log, registrar);
  } else if (type == typeid(int)) {
    RegisterValue<int>(log, registrar);
  } else if (type == typeid(unsigned int)) {
    RegisterValue<unsigned int>(log, registrar);
  } else if (type == typeid(long)) {
    RegisterValue<long>(log, registrar);
  } else if (type == typeid(unsigned long)) {
    RegisterValue<unsigned long>(log, registrar);
  } else if (type == typeid(long long)) {
    RegisterValue<long long>(log, registrar);
  } else if (type == typeid(unsigned long long)) {
    RegisterValue<unsigned long long>(log, registrar);
  } else if (type == typeid(float)) {
    RegisterValue<float>(log, registrar);
  } else if (type == typeid(double)) {
    RegisterValue<double>(log, registrar);
  } else if (type == typeid(long double)) {
    RegisterValue<long double>(log, registrar);
  } else if (type == typeid(std::string)) {
    RegisterValue<std::string>(log, registrar);
  } else {
    throw std::invalid_argument("Unsupported primitive publication: " +
                                publication.GetChannel());
  }
}

void WPILogWriter::Log(const control_loop::ContextInternal& context) {
  std::lock_guard lock(mutex_);
  for (const auto& [path, index] : log_paths_) {
    const auto& slot = slots_.at(index);
    // Two publications of the same message type have different slot.channel
    // values, so each appends to the entries created for its own channel.
    const auto* message =
        context.GetMessage<control_loop::IMessage>(slot.channel);
    if (message != nullptr && !slot.field->Append(*message)) {
      throw std::runtime_error("WPILog message type mismatch: " + path);
    }
  }
}

void WPILogWriter::Flush() {
  std::lock_guard lock(mutex_);
  log_->Flush();
}

}  // namespace logging
