#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <wpi/DataLogReader.h>
#include <wpi/MemoryBuffer.h>

namespace wpilog_test {

// The callback must consume the record before returning: reader owns its data.
template <class Visit>
void VisitLogValues(const std::filesystem::path& path, Visit visit) {
  auto buffer = wpi::MemoryBuffer::GetFile(path.string());
  if (!buffer) throw std::runtime_error("Cannot read WPILog: " + path.string());
  wpi::log::DataLogReader reader(std::move(*buffer));
  if (!reader.IsValid()) throw std::runtime_error("Invalid WPILog: " + path.string());

  std::unordered_map<int, std::string> names;
  for (const auto& record : reader) {
    if (record.IsStart()) {
      wpi::log::StartRecordData start;
      if (!record.GetStartData(&start)) {
        throw std::runtime_error("Invalid WPILog start record");
      }
      names.emplace(start.entry, start.name);
    } else if (!record.IsControl()) {
      visit(names.at(record.GetEntry()), record);
    }
  }
}

}  // namespace wpilog_test
