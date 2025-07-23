/* Copyright 2025 The Shardy Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "shardy/common/logging.h"

#include <chrono>  // NOLINT(build/c++11)
#include <cstdint>
#include <ctime>

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace sdy {
namespace log {

namespace {

static llvm::ManagedStatic<llvm::sys::Mutex> logMutex;

char severityToChar(LogSeverity s) {
  switch (s) {
    case INFO:
      return 'I';
    case WARNING:
      return 'W';
    case ERROR:
      return 'E';
    case FATAL:
      return 'F';
  }
  llvm_unreachable("Unknown LogSeverity");
}

// Function to convert std::time_t to std::tm (thread-safe)
inline std::tm toLocalTime(std::time_t epochSeconds) {
  std::tm tmBuf;  // User-provided buffer for thread-safety
#ifdef _WIN32
  localtime_s(&tmBuf, &epochSeconds);  // Windows-specific thread-safe version
#else  // This branch will be taken on macOS, Linux, and other POSIX systems
  localtime_r(&epochSeconds, &tmBuf);  // POSIX-specific thread-safe version
#endif
  return tmBuf;  // Return by value
}

}  // namespace

LogMessage::LogMessage(LogMessageData data)
    : data(data), message(), strStream(message) {
  auto nowPoint = std::chrono::system_clock::now();
  int64_t epochMicroseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(
          nowPoint.time_since_epoch())
          .count();
  int64_t epochSecondsNumeric = epochMicroseconds / 1000000;
  int64_t remainingMicroseconds = epochMicroseconds % 1000000;

  std::time_t currentTime = std::chrono::system_clock::to_time_t(nowPoint);
  std::tm localTime = toLocalTime(currentTime);

  strStream << llvm::formatv(
      "{0}{1:02}{2:02} {3:02}:{4:02}:{5}.{6:06} {7} {8}:{9}] ",
      // {0} Severity character
      severityToChar(data.severity),
      // {1}{2} MMDD
      localTime.tm_mon + 1, localTime.tm_mday,
      // {3}:{4}: HH:MM:
      localTime.tm_hour, localTime.tm_min,
      // {5}.{6} Epoch Seconds.Microseconds
      epochSecondsNumeric, remainingMicroseconds,
      // {7} PID
      llvm::sys::Process::getProcessId(),
      // {9}:{9}] File:Line]
      data.file, data.line);

  // If it's a CHECK, add the condition string
  if (data.conditionStr) {
    strStream << "Check failed: " << data.conditionStr << " ";
  }
}

LogMessage::~LogMessage() {
  llvm::sys::ScopedLock guard(*logMutex);
  strStream.flush();
  switch (data.severity) {
    case INFO:
      llvm::outs() << message << "\n";
      llvm::outs().flush();
      break;
    case WARNING:
    case ERROR:
      llvm::errs() << message << "\n";
      llvm::errs().flush();
      break;
    case FATAL:
      // Message will be reported in ~LogMessageFatal.
      break;
  }
}

llvm::raw_ostream& LogMessage::stream() { return strStream; }

LogMessageFatal::LogMessageFatal(LogMessageData failureData)
    : LogMessage(failureData) {}

LogMessageFatal::~LogMessageFatal() {
  strStream.flush();
  llvm::report_fatal_error(message.c_str());
}

}  // namespace log
}  // namespace sdy
}  // namespace mlir
