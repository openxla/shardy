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

#ifndef SHARDY_COMMON_LOGGING_H_
#define SHARDY_COMMON_LOGGING_H_

#include <string>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/RawOstreamExtras.h"

namespace mlir {
namespace sdy {
namespace log {

// Enum for logging severity levels
enum LogSeverity {
  INFO,
  WARNING,
  ERROR,
  FATAL,  // Used by SDY_LOG(FATAL) and SDY_CHECK
};

// Helper struct to capture file/line/function info for logs/checks
struct LogMessageData {
  LogSeverity severity;
  const char* file;
  int line;
  const char* conditionStr = nullptr;  // Optional: for CHECK messages
};

// Base class for logging messages. This is where the stream operator overloads
// will live, allowing us to build the error string.
class LogMessage {
 public:
  LogMessage(LogMessageData data);

  // The destructor is where the log message is printed.
  // This ensures the full message is built before reporting.
  ~LogMessage();

  // Returns the internal string stream. This allows chained '<<' operations.
  llvm::raw_ostream& stream();

 protected:
  LogMessageData data;
  std::string message;
  llvm::raw_string_ostream strStream;
};

class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(LogMessageData failureData);

  // The destructor is where the fatal error reporting happens.
  // This ensures the full message is built before reporting.
  [[noreturn]] ~LogMessageFatal();
};

}  // namespace log
}  // namespace sdy
}  // namespace mlir

#define SDY_LOG(severity) SDY_INTERNAL_CHECK_OR_LOG(severity, nullptr)

// TODO(tomnatan): make the verbose level actually work.
#define SDY_VLOG(verbose_level) \
  (verbose_level > 0) ? mlir::thread_safe_nulls() : SDY_LOG(INFO)
#define SDY_VLOG_IS_ON(verbose_level) false

#define SDY_CHECK(condition)              \
  (condition) ? mlir::thread_safe_nulls() \
              : SDY_INTERNAL_CHECK_OR_LOG(FATAL, #condition)

#define SDY_CHECK_EQ(val1, val2) \
  SDY_INTERNAL_CHECK_OP(==, val1, val2)

#define SDY_CHECK_NE(val1, val2) \
  SDY_INTERNAL_CHECK_OP(!=, val1, val2)

#define SDY_CHECK_LE(val1, val2) \
  SDY_INTERNAL_CHECK_OP(<=, val1, val2)

#define SDY_CHECK_LT(val1, val2) \
  SDY_INTERNAL_CHECK_OP(<, val1, val2)

#define SDY_CHECK_GE(val1, val2) \
  SDY_INTERNAL_CHECK_OP(>=, val1, val2)

#define SDY_CHECK_GT(val1, val2) \
  SDY_INTERNAL_CHECK_OP(>, val1, val2)

// =================================================================
// == Implementation logs, do not rely on anything below here. ==
// =================================================================

#define SDY_INTERNAL_CHECK_OR_LOG(severity, condition_str)                  \
  (mlir::sdy::log::severity == mlir::sdy::log::FATAL                        \
       ? mlir::sdy::log::LogMessageFatal(                                   \
             {mlir::sdy::log::FATAL, __FILE__, __LINE__, condition_str})    \
             .stream()                                                      \
       : mlir::sdy::log::LogMessage(                                        \
             {mlir::sdy::log::severity, __FILE__, __LINE__, condition_str}) \
             .stream())

#define SDY_INTERNAL_CHECK_OP(op, val1, val2) \
  SDY_CHECK(val1 op val2) << "(" << val1 << " vs. " << val2 << ") "

#endif  // SHARDY_COMMON_LOGGING_H_
