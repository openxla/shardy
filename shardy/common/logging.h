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
namespace detail {

// Helper struct to capture file/line info
struct CheckFailureData {
  const char* file;
  int line;
  const char* conditionStr;
};

// Base class for logging messages. This is where the stream operator overloads
// will live, allowing us to build the error string.
class LogMessage {
 public:
  LogMessage(CheckFailureData data);

  // Returns the internal string stream. This allows chained '<<' operations.
  llvm::raw_ostream& stream();

 protected:
  CheckFailureData data;
  std::string message;
  llvm::raw_string_ostream strStream;
};

class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(CheckFailureData failureData);

  // The destructor is where the fatal error reporting happens.
  // This ensures the full message is built before reporting.
  [[noreturn]] ~LogMessageFatal();
};

}  // namespace detail
}  // namespace sdy
}  // namespace mlir

#define SDY_CHECK(condition)                                                 \
  (condition)                                                                \
      ? mlir::thread_safe_nulls()                                            \
      : mlir::sdy::detail::LogMessageFatal({__FILE__, __LINE__, #condition}) \
            .stream()

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
// == Implementation details, do not rely on anything below here. ==
// =================================================================

#define SDY_INTERNAL_CHECK_OP(op, val1, val2) \
  SDY_CHECK(val1 op val2) << "(" << val1 << " vs. " << val2 << ") "

#endif  // SHARDY_COMMON_LOGGING_H_
