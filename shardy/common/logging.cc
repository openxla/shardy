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

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace sdy {
namespace detail {

LogMessage::LogMessage(CheckFailureData data)
    : data(data), message(), strStream(message) {
  strStream << "[" << data.file << ":" << data.line
            << "]: Check failed: " << data.conditionStr << " ";
}

llvm::raw_ostream& LogMessage::stream() { return strStream; }

LogMessageFatal::LogMessageFatal(CheckFailureData failureData)
    : LogMessage(failureData) {}

LogMessageFatal::~LogMessageFatal() {
  llvm::report_fatal_error(message.c_str());
}

}  // namespace detail
}  // namespace sdy
}  // namespace mlir
