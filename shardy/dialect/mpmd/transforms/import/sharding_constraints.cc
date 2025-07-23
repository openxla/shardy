/* Copyright 2025 The MPMD Authors.

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

#include "shardy/dialect/mpmd/transforms/import/sharding_constraints.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::mpmd {

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                        const InputOutputEquishardingConstraint& constraint) {
  return os << constraint.input_index << ":" << constraint.output_index;
}

}  // namespace mlir::mpmd

namespace llvm::cl {
using ::mlir::mpmd::InputOutputEquishardingConstraint;

template class basic_parser<InputOutputEquishardingConstraint>;

bool parser<InputOutputEquishardingConstraint>::parse(
    Option& opt, StringRef, StringRef arg,
    InputOutputEquishardingConstraint& value) {
  auto [input_index_str, output_index_str] = arg.split(':');
  auto& [input_index, output_index] = value;
  if (input_index_str.getAsInteger(10, input_index)) {
    return opt.error("Failed to parse input index: " + input_index_str);
  }
  if (output_index_str.getAsInteger(10, output_index)) {
    return opt.error("Failed to parse output index: " + output_index_str);
  }
  return false;
}

void parser<InputOutputEquishardingConstraint>::anchor() {}

}  // namespace llvm::cl
