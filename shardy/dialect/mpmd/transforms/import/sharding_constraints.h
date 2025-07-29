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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_SHARDING_CONSTRAINTS_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_SHARDING_CONSTRAINTS_H_

#include <cstdint>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::mpmd {

// A constraint enforcing that an input and an output (of the same global shape)
// should be partitioned the same way.
struct InputOutputEquishardingConstraint {
  InputOutputEquishardingConstraint(): input_index(-1), output_index(-1) {};
  InputOutputEquishardingConstraint(int64_t input_index, int64_t output_index)
      : input_index(input_index), output_index(output_index) {}

  int64_t input_index;
  int64_t output_index;

  bool operator==(const InputOutputEquishardingConstraint& other) const {
    return input_index == other.input_index &&
           output_index == other.output_index;
  }
};

llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os, const InputOutputEquishardingConstraint& constraint);

}  // namespace mlir::mpmd

namespace llvm::cl {

using ::mlir::mpmd::InputOutputEquishardingConstraint;

extern template class basic_parser<InputOutputEquishardingConstraint>;

template <>
class parser<InputOutputEquishardingConstraint>
    : public basic_parser<InputOutputEquishardingConstraint> {
 public:
  parser(Option& opt) : basic_parser(opt) {}
  bool parse(Option& opt, StringRef argName, StringRef arg,
             InputOutputEquishardingConstraint& value);
  StringRef getValueName() const override {
    return "input-output-equisharding-constraint";
  }
  void anchor() override;
};

}  // namespace llvm::cl

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_SHARDING_CONSTRAINTS_H_
