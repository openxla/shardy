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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_OPTIMIZE_PIPELINE_SCHEDULE_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_OPTIMIZE_PIPELINE_SCHEDULE_H_

#include <cstddef>
#include <functional>
#include <optional>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"

namespace mlir::mpmd {

enum class PipelineSchedule {
  kNone,
  k1F1B,
  kGPipe,
  kCircular,
  kCircularWithReversedBackward,
  kGPipeBut1F1BForLastMesh,
  kZeroBubbleH1,
  kZeroBubbleH2ZeroTxLatency,
  kZeroBubbleH2HalfTxLatency,
  kZeroBubbleH2FullTxLatency,
  kParallelPipelinesWithWrapAround,
};

// Parses the given string as a `PipelineSchedule` or return std::nullopt if
// parsing failed.
std::optional<PipelineSchedule> ParsePipelineSchedule(StringRef schedule_str);

// Returns a string representation of the given `schedule`.
std::string ToString(PipelineSchedule schedule);

using FragmentComparator =
    std::function<bool(FragmentOp, FragmentOp)>;

// Returns a fragment comparator for the given pipeline schedule.
FragmentComparator BuiltinFragmentComparator(PipelineSchedule schedule);

// A `FragmentComparator` option with a custom parser/printer.
struct FragmentComparatorOption {
  FragmentComparator value;
  // If the comparator was derived from a built-in `PipelineSchedule`, this
  // will contain the schedule, otherwise it will be `std::nullopt`.
  std::optional<PipelineSchedule> schedule;

  static FragmentComparatorOption GetBuiltIn(PipelineSchedule schedule) {
    return FragmentComparatorOption{BuiltinFragmentComparator(schedule)};
  }
};

// This is needed by MLIR, however we can't really print anything meaningful
// about a specific instance.
llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              FragmentComparatorOption comparator);

}  // namespace mlir::mpmd

namespace llvm::cl {

extern template class basic_parser<mlir::mpmd::FragmentComparatorOption>;

template <>
class parser<mlir::mpmd::FragmentComparatorOption>
    : public basic_parser<mlir::mpmd::FragmentComparatorOption> {
 public:
  parser(Option& opt) : basic_parser(opt) {}
  bool parse(Option& opt, StringRef argName, StringRef arg,
             mlir::mpmd::FragmentComparatorOption& value);
  StringRef getValueName() const override { return "fragment-comparator"; }
  void printOptionDiff(const Option& opt,
                       const mlir::mpmd::FragmentComparatorOption& value,
                       const OptVal& defaultValue, size_t globalWidth) const;
  void anchor() override;
};

}  // namespace llvm::cl

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_OPTIMIZE_PIPELINE_SCHEDULE_H_
