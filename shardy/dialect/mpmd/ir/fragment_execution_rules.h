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

#ifndef SHARDY_DIALECT_MPMD_IR_FRAGMENT_EXECUTION_RULES_H_
#define SHARDY_DIALECT_MPMD_IR_FRAGMENT_EXECUTION_RULES_H_

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/PatternMatch.h"
#include "shardy/dialect/mpmd/ir/dialect.h"

namespace mlir::mpmd {

// Describes the origin of a fragment.
struct FragmentOrigin {
  std::string computation_name;
  int64_t transpose_count = 0;

  bool operator==(const FragmentOrigin& other) const {
    return computation_name == other.computation_name &&
           transpose_count == other.transpose_count;
  }

  bool operator!=(const FragmentOrigin& other) const {
    return !(*this == other);
  }

  bool operator<(const FragmentOrigin& other) const {
    return std::tie(computation_name, transpose_count) <
           std::tie(other.computation_name, other.transpose_count);
  }

  template <typename OS>
  friend OS& operator<<(OS& os, const FragmentOrigin& origin) {
    os << "\"" << origin.computation_name << "\"";
    if (origin.transpose_count > 0) {
      os << "(" << origin.transpose_count << ")";
    }
    return os;
  }

  friend llvm::hash_code hash_value(const FragmentOrigin& origin) {
    return llvm::hash_combine(origin.computation_name, origin.transpose_count);
  }
};

// Enum to represent the type of fragment split behavior.
// These values are determined by the presence of specific MLIR attributes
// on fragment operations during compilation.
enum class SplitFragmentType {
  // Indicates this fragment portion retains transferred data from the original
  // fragment. Set when the fragment has the kSplitKeepTransferredAttrName
  // attribute.
  kKeepTransferred,
  // Indicates this fragment portion drops transferred data from the original
  // fragment. Set when the fragment has the kSplitDropTransferredAttrName
  // attribute.
  kDropTransferred
};

template <typename OS>
inline OS& operator<<(OS& os, const SplitFragmentType& type) {
  switch (type) {
    case SplitFragmentType::kKeepTransferred:
      os << "kKeepTransferred";
      break;
    case SplitFragmentType::kDropTransferred:
      os << "kDropTransferred";
      break;
  }
  return os;
}

template <typename OS>
inline OS& operator<<(OS& os, const std::optional<SplitFragmentType>& type) {
  if (type.has_value()) {
    os << *type;
  } else {
    os << "nullopt";
  }
  return os;
}

// Holds the metadata of a fragment.
struct FragmentInfo {
  std::vector<FragmentOrigin> origins;
  std::optional<int> stage_id;
  std::optional<int> call_counter;
  std::optional<SplitFragmentType> split_type;

  bool operator==(const FragmentInfo& other) const {
    return llvm::equal(origins, other.origins) && stage_id == other.stage_id &&
           call_counter == other.call_counter && split_type == other.split_type;
  }

  bool operator!=(const FragmentInfo& other) const { return !(*this == other); }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const FragmentInfo& info) {
    os << "FragmentInfo(origins=[";
    llvm::interleave(info.origins, os, ",");
    os << "]";
    if (info.stage_id.has_value()) {
      os << ",stage=" << *info.stage_id;
    }
    if (info.call_counter.has_value()) {
      os << ",call_counter=" << *info.call_counter;
    }
    // Intentionally do not print split_type if it is nullopt
    if (info.split_type.has_value()) {
      os << ",split_type=" << *info.split_type;
    }
    os << ")";
    return os;
  }
};

struct FragmentInfoMapInfo : public DenseMapInfo<FragmentInfo> {
  static unsigned getHashValue(const FragmentInfo& info) {
    return llvm::hash_combine(llvm::hash_combine_range(info.origins),
                              info.stage_id, info.call_counter,
                              info.split_type);
  }
  static bool isEqual(const FragmentInfo& lhs, const FragmentInfo& rhs) {
    return lhs == rhs;
  }

  static inline FragmentInfo getEmptyKey() {
    return FragmentInfo{/*origins=*/{},
                        /*stage_id=*/DenseMapInfo<int>::getEmptyKey(),
                        /*call_counter=*/DenseMapInfo<int>::getEmptyKey(),
                        /*split_type=*/std::nullopt};
  }

  static inline FragmentInfo getTombstoneKey() {
    return FragmentInfo{/*origins=*/{},
                        /*stage_id=*/DenseMapInfo<int>::getTombstoneKey(),
                        /*call_counter=*/DenseMapInfo<int>::getTombstoneKey(),
                        /*split_type=*/SplitFragmentType::kDropTransferred};
  }
};

// Describes a rule to merge fragments. A rule is defined by a list of sources
// and a target. It is applied to a set of fragments when each one matches one
// of the source info objects. The result of merging the source fragments is
// labelled with the target info.
struct FragmentMergeRule {
  std::vector<FragmentInfo> sources;
  FragmentInfo target;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const FragmentMergeRule& rule) {
    os << "FragmentMergeRule(sources=[";
    llvm::interleave(rule.sources, os, ",");
    return os << "],target=" << rule.target << ")";
  }
};

using FragmentMergeRules = std::vector<FragmentMergeRule>;

// Describes a rule for scheduling fragments. A rule is defined by an ordered
// sequence of fragments. This ordering dictates the execution order of the
// fragments on a given mesh.
struct FragmentScheduleRule {
  // The sequence of fragments to be scheduled. The order of fragments in this
  // vector defines their execution order.
  std::vector<FragmentInfo> ordered_fragments;

  static constexpr llvm::StringRef kFragmentScheduleRulePrefix =
      "FragmentScheduleRule(ordered_fragments=[";

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const FragmentScheduleRule& rule) {
    os << kFragmentScheduleRulePrefix;
    llvm::interleave(rule.ordered_fragments, os, "->");
    return os << "])";
  }
};

using FragmentScheduleRules = std::vector<FragmentScheduleRule>;

// Returns the fragment info of a fragment op.
FragmentInfo GetFragmentInfo(FragmentOp fragment);

// Sets the fragment info of a fragment op. Overwrites any existing info.
void SetFragmentInfo(FragmentOp fragment, const FragmentInfo& metadata,
                     RewriterBase& rewriter);

// Returns the split fragment type of a fragment op. If the fragment op is not
// split, returns std::nullopt.
std::optional<SplitFragmentType> GetSplitFragmentType(FragmentOp fragment);

}  // namespace mlir::mpmd

namespace llvm::cl {

extern template class basic_parser<mlir::mpmd::FragmentMergeRule>;

template <>
class parser<mlir::mpmd::FragmentMergeRule>
    : public basic_parser<mlir::mpmd::FragmentMergeRule> {
 public:
  parser(Option& opt) : basic_parser(opt) {}
  bool parse(Option& opt, StringRef argName, StringRef arg,
             mlir::mpmd::FragmentMergeRule& value);
  StringRef getValueName() const override { return "fragment-merge-rule"; }
  void printOptionDiff(const Option& opt,
                       const mlir::mpmd::FragmentMergeRule& value,
                       const OptVal& defaultValue, size_t globalWidth) const;
  void anchor() override;
};

extern template class basic_parser<mlir::mpmd::FragmentScheduleRule>;

template <>
class parser<mlir::mpmd::FragmentScheduleRule>
    : public basic_parser<mlir::mpmd::FragmentScheduleRule> {
 public:
  parser(Option& opt) : basic_parser(opt) {}
  bool parse(Option& opt, StringRef argName, StringRef arg,
             mlir::mpmd::FragmentScheduleRule& value);
  StringRef getValueName() const override { return "fragment-schedule-rule"; }
  void printOptionDiff(const Option& opt,
                       const mlir::mpmd::FragmentScheduleRule& value,
                       const OptVal& defaultValue, size_t globalWidth) const;
  void anchor() override;
};

}  // namespace llvm::cl

#endif  // SHARDY_DIALECT_MPMD_IR_FRAGMENT_EXECUTION_RULES_H_
