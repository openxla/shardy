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

#include "shardy/dialect/mpmd/ir/fragment_execution_rules.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/utils.h"

namespace mlir::mpmd {

namespace {

std::vector<FragmentOrigin> GetFragmentOrigins(FragmentOp op) {
  std::vector<FragmentOrigin> origins;
  for (Attribute origin_attr : op.getOrigin()) {
    auto user_origin = cast<UserOriginAttr>(origin_attr);
    origins.push_back(
        FragmentOrigin{/*computation_name=*/user_origin.getUserName().str(),
                       /*transpose_count=*/user_origin.getTransposeCount()});
  }
  llvm::sort(origins);
  return origins;
}

// Parses a fragment origin string of the form
// "computation_name(transpose_count)" where "(transpose_count)" is optional.
bool ParseFragmentOrigin(llvm::cl::Option& opt, llvm::StringRef& arg,
                         FragmentOrigin& origin) {
  arg.consume_front("\"");
  auto [name, rest] = arg.split('"');
  if (name == arg) {
    return opt.error("Expected a '\"'");
  }
  origin.computation_name = name;
  origin.transpose_count = 0;
  arg = rest;
  if (!arg.consume_front("(")) {
    return false;
  }
  if (arg.consumeInteger(10, origin.transpose_count)) {
    return opt.error("Expected a transpose count");
  }
  if (!arg.consume_front(")")) {
    return opt.error("Expected ')'");
  }
  return false;
}

// Parses a fragment info string of the form
// "FragmentInfo(origins=[<origin>,...],stage=<n>,call_counter=<m>)"
// where the stage and call_counter fields are optional.
bool ParseFragmentInfo(llvm::cl::Option& opt, llvm::StringRef& arg,
                       FragmentInfo& info) {
  if (!arg.consume_front("FragmentInfo(origins=[")) {
    return opt.error("Expected 'FragmentInfo(origins=['");
  }
  while (!arg.starts_with("]")) {
    if (ParseFragmentOrigin(opt, arg, info.origins.emplace_back())) {
      return true;  // opt.error was called inside ParseFragmentOrigin
    }
    if (!arg.consume_front(",")) {
      break;
    }
  }
  if (!arg.consume_front("]")) {
    return opt.error("Expected ']'");
  }
  while (arg.consume_front(",")) {
    if (arg.consume_front("stage=")) {
      if (info.stage_id.has_value()) {
        return opt.error("'stage' specified more than once");
      }
      int stage_id;
      if (arg.consumeInteger(10, stage_id)) {
        return opt.error("Expected an integer value for 'stage'");
      }
      info.stage_id = stage_id;
    } else if (arg.consume_front("call_counter=")) {
      if (info.call_counter.has_value()) {
        return opt.error("'call_counter' specified more than once");
      }
      int call_counter;
      if (arg.consumeInteger(10, call_counter)) {
        return opt.error("Expected an integer value for 'call_counter'");
      }
      info.call_counter = call_counter;
    } else if (arg.consume_front("split_type=")) {
      if (info.split_type.has_value()) {
        return opt.error("'split_type' specified more than once");
      }
      if (arg.consume_front("kKeepTransferred")) {
        info.split_type = SplitFragmentType::kKeepTransferred;
      } else if (arg.consume_front("kDropTransferred")) {
        info.split_type = SplitFragmentType::kDropTransferred;
      } else {
        return opt.error(
            "Expected 'kKeepTransferred' or 'kDropTransferred' for "
            "'split_type'");
      }
    } else {
      return opt.error(
          "Expected 'stage=', 'call_counter=', or "
          "'split_type=' after ','");
    }
  }
  if (!arg.consume_front(")")) {
    return opt.error("Expected ')'");
  }
  return false;
}

}  // namespace

FragmentInfo GetFragmentInfo(FragmentOp fragment) {
  std::optional<uint64_t> stage_id;
  if (fragment.getStageIdAttr()) {
    stage_id = fragment.getStageIdAttr().getInt();
  }
  std::optional<int64_t> call_counter = TryToFindCallCounter(fragment);
  std::vector<FragmentOrigin> origins = GetFragmentOrigins(fragment);
  std::optional<SplitFragmentType> split_type;
  if (IsSplitKeepTransferred(fragment)) {
    split_type = SplitFragmentType::kKeepTransferred;
  } else if (IsSplitDropTransferred(fragment)) {
    split_type = SplitFragmentType::kDropTransferred;
  }
  return FragmentInfo{origins, stage_id, call_counter, split_type};
}

void SetFragmentInfo(FragmentOp fragment, const FragmentInfo& metadata,
                     RewriterBase& rewriter) {
  std::vector<Attribute> new_origins;
  for (const auto& origin : metadata.origins) {
    Attribute attr = UserOriginAttr::get(
        rewriter.getContext(),
        StringAttr::get(rewriter.getContext(), origin.computation_name),
        origin.transpose_count);
    new_origins.push_back(attr);
  }

  fragment.setOriginAttr(ArrayAttr::get(rewriter.getContext(), new_origins));

  fragment.setStageId(metadata.stage_id);

  if (metadata.call_counter.has_value()) {
    fragment->setAttr(kCallCounterAttrName,
                      rewriter.getUI32IntegerAttr(*metadata.call_counter));
  } else {
    fragment->removeAttr(kCallCounterAttrName);
  }

  // Handle split type attributes
  if (metadata.split_type.has_value()) {
    if (*metadata.split_type == SplitFragmentType::kDropTransferred) {
      fragment->setAttr(kSplitDropTransferredAttrName,
                        UnitAttr::get(rewriter.getContext()));
      fragment->removeAttr(kSplitKeepTransferredAttrName);
    } else if (*metadata.split_type == SplitFragmentType::kKeepTransferred) {
      fragment->setAttr(kSplitKeepTransferredAttrName,
                        UnitAttr::get(rewriter.getContext()));
      fragment->removeAttr(kSplitDropTransferredAttrName);
    }
  } else {
    fragment->removeAttr(kSplitDropTransferredAttrName);
    fragment->removeAttr(kSplitKeepTransferredAttrName);
  }
}

}  // namespace mlir::mpmd

namespace llvm::cl {

using ::mlir::mpmd::FragmentMergeRule;
using ::mlir::mpmd::FragmentScheduleRule;

template class basic_parser<FragmentMergeRule>;

// Parses a fragment merge rule string of the form
// "FragmentMergeRule(sources=[<source>,...],target=<target>)"
// <source> and <target> are FragmentInfo strings.
bool parser<FragmentMergeRule>::parse(Option& opt, StringRef, StringRef arg,
                                      FragmentMergeRule& value) {
  if (!arg.consume_front("FragmentMergeRule(sources=[")) {
    return opt.error("Expected 'FragmentMergeRule(sources=['");
  }
  while (!arg.starts_with("]")) {
    if (mlir::mpmd::ParseFragmentInfo(opt, arg, value.sources.emplace_back())) {
      return true;  // opt.error was called inside ParseFragmentInfo
    }
    if (!arg.consume_front(",")) {
      break;
    }
  }
  if (!arg.consume_front("],target=")) {
    return opt.error("Expected ',' or '],target='");
  }
  if (mlir::mpmd::ParseFragmentInfo(opt, arg, value.target)) {
    return true;  // opt.error was called inside ParseFragmentInfo
  }
  if (!arg.consume_front(")")) {
    return opt.error("Expected ')'");
  }
  return false;
}

void parser<FragmentMergeRule>::printOptionDiff(const Option& opt,
                                                const FragmentMergeRule& value,
                                                const OptVal& defaultValue,
                                                size_t globalWidth) const {
  printOptionName(opt, globalWidth);
  outs() << "= " << value << "\n";
}

void parser<FragmentMergeRule>::anchor() {}

template class basic_parser<FragmentScheduleRule>;

// Parses a fragment schedule rule string of the form
// "FragmentScheduleRule(ordered_fragments=[<fragment1>-><fragment2>...])"
// <fragment>s are FragmentInfo strings.
bool parser<FragmentScheduleRule>::parse(Option& opt, StringRef, StringRef arg,
                                         FragmentScheduleRule& value) {
  if (!arg.consume_front(FragmentScheduleRule::kFragmentScheduleRulePrefix)) {
    return opt.error("Expected '" +
                     FragmentScheduleRule::kFragmentScheduleRulePrefix + "'");
  }
  while (!arg.starts_with("]")) {
    if (mlir::mpmd::ParseFragmentInfo(opt, arg,
                                      value.ordered_fragments.emplace_back())) {
      return true;  // opt.error was called inside ParseFragmentInfo
    }
    if (!arg.consume_front("->")) {
      break;
    }
  }
  if (!arg.consume_front("])")) {
    return opt.error("Expected '])'");
  }
  return false;
}

void parser<FragmentScheduleRule>::printOptionDiff(
    const Option& opt, const FragmentScheduleRule& value,
    const OptVal& defaultValue, size_t globalWidth) const {
  printOptionName(opt, globalWidth);
  outs() << "= " << value << "\n";
}

void parser<FragmentScheduleRule>::anchor() {}

}  // namespace llvm::cl
