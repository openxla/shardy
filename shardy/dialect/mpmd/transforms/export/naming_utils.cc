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

#include "shardy/dialect/mpmd/transforms/export/naming_utils.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/utils.h"

namespace mlir::mpmd {

namespace {

// Splits a string of the form "abcd_21354" to (a StringRef) "abcd" and (an
// integer) 21354. If the input cannot be split like this, then the function
// returns failure(); otherwise success().
// TODO(dvytin): we might need to generalize this based on further examples.
LogicalResult SplitToPrefixAndCounter(StringRef name, StringRef& prefix,
                                            int64_t& counter) {
  size_t last_separator_pos = name.find_last_of('_');
  if (last_separator_pos != StringRef::npos) {
    StringRef suffix = name.substr(last_separator_pos + 1);
    if (!suffix.consumeInteger(/*Radix=*/10, counter) && suffix.empty()) {
      prefix = name.substr(0, last_separator_pos);
      return success();
    }
  }
  return failure();
}

// Parses the origin attribute and returns a string in a way that also takes any
// transpose count into account, that we refer to as ``head''. If the original
// name of the origin was splittable into a prefix and a counter, it also
// returns the counter otherwise returns nullopt. Some examples:
//    #mpmd.origin<"foo"(2)>    ~>  head="foo(2)", counter=nullopt
//    #mpmd.origin<"boo_1">     ~>  head="boo",    counter=1
//    #mpmd.origin<"boo_1"(3)>  ~>  head="boo(3)", counter=1
std::pair<std::string, std::optional<int64_t>> ParseToHeadAndCounter(
    UserOriginAttr origin_attr, bool keep_transpose_count = true) {
  StringRef origin_ref = origin_attr.getUserName().getValue();

  StringRef head_ref;
  int64_t counter = 0;
  bool did_split =
      SplitToPrefixAndCounter(origin_ref, head_ref, counter).succeeded();

  std::string head;
  llvm::raw_string_ostream stream(head);
  stream << (did_split ? head_ref : origin_ref);

  if (keep_transpose_count && origin_attr.getTransposeCount() > 0) {
    stream << "(" << origin_attr.getTransposeCount() << ")";
  }
  std::optional<int64_t> parsed_counter =
      did_split ? std::make_optional(counter) : std::nullopt;
  return std::make_pair(std::move(head), parsed_counter);
}

// An auxiliary struct for creating an informative fragment name for sequences
// of UserOrigin attributes.
struct PrinterState {
  std::string head;
  // TODO(dvytin): modify to allow bwd origin counters to be printed in reverse.
  int64_t min;
  int64_t max;

  std::string ToString() const {
    std::string ret = StrCat(this->head, "_", this->min);
    if (this->max != this->min) {
      return StrCat(ret, ":", this->max);
    }
    return ret;
  }
};

}  // namespace

std::string Truncate(StringRef str, int64_t max_length) {
  SDY_CHECK_GE(max_length, 32);  // A comfortable width to fit the unique id.
  StringRef delimiter = "<...>";
  if (str.size() > max_length) {
    return StrCat(str.substr(0, max_length - delimiter.size()), delimiter);
  }
  return str.str();
}

std::string GetInformativeFragmentName(ArrayRef<Attribute> origin) {
  std::vector<std::string> constructed;
  std::optional<PrinterState> current_state = std::nullopt;

  for (Attribute attr : origin) {
    UserOriginAttr origin_attr = cast<UserOriginAttr>(attr);
    const auto& [head, parsed_counter] = ParseToHeadAndCounter(origin_attr);
    if (!parsed_counter.has_value()) {
      // Failed to split. Dump and reset any current state.
      if (current_state.has_value()) {
        constructed.push_back((*current_state).ToString());
        current_state = std::nullopt;
      }
      // And then dump this attribute's head.
      constructed.push_back(std::move(head));
      continue;
    }
    // Successful split.
    if (!current_state.has_value()) {
      current_state = PrinterState{std::move(head), /*min=*/*parsed_counter,
                                   /*max=*/*parsed_counter};
      continue;
    }
    // Current state has value (and successful split).
    if (head == current_state->head) {
      // Same head.
      if (*parsed_counter == current_state->min - 1) {
        current_state->min = *parsed_counter;
      } else if (*parsed_counter == current_state->max + 1) {
        current_state->max = *parsed_counter;
      } else {
        constructed.push_back((*current_state).ToString());
        current_state->min = *parsed_counter;
        current_state->max = *parsed_counter;
      }
    } else {
      // Different head.
      constructed.push_back((*current_state).ToString());
      current_state->head = std::move(head);
      current_state->min = *parsed_counter;
      current_state->max = *parsed_counter;
    }
  }
  if (current_state.has_value()) {
    constructed.push_back((*current_state).ToString());
  }

  std::string result;
  llvm::raw_string_ostream stream(result);

  if (constructed.empty()) {
    stream << "inferred";
  }
  llvm::interleave(constructed, stream, ".");
  return result;
}

namespace {

inline constexpr StringRef kForwardKeyword = "fwd";
inline constexpr StringRef kBackwardKeyword = "bwd";
inline constexpr StringRef kTransposeKeyword = "transpose";

std::string GetMostFrequentName(ArrayRef<Attribute> origins) {
  // Keeps track of each name and the counters associated with it.
  // We use a sorted map since in case of draws in popularity, we use
  // lexicographic order to break ties.
  std::map<std::string, std::vector<std::optional<int64_t>>, std::less<>>
      name_to_counters;
  bool is_bwd = true;
  for (Attribute attr : origins) {
    UserOriginAttr origin_attr = cast<UserOriginAttr>(attr);
    const auto& [head, parsed_counter] =
        ParseToHeadAndCounter(origin_attr, /*keep_transpose_count=*/false);
    name_to_counters[head].push_back(parsed_counter);
    is_bwd = is_bwd && origin_attr.getTransposeCount() == 1;
  }

  // When we have many candidates for naming, we pick the most popular one and
  // append "..." to the end of the name.
  StringRef tail = name_to_counters.size() == 1 ? "" : "...";

  // Find the most popular name.
  auto it =
      llvm::max_element(name_to_counters, [](const auto& a, const auto& b) {
        return a.second.size() < b.second.size();
      });
  std::set<int64_t> counters;
  for (const auto& counter : it->second) {
    if (counter.has_value()) {
      counters.insert(*counter);
    }
  }
  if (counters.empty()) {
    // There's no counter associated with this name.
    return StrCat(it->first, tail);
  }
  if (counters.size() == 1) {
    return StrCat(it->first, "_", *counters.begin(), tail);
  }
  // TODO: jupvfranco - Consider the rare cases in which the counters are not
  // contiguous.
  int64_t min = *llvm::min_element(counters);
  int64_t max = *llvm::max_element(counters);
  return is_bwd ? StrCat(it->first, "_", max, ":", min, tail)
                : StrCat(it->first, "_", min, ":", max, tail);
}

void GetPhasesFromUserOrigins(ArrayRef<Attribute> origins,
                              std::vector<std::string>& constructed) {
  std::set<int64_t> transpose_counts;
  for (Attribute attr : origins) {
    auto origin = cast<UserOriginAttr>(attr);
    transpose_counts.insert(origin.getTransposeCount());
  }
  llvm::transform(transpose_counts, std::back_inserter(constructed),
                  [](int64_t transpose_count) -> std::string {
                    if (transpose_count == 0) {
                      return kForwardKeyword.str();
                    }
                    if (transpose_count == 1) {
                      return kBackwardKeyword.str();
                    }
                    return StrCat(kTransposeKeyword, transpose_count);
                  });
}

}  // namespace

// Note: unique id and function name aren't included.
std::string GetFullNameFromMetadata(ArrayRef<Attribute> origins,
                                    std::optional<int64_t> stage_id) {
  std::vector<std::string> constructed;

  // Step 0. Find a name for the fragment.
  if (stage_id.has_value()) {
    constructed.push_back(StrCat("stage", *stage_id));
  } else if (origins.empty()) {
    constructed.push_back("inferred");
  } else {
    constructed.push_back(GetMostFrequentName(origins));
  }

  // Step 1. Find the phases.
  GetPhasesFromUserOrigins(origins, constructed);

  std::string result;
  llvm::raw_string_ostream stream(result);
  llvm::interleave(constructed, stream, "_");
  return result;
}

namespace {

bool AllHaveCallCounters(ArrayRef<CallSite> call_sites) {
  return llvm::all_of(call_sites, [](const CallSite& call_site) {
    return call_site.second.has_value();
  });
}

bool IsAscendingAndContiguous(ArrayRef<CallSite> call_sites) {
  for (int i = 1; i < call_sites.size(); ++i) {
    if (*call_sites[i - 1].second + 1 != *call_sites[i].second) {
      return false;
    }
  }
  return true;
}

bool IsDescendingAndContiguous(ArrayRef<CallSite> call_sites) {
  for (int i = 1; i < call_sites.size(); ++i) {
    if (*call_sites[i - 1].second - 1 != *call_sites[i].second) {
      return false;
    }
  }
  return true;
}

// Requires: call_sites.size() > 1;
bool IsNamingConsistent(ArrayRef<CallSite> call_sites) {
  SDY_CHECK_GT(call_sites.size(), 1);
  for (const CallSite& call_site : call_sites) {
    if (call_site.first != call_sites[0].first) {
      return false;
    }
  }
  return true;
}

std::optional<std::string> GetMeshCallCountersSummary(
    ArrayRef<CallSite> call_sites) {
  SDY_CHECK(!call_sites.empty());

  if (call_sites.size() == 1) {
    auto [name, call_counter] = call_sites[0];
    if (call_counter.has_value()) {
      return StrCat("call", *call_counter);
    }
    return std::nullopt;
  }

  if (!IsNamingConsistent(call_sites)) {
    return std::nullopt;
  }

  if (!AllHaveCallCounters(call_sites)) {
    return std::nullopt;
  }
  auto compare_fn = [](const CallSite& a, const CallSite& b) {
    return a.second.value() < b.second.value();
  };

  if (IsAscendingAndContiguous(call_sites)) {
    const uint32_t min = *llvm::min_element(call_sites, compare_fn)->second;
    const uint32_t max = *llvm::max_element(call_sites, compare_fn)->second;
    return StrCat("calls", min, "to", max);
  }

  if (IsDescendingAndContiguous(call_sites)) {
    const uint32_t min = *llvm::min_element(call_sites, compare_fn)->second;
    const uint32_t max = *llvm::max_element(call_sites, compare_fn)->second;
    return StrCat("calls", max, "from", min);
  }

  return std::nullopt;
}

}  // namespace

std::string GetCallSitesSummaryName(const MeshToCallSites& mesh_call_sites) {
  std::set<std::string> all_summaries;
  bool has_name_without_call_counter = false;
  // Keep all names in a sorted set to ensure deterministic behavior.
  std::set<std::string> all_names;
  for (const auto& [_, call_sites] : mesh_call_sites) {
    all_names.insert(call_sites[0].first);
    std::optional<std::string> summary = GetMeshCallCountersSummary(call_sites);
    if (summary.has_value()) {
      all_summaries.insert(*summary);
    } else {
      has_name_without_call_counter = true;
    }
  }

  SDY_CHECK(!all_names.empty());
  StringRef name = *all_names.begin();
  if (all_summaries.size() == 1 && !has_name_without_call_counter) {
    const std::optional<std::string>& summary = *all_summaries.begin();
    return StrCat(name, "_", *summary);
  }
  return name.str();
}

}  // namespace mlir::mpmd
