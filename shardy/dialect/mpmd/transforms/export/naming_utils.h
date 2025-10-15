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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_EXPORT_NAMING_UTILS_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_EXPORT_NAMING_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "mlir/IR/Attributes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::mpmd {

// Returns an informative string for an array of UserOrigin attributes, suitable
// for becoming a fragment function symbol name, by compressing together any
// consecutive blocks that have been parsed with the same head and also have
// consecutive counters. For example:
//    ["f_1", "f_2", "f_3", "loss", "f_3"(1), "f_2"(1), "f_1"(1)]
// will print as:
//    "f_1:3.loss.f(1)_1:3"
std::string GetInformativeFragmentName(ArrayRef<Attribute> origin);

// Small utility to truncate a string up to max_length.
std::string Truncate(StringRef str, int64_t max_length);

// Returns a name for a fragment based on its metadata, i.e., its user origins,
// stage id and call counter, when defined.
//
// The returned name has the following format:
//   <name_from_origin>_<phase>
//
// Where `name_from_origin` includes a summary of all user origins:
//   - "stage<stage_id>" if `stage_id` is defined.
//   - "inferred" if the list of origins is empty.
//   - or the most frequent name from the list of origins, followed by a summary
//   of counters (e.g., block_1:3 for block 1, block 2, and block 3) followed by
//   "..." if there are other names different from the most frequent one.
//
// `phase` is a summary of the phases from the list of origins:
//   - "fwd" for each origin with transpose count 0.
//   - "bwd" for each origin with transpose count 1.
//   - "transpose<counter> for each origin with transpose count > 1.
//
// Note we don't include `call_counter` as part of the metadata used to generate
// the name, as fragments are often reused across different calls.
std::string GetFullNameFromMetadata(ArrayRef<Attribute> origins,
                                    std::optional<int64_t> stage_id,
                                    bool all_forward = false);

// A call site is a pair of fragment name and optional call counter.
using CallSite = std::pair<std::string, std::optional<uint32_t>>;

using MeshToCallSites = DenseMap<StringRef, SmallVector<CallSite>>;

// Given a map from mesh names to lists of `CallSite`s, returns a name for the
// group of fragments that share the same code. The name is a best effort
// attempt to summarize the call sites across all meshes.
//
// The returned name has the following format:
//   <name>_<call_counter_summary>
//
// Where `name` is the name of the first (in lexicographic order) call site,
// computed from each fragment's metadata, and `call_counter_summary` is a
// summary of the call counters across all meshes, or empty if there is no call
// counter or if the call counters are inconsistent across meshes.
//
// Example:
//   - stage2_fwd_calls0to1
//   - stage2_fwd_calls1from0
//
// The call counter summary of each mesh is computed as follows:
//   - If there's a single call site, then we simply return "_call{counter}" or
// empty string otherwise.
//   - Otherwise, we check if the different call-sites are consistent with each
// other, assuming the different call sites result from a (forward or backward)
// loop over the same fragment (as it would happen with microbatching).
// In particular, if the different call sites have different names, or are not
// contiguously numbered, or they are not in ascending or descending order
// (based on a program order), then they most likely don't belong to the same
// loop. In these cases, we return an empty string. But when they are indeed
// consistent, then we return: "_calls{min_counter}to{max_counter}" for the
// ascending case, and "_calls{max_counter}from{min_counter}" for the descending
// case.
std::string GetCallSitesSummaryName(const MeshToCallSites& mesh_call_sites);

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_EXPORT_NAMING_UTILS_H_
