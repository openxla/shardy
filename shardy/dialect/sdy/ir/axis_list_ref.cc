
/* Copyright 2024 The Shardy Authors.

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

#include "shardy/dialect/sdy/ir/axis_list_ref.h"

#include <cstdint>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/transforms/common/macros.h"

namespace mlir {
namespace sdy {

bool AxisListRef::operator<(const AxisListRef& rhs) const {
  if (size() != rhs.size()) {
    return size() < rhs.size();
  }
  for (auto [axisRef, rhsAxisRef] : llvm::zip_equal(axisRefs, rhs.axisRefs)) {
    if (axisRef != rhsAxisRef) {
      return axisRef < rhsAxisRef;
    }
  }
  if (tailAxisRef != rhs.tailAxisRef) {
    return tailAxisRef < rhs.tailAxisRef;
  }
  return false;
}

SmallVector<AxisRefAttr> AxisListRef::toVector() const {
  if (empty()) {
    return {};
  }
  SmallVector<AxisRefAttr> resultAxes = llvm::to_vector(axisRefs);
  resultAxes.push_back(tailAxisRef);
  return resultAxes;
}

// Checks if this axes is a strict prefix of the axes of `rhs`.
bool AxisListRef::strictPrefixOf(const AxisListRef& rhs) const {
  if (empty()) {
    return !rhs.empty();
  }
  if (size() > rhs.size()) {
    return false;
  }
  for (auto [axisRef, rhsAxisRef] : llvm::zip(axisRefs, rhs.axisRefs)) {
    if (axisRef != rhsAxisRef) {
      return false;
    }
  }
  if (size() == rhs.size()) {
    return tailAxisRef.strictPrefixOf(rhs.tailAxisRef);
  }
  return tailAxisRef.prefixOf(rhs.axisRefs[axisRefs.size()]);
}

// Returns the product of the sharding sizes of all individual axes
int64_t AxisListRef::getShardingSize(MeshAttr mesh) const {
  if (empty()) {
    return 1;
  }
  int64_t shardingSize = 1;
  for (AxisRefAttr axisRef : axisRefs) {
    shardingSize *= axisRef.getSize(mesh);
  }
  return shardingSize * tailAxisRef.getSize(mesh);
};

void AxisListRef::clear() {
  axisRefs = {};
  tailAxisRef = AxisRefAttr();
}

void AxisListRef::trim(int64_t newSizeExcludingNewTail,
                       std::optional<AxisRefAttr> newTailAxisRef) {
  if (!newTailAxisRef) {
    if (newSizeExcludingNewTail == 0) {
      clear();
    } else {
      tailAxisRef = axisRefs[newSizeExcludingNewTail - 1];
      axisRefs = axisRefs.take_front(newSizeExcludingNewTail - 1);
    }
    return;
  }
  axisRefs = axisRefs.take_front(newSizeExcludingNewTail);
  tailAxisRef = *newTailAxisRef;
}

std::optional<AxisRefAttr> AxisListRef::getPrefixOfInputWithoutOverlap(
    AxisRefAttr axisRef) const {
  if (empty()) {
    return axisRef;
  }
  AxisRefAttr prefixAxisRef = axisRef;
  for (AxisRefAttr againstAxisRef : axisRefs) {
    SDY_ASSIGN_OR_RETURN_IF_NULLOPT(
        prefixAxisRef, prefixAxisRef.getPrefixWithoutOverlap(againstAxisRef));
  }
  SDY_ASSIGN_OR_RETURN_IF_NULLOPT(
      prefixAxisRef, prefixAxisRef.getPrefixWithoutOverlap(tailAxisRef));
  return prefixAxisRef;
}

bool AxisListRef::truncateWithoutOverlap(const AxisListRef& rhs) {
  if (empty()) {
    return false;
  }
  for (const auto& [axisRefIndex, axisRef] : llvm::enumerate(axisRefs)) {
    if (auto prefixAxisRef = rhs.getPrefixOfInputWithoutOverlap(axisRef);
        prefixAxisRef != axisRef) {
      trim(axisRefIndex, prefixAxisRef);
      return true;
    }
  }
  if (auto prefixAxisRef = rhs.getPrefixOfInputWithoutOverlap(tailAxisRef);
      prefixAxisRef != tailAxisRef) {
    trim(axisRefs.size(), prefixAxisRef);
    return true;
  }
  return false;
}

}  // namespace sdy
}  // namespace mlir
