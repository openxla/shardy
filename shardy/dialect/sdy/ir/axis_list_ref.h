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

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "shardy/dialect/sdy/ir/dialect.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

// A reference to a list of AxisRefs, that provides API for trimming the list
// such that the last AxisRef can be replaced, while keeping it a reference.
//
// This is useful for getting a prefix of the list of axes, where the last axis
// is also a prefix sub-axis of an axis in the original list.
class AxisListRef {
 public:
  // Assumes that input `tailAxisRef` is non-empty.
  // TODO(enver): Drop this ctor, or make it private, as it is not used at the
  // moment.
  AxisListRef(ArrayRef<AxisRefAttr> axisRefs, AxisRefAttr tailAxisRef)
      : axisRefs(axisRefs), tailAxisRef(tailAxisRef) {}

  // Assumes that input `axisRefs` is non-empty.
  AxisListRef(ArrayRef<AxisRefAttr> axisRefs)
      : axisRefs(axisRefs.drop_back()), tailAxisRef(axisRefs.back()) {}

  AxisListRef() = default;
  AxisListRef(bool isTombstone) : isTombstone(isTombstone) {}

  // TODO(enver): Define an iterator that iterates on the concatenation of
  // axisRefs and tail, and use it for the methods below.

  // Checks if the axes is empty.
  bool empty() const {
    // If `tailAxisRef` is empty, then `axisRefs` is empty as well. Hence, it is
    // sufficient to check if `tailAxisRef` empty.
    return !tailAxisRef;
  }

  int64_t size() const { return empty() ? 0 : axisRefs.size() + 1; }

  bool operator<(const AxisListRef& rhs) const;

  bool operator==(const AxisListRef& rhs) const {
    return axisRefs == rhs.axisRefs && tailAxisRef == rhs.tailAxisRef;
  }

  // Checks if any two axes, one from this, and the other from `rhs`, overlap.
  bool overlaps(const AxisListRef& rhs) const;

  SmallVector<AxisRefAttr> toVector() const;

  std::pair<ArrayRef<AxisRefAttr>, AxisRefAttr> toPair() const {
    return std::make_pair(axisRefs, tailAxisRef);
  }

  // Checks if this axes is a strict prefix of the axes of `rhs`.
  bool strictPrefixOf(const AxisListRef& rhs) const;

  // Returns the product of the sharding sizes of all individual axes
  int64_t getShardingSize(MeshAttr mesh) const;

  // Returns the product of the sharding sizes of all individual axes excluding
  // the `prefix`.
  //
  // Assumes `prefix` is a prefix of this `AxisListRef`.
  int64_t getShardingSize(MeshAttr mesh, const AxisListRef& prefix) const {
    return getShardingSize(mesh) / prefix.getShardingSize(mesh);
  }

 private:
  // The axes that this FactorAxesPair holds is defined by `axisRefs` and
  // `tailAxisRef` together as the concatantion of the two. If `tailAxisRef` is
  // empty, then `axisRefs` is empty as well.
  ArrayRef<AxisRefAttr> axisRefs;
  AxisRefAttr tailAxisRef;
  // TODO(enver): Use ArrayRef::getTombstoneKey or AxisRefAttr::getTombstoneKey,
  // either for `axisRefs` or `tailAxisRef` respectively, instead.
  bool isTombstone = false;
  // Checks if `axisRef` overlaps with axes of this FactorAxesPair.
  // Assumes `axisRef` is non-empty.
  bool overlaps(AxisRefAttr axisRef) const;
};

struct AxisListRefInfo : public llvm::DenseMapInfo<AxisListRef> {
  static unsigned getHashValue(const AxisListRef& m) {
    return llvm::hash_value(m.toPair());
  }
  static bool isEqual(const AxisListRef& lhs, const AxisListRef& rhs) {
    return lhs == rhs;
  }

  static inline AxisListRef getEmptyKey() { return AxisListRef(); }

  static inline AxisListRef getTombstoneKey() {
    return AxisListRef(/*isTombstone=*/true);
  }
};

}  // namespace sdy
}  // namespace mlir
