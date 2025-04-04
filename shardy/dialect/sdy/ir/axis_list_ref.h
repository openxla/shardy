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

#ifndef SHARDY_DIALECT_SDY_IR_AXIS_LIST_REF_H_
#define SHARDY_DIALECT_SDY_IR_AXIS_LIST_REF_H_

#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

// This iterator is intended to be used by the `AxisListRef` class to
// enable iteration over the axes held by instances of `AxisListRef`.
// Note that dereferencing this iterator yields a reference to `AxisRefAttr`.
class AxisListRefIterator {
 public:
  // Required for `std::iterator_traits`:
  using value_type = const AxisRefAttr;
  using difference_type = int;
  using pointer = const AxisRefAttr*;
  using reference = const AxisRefAttr&;
  using iterator_category = std::forward_iterator_tag;
  using AxisRefAttrIterator = ArrayRef<AxisRefAttr>::const_iterator;

  explicit AxisListRefIterator(AxisRefAttrIterator axisListCurrent,
                               AxisRefAttrIterator axisListEnd,
                               bool isTailIterated,
                               const AxisRefAttr& tailAxisRef)
      : axisListCurrent(axisListCurrent),
        axisListEnd(axisListEnd),
        isTailIterated(isTailIterated),
        tailAxisRef(tailAxisRef) {}

  AxisListRefIterator& operator++() {
    if (axisListCurrent != axisListEnd) {
      ++axisListCurrent;
    } else {
      isTailIterated = true;
    }
    return *this;
  }

  bool operator==(const AxisListRefIterator& other) const {
    return axisListCurrent == other.axisListCurrent &&
           axisListEnd == other.axisListEnd &&
           isTailIterated == other.isTailIterated &&
           tailAxisRef == other.tailAxisRef;
  }

  bool operator!=(const AxisListRefIterator& other) const {
    return !(*this == other);
  }

  reference operator*() const {
    if (axisListCurrent != axisListEnd) {
      return *axisListCurrent;
    }
    if (isTailIterated) {
      // If this iterator is at the end, its behaviour should be similar to
      // derefercing the end of AxisRefAttrIterator, and as a workaround,
      // it is dereferencing `axisListEnd`.
      return *axisListEnd;
    }
    return tailAxisRef;
  }

 private:
  AxisRefAttrIterator axisListCurrent;
  AxisRefAttrIterator axisListEnd;
  bool isTailIterated;
  const AxisRefAttr& tailAxisRef;
};

// A reference to a list of AxisRefs, that provides API for trimming the list
// such that the last AxisRef can be replaced, while keeping it a reference.
//
// This is useful for getting a prefix of the list of axes, where the last axis
// is also a prefix sub-axis of an axis in the original list.
class AxisListRef {
 public:
  AxisListRef(ArrayRef<AxisRefAttr> axisRefs) {
    if (!axisRefs.empty()) {
      this->axisRefs = axisRefs.drop_back();
      this->tailAxisRef = axisRefs.back();
    }
  }

  AxisListRef() = default;

  // Checks if the axes is empty.
  bool empty() const {
    // If `tailAxisRef` is empty, then `axisRefs` is empty as well. Hence, it is
    // sufficient to check if `tailAxisRef` empty.
    return !tailAxisRef;
  }
  // Clears this AxisListRef.
  void clear();

  int64_t size() const { return empty() ? 0 : axisRefs.size() + 1; }

  bool operator<(const AxisListRef& rhs) const;

  bool operator==(const AxisListRef& rhs) const {
    return axisRefs == rhs.axisRefs && tailAxisRef == rhs.tailAxisRef;
  }

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
  int64_t getExpandedShardingSize(MeshAttr mesh,
                                  const AxisListRef& prefix) const {
    return getShardingSize(mesh) / prefix.getShardingSize(mesh);
  }
  // Truncates `this` to its largest prefix so that it does not overlap with
  // `rhs`. Returns true if `this` has been truncated, and false otherwise,
  // which happens if `this` did not overlap with `rhs` in the first place.
  bool truncateWithoutOverlap(const AxisListRef& rhs);

  using const_iterator = AxisListRefIterator;
  const_iterator begin() const {
    return const_iterator(axisRefs.begin(), axisRefs.end(),
                          /*isTailIterated=*/empty(), tailAxisRef);
  }
  const_iterator end() const {
    return const_iterator(axisRefs.end(), axisRefs.end(),
                          /*isTailIterated=*/true, tailAxisRef);
  }

  friend struct AxisListRefInfo;

 private:
  // Creates an AxisListRef with given `axisRefs` and `tailAxisRef`.
  AxisListRef(ArrayRef<AxisRefAttr> axisRefs, AxisRefAttr tailAxisRef)
      : axisRefs(axisRefs), tailAxisRef(tailAxisRef) {}
  // Returns prefix of input `axisRef` that does not overlap with this axes.
  // TODO(enver): Move this method to utilities.
  // TODO(enver): Instead make this a method of AxisRefAttr, after moving
  // AxesWithTail to a general data structure in Shardy.
  // TODO(enver): Reuse getPrefixOfInputWithout method on
  // shardy/dialect/sdy/transforms/propagation/basic_factor_propagation.cc,
  // instead, after an iterater is added.
  std::optional<AxisRefAttr> getPrefixOfInputWithoutOverlap(
      AxisRefAttr axisRef) const;

  // Trims axes to have the first `newSizeExcludingNewTail` axes and, in case
  // non-empty, `newTailAxisRef` as an additional final axis.
  //
  // As a result, `newSizeExcludingNewTail` is the new size of AxisListRef
  // excluding `newTailAxisRef`. That is, if `newTailAxisRef` is non-empty then
  // the new size of AxisListRef equals to `newSizeExcludingNewTail`+1,
  // otherwise it equals to `newSizeExcludingNewTail`.
  //
  // Assumes that:
  //  1. `this` AxisListRef is non-empty, and
  //  2. `newSizeExcludingNewTail` is strictly smaller than size().
  //  3. Input `newTailAxisRef` is a prefix of the (`newSize`+1)st axis.
  // TODO(enver): It is a bit confusing to have two 'New' in the name of
  // `newSizeExcludingNewTail`.
  void trim(int64_t newSizeExcludingNewTail,
            std::optional<AxisRefAttr> newTailAxisRef);

  // `AxisListRef` is the concatenation of `axisRefs` and `tailAxisRef`. If
  // `tailAxisRef` is empty, then `axisRefs` is empty as well.
  ArrayRef<AxisRefAttr> axisRefs;
  AxisRefAttr tailAxisRef;
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
    return AxisListRef(
        /*axisRefs=*/DenseMapInfo<ArrayRef<AxisRefAttr>>::getTombstoneKey(),
        /*tailAxisRef=*/AxisRefAttr());
  }
};

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_IR_AXIS_LIST_REF_H_
