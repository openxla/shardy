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

#include "shardy/dialect/sdy/transforms/propagation/utils.h"

#include <algorithm>
#include <iterator>

#include "llvm/ADT/BitVector.h"  // IWYU pragma: keep
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

SmallVector<int> toSetBitsVector(const BitVector& bitVector) {
  SmallVector<int> result;
  llvm::copy(bitVector.set_bits(), std::back_inserter(result));
  return result;
}

namespace {

// Determines whether the two directions are forward and backward.
// FORWARD, BACKWARD -> true
// BACKWARD, FORWARD -> true
// All other instances returns false.
bool bothForwardAndBackward(PropagationDirection lhs,
                            PropagationDirection rhs) {
  return (lhs == PropagationDirection::FORWARD &&
          rhs == PropagationDirection::BACKWARD) ||
         (lhs == PropagationDirection::BACKWARD &&
          rhs == PropagationDirection::FORWARD);
}

}  // namespace

PropagationDirection unionOfPropagationDirections(PropagationDirection d1,
                                                  PropagationDirection d2) {
  return bothForwardAndBackward(d1, d2) ? PropagationDirection::BOTH
                                        : std::max(d2, d1);
}

PropagationDirection intersectionOfPropagationDirections(
    PropagationDirection d1, PropagationDirection d2) {
  return bothForwardAndBackward(d1, d2) ? PropagationDirection::NONE
                                        : std::min(d2, d1);
}

bool isFullyReplicated(TensorShardingAttr sharding) {
  return !sharding || sharding.isFullyReplicated();
}

bool isEquivalent(TensorShardingAttr sharding,
                  TensorShardingAttr anotherSharding) {
  if (!sharding) {
    return isFullyReplicated(anotherSharding);
  }
  return sharding.isEquivalent(anotherSharding);
}

}  // namespace sdy
}  // namespace mlir
