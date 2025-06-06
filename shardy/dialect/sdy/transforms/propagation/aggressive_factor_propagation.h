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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_AGGRESSIVE_FACTOR_PROPAGATION_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_AGGRESSIVE_FACTOR_PROPAGATION_H_

#include <cstdint>

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/propagation/basic_factor_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/factor_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

namespace mlir {
namespace sdy {

// An aggressive strategy of propagating sharding axes along factors. There are
// two main differences from `BasicFactorPropagation`.
//
// `BasicFactorPropagation` propagates the same sharding axes to all tensors
// along a factor. This strategy can propagate different sharding axes to
// different tensors along the same factor. For example, Tensors T0, T1, T2
// contain Factor F0. T0/F0 is already sharded along ["a", "b"], and "b" is
// already used by T2 ("b" can be explicitly replicated/unreduced, or it is used
// to shard another factor). `BasicFactorPropagation` propagates ["a"] to both
// T1/F0 and T2/F0, while this strategy propagates ["a", "b"] to T1/F0 and ["a"]
// to T2/F0, respectively. If T2/F0 is closed, `BasicFactorPropagation`
// propagates nothing, while this strategy propagates nothing to T2/F0 and still
// propagates ["a", "b"] to T1/F0.
//
// `BasicFactorPropagation` is conservative in terms of conflicts across
// factors. The overlapped axis between factors cannot be propagated. This
// strategy is more aggressive by allowing the overlapped axis being propagated
// along different factors if there is no overlapped axis in the current
// shardings.
//
// To resolve conflicts across factors, when there are multiple choices (that
// cannot co-exist), we prefer the factor with the larger source tensor (the
// tensor from which the factor sharding is propagated), as it's normally
// beneficial to reshard the smaller tensor. If two factors have the same source
// tensor size, we sort based on the source tensor index and finally factor
// index.
//
// Let us take C = dot(A, B) as an example. F0 is the factor corresponding to a
// non-contracting dimension of A. F1 corresponds to a non-contracting dimension
// of B. F2 corresponds to a contracting dimension. '-' means that the tensor
// does not contain the factor.
//
//     F0    F1    F2
// A  "a"    -
// B   -
// C        "a"    -
// Case 1. Conflict with a single choice. `BasicFactorPropagation` propagates
// nothing, while this strategy propagates "a" to B/F1.
//
//     F0    F1    F2
// A  "a"    -
// B   -    "a"
// C               -
// Case 2. Conflict with multiple choices. `BasicFactorPropagation` propagates
// nothing, while this strategy propagates "a" to C/F1, since F1 is preferred
// over F0 (tensor B is larger than A).
class AggressiveFactorPropagation : public BasicFactorPropagation {
 public:
  UpdateTensorShardings propagateFactorShardings(
      ShardingProjection& projection,
      PropagationDirectionAlongFactor directionAlongFactor,
      ArrayRef<int64_t> factorSizes, MeshAttr mesh,
      bool conservativePropagation, Operation* op) const override;

 private:
  // Returns the axes to propagate to an individual factor in the given
  // `tensorFactorShardings` of a tensor.
  SmallVector<AxisRefAttr> getPropagatedFactorSharding(
      int64_t factorIndex, const TensorFactorShardings& tensorFactorShardings,
      const FactorIndexToSharding& factorIndexToSharding,
      AxesPerFactorRef axesPerFactor, MeshAttr mesh,
      bool conservativePropagation, ArrayRef<int64_t> factorSizes) const;
};

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_AGGRESSIVE_FACTOR_PROPAGATION_H_
