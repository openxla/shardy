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
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

namespace mlir {
namespace sdy {

// An aggressive strategy of propagating sharding axes along factors.
//
// This strategy is the same as `BasicFactorPropagation` on the conflicts within
// a factor. They are different on the conflicts across factors.
//
// `BasicFactorPropagation` considers the conflicts across factors with a strict
// criterion. The result cannot overlap with the sharded axes or overflow axes
// related to all other factors. This aggressive strategy ignores "fake
// conflicts", which are propagation choices that can co-exist. This aggressive
// strategy ensures that the resultant axes can be propagated to all tensors
// containing the factor. Several examples of fake conflicts:
//
// 1. An axis is in factors Fi and Fj. If it is infeasible to propagate that
// axis along factor Fi, we may propagate that axis along factor Fj if all the
// destination tensors have not used that axis.
//
// 2. Two factors Fi and Fj do not co-exist in any tensor, so they never
// interfere with each other. If Fi and Fj are sharded along the same axis, we
// can propagate that axis along both factors.
//
// Although fake conflicts can co-exist without inference, we may still need to
// all-gather some tensors.
class AggressiveFactorPropagation : public BasicFactorPropagation {
 public:
  AxesPerFactor getCompatibleMajorShardingAxesForAllFactors(
      const ShardingProjection& projection, PropagationDirection direction,
      ArrayRef<int64_t> factorSizes, MeshAttr mesh, Operation* op,
      bool conservativePropagation) const override;

  UpdateTensorShardings propagateFactorShardings(
      ShardingProjection& projection, PropagationDirection direction,
      ArrayRef<int64_t> factorSizes, MeshAttr mesh, Operation* op,
      bool conservativePropagation) const override;
};

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_AGGRESSIVE_FACTOR_PROPAGATION_H_
