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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_UTILS_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_UTILS_H_

#include "llvm/ADT/SmallVector.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

namespace mlir {
namespace sdy {

// Returns the greatest common prefix of given two arrays axis refs.
SmallVector<AxisRefAttr> getGreatestCommonPrefix(ArrayRef<AxisRefAttr> first,
                                                 ArrayRef<AxisRefAttr> second);

// Returns an array of axis-ref arrays where each axis-ref array defines a
// factor sharding, for the corresponding factor, as the greatest common
// prefix of factor shardings across all operands and results.
// TODO(enver): The number of factors can instead be an api of
// ShardingProjection, and this method can get it from the projection.
AxisRefsList getGreatestCommonPrefixAxes(const ShardingProjection& projection);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_EXPORT_UTILS_H_
