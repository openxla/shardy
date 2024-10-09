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

#include "shardy/dialect/sdy/transforms/export/utils.h"

#include "llvm/ADT/STLExtras.h"    // IWYU pragma: keep
#include "llvm/ADT/SmallVector.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "third_party/openxla/shardy/src/shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

namespace mlir {
namespace sdy {

SmallVector<AxisRefAttr> getGreatestCommonPrefix(ArrayRef<AxisRefAttr> first,
                                                 ArrayRef<AxisRefAttr> second) {
  SmallVector<AxisRefAttr> result;
  for (auto [firstAxisRef, secondAxisRef] : llvm::zip(first, second)) {
    if (firstAxisRef == secondAxisRef) {
      result.push_back(firstAxisRef);
      continue;
    }
    if (auto prefix = firstAxisRef.getGreatestCommonPrefix(secondAxisRef);
        prefix) {
      result.push_back(*prefix);
    }
    break;
  }
  return result;
}

SmallVector<SmallVector<AxisRefAttr>> getGreatestCommonPrefixAxes(
    const ShardingProjection& projection) {
  FactorIndexToSharding factorIndexToCommonSharding;
  int numFactors = 0;
  for (const TensorFactorShardings& tensorFactorSharding :
       llvm::concat<const TensorFactorShardings>(projection.getOperands(),
                                                 projection.getResults())) {
    // Detects conflicts within the same factor.
    for (const auto& [factorIndex, factorSharding] :
         tensorFactorSharding.factorIndexToSharding) {
      if (factorIndex + 1 > numFactors) {
        numFactors = factorIndex + 1;
      }
      auto commonFactorShardingIt =
          factorIndexToCommonSharding.find(factorIndex);
      if (commonFactorShardingIt == factorIndexToCommonSharding.end()) {
        factorIndexToCommonSharding[factorIndex] = factorSharding;
        continue;
      }
      factorIndexToCommonSharding[factorIndex] = {
          .axisRefs =
              getGreatestCommonPrefix(commonFactorShardingIt->second.axisRefs,
                                      factorSharding.axisRefs)};
    }
  }
  SmallVector<SmallVector<AxisRefAttr>> factorAxisRefs(numFactors);
  for (const auto& [factorIndex, factorSharding] :
       factorIndexToCommonSharding) {
    factorAxisRefs[factorIndex] = llvm::to_vector(factorSharding.axisRefs);
  }
  return factorAxisRefs;
}

}  // namespace sdy
}  // namespace mlir
