/* Copyright 2026 The Shardy Authors.

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

#include <cstdint>

#include "llvm/Support/MathExtras.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

bool isCommunicationFreeSliceDim(int64_t dimIdx, stablehlo::SliceOp sliceOp,
                                 TensorShardingAttr sharding, MeshAttr mesh) {
  int64_t shardCount = sharding.getDimShardings()[dimIdx].getShardedSize(mesh);

  if (shardCount <= 1) {
    return true;
  }

  if (sliceOp.getStartIndices()[dimIdx] != 0 ||
      sliceOp.getStrides()[dimIdx] != 1) {
    return false;
  }

  ArrayRef<int64_t> inShape = getTensorShape(sliceOp.getOperand());
  ArrayRef<int64_t> outShape = getTensorShape(sliceOp.getResult());
  int64_t inDimSize = inShape[dimIdx];
  int64_t outDimSize = outShape[dimIdx];

  // Conservatively return false for dynamic shapes if sharded across devices.
  if (inDimSize == ShapedType::kDynamic || outDimSize == ShapedType::kDynamic) {
    return false;
  }

  return llvm::divideCeil(inDimSize, shardCount) ==
         llvm::divideCeil(outDimSize, shardCount);
}

}  // namespace sdy
}  // namespace mlir
