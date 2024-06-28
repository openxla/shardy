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

#include "shardy/dialect/sdy/transforms/common/op_properties.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

bool isElementwise(Operation* op) {
  if (op->hasTrait<OpTrait::Elementwise>() ||
      op->hasTrait<hlo::OpTrait::BroadcastingElementwise>()) {
    return true;
  }
  if (auto bitcastConvert = dyn_cast<stablehlo::BitcastConvertOp>(op)) {
    // Shapes are equal iff bit-widths are equal.
    return getTensorShape(bitcastConvert.getOperand()) ==
           getTensorShape(bitcastConvert.getResult());
  }
  return false;
}

}  // namespace sdy
}  // namespace mlir
