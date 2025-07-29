/* Copyright 2025 The MPMD Authors.

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

#include "shardy/dialect/mpmd/transforms/common/distributed_function_pass.h"

#include "shardy/dialect/mpmd/ir/utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::mpmd {

using ::mlir::func::FuncOp;

void DistributedFunctionPass::runOnOperation() {
  FuncOp func_op = getOperation();
  if (IsDistributedFunction(func_op)) {
    runOnFunc(func_op);
  }
}

}  // namespace mlir::mpmd
