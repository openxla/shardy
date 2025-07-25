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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_COMMON_DISTRIBUTED_FUNCTION_PASS_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_COMMON_DISTRIBUTED_FUNCTION_PASS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::mpmd {

// A base class for an OperationPass that should run only on distributed
// functions (i.e., functions that have a mesh or topology in its attributes).
// This is needed to omit any reduction specific functions for all our
// partitioning and optimization passes.
// TODO(aswietlik): Consider making this a ModuleOp pass with
// GetMainFunction(module_op) instead.
class DistributedFunctionPass : public OperationPass<func::FuncOp> {
 public:
  using OperationPass<func::FuncOp>::OperationPass;

  DistributedFunctionPass() = default;
  DistributedFunctionPass(const DistributedFunctionPass& other) = default;

 protected:
  // The pass will call this method on the distributed function.
  virtual void runOnFunc(func::FuncOp func_op) {}

 private:
  void runOnOperation() final;
};

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_COMMON_DISTRIBUTED_FUNCTION_PASS_H_
