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

#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/register.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/import/passes.h"
#include "shardy/dialect/mpmd/transforms/import/sharding_constraints.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir::mpmd {
namespace {

using ::mlir::func::FuncOp;

MATCHER(IsTransferOp, "") { return isa<TransferOp>(arg.getDefiningOp()); }

void enforceEquishardingConstraints(
    ModuleOp module,
    SmallVector<InputOutputEquishardingConstraint> constraints) {
  PassManager pm(module->getContext());
  pm.enableVerifier();
  pm.addNestedPass<FuncOp>(createEnforceEquishardingPass(
      EnforceEquishardingPassOptions{constraints}));
  SDY_CHECK(succeeded(pm.run(module)));
}

TEST(EnforceEquishardingConstraints, NoConstraints) {
  MLIRContext context;
  loadAllRequiredDialects(&context);

  const std::string kProgram = R"mlir(
  func.func @main(%arg0: !mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>)
      -> (!mpmd.mesh_tensor<"m2", tensor<32x256xf32>, sharding=<@mesh, [{?}, {"y"}]>>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>,
      <"m2": <["y"=4]>>
    >} {
    %0 = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>)
        -> !mpmd.mesh_tensor<"m2", tensor<32x256xf32>, sharding=<@mesh, [{?}, {"y"}]>>
    func.return %0 : !mpmd.mesh_tensor<"m2", tensor<32x256xf32>, sharding=<@mesh, [{?}, {"y"}]>>
  })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp orig_fn = GetMainFunction(*module);
  FunctionType orig_type = orig_fn.getFunctionType();

  enforceEquishardingConstraints(*module, {});

  EXPECT_EQ(orig_type, GetMainFunction(*module).getFunctionType());
}

TEST(EnforceEquishardingConstraints, AlreadySatisfied) {
  MLIRContext context;
  loadAllRequiredDialects(&context);

  const std::string kProgram = R"mlir(
  func.func @main(%arg0: !mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>,
                  %arg1: !mpmd.mesh_tensor<"m2", tensor<32x256xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
      -> (!mpmd.mesh_tensor<"m2", tensor<32x256xf32>, sharding=<@mesh, [{?}, {"y"}]>>,
          !mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>,
      <"m2": <["y"=4]>>
    >} {
    func.return %arg1, %arg0 : !mpmd.mesh_tensor<"m2", tensor<32x256xf32>, sharding=<@mesh, [{?}, {"y"}]>>,
                               !mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>
  })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp orig_fn = GetMainFunction(*module);
  FunctionType orig_type = orig_fn.getFunctionType();

  enforceEquishardingConstraints(*module,
                                 {InputOutputEquishardingConstraint(0, 1),
                                  InputOutputEquishardingConstraint(1, 0)});

  EXPECT_EQ(orig_type, GetMainFunction(*module).getFunctionType());
}

TEST(EnforceEquishardingConstraints, InsertTransfer) {
  MLIRContext context;
  loadAllRequiredDialects(&context);

  const std::string kProgram = R"mlir(
  func.func @main(%arg0: !mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{?}, {"x"}]>>,
                  %arg1: !mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>,
                  %arg2: !mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>)
      -> (!mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{?}, {"x"}]>>,
          !mpmd.mesh_tensor<"m2", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>,
          !mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=4]>>,
      <"m2": <["x"=4]>>
    >} {
    %0 = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{?}, {"x"}]>>)
        -> !mpmd.mesh_tensor<"m2", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>
    %1 = mpmd.fragment<mesh="m2", origin=["f"]> (%0) (%arg3: tensor<32x256xf32>) {
      mpmd.return %arg3 : tensor<32x256xf32>
    } : (!mpmd.mesh_tensor<"m2", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>)
        -> !mpmd.mesh_tensor<"m2", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>

    %2 = mpmd.transfer %arg1 : (!mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>)
        -> !mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{?}, {"x"}]>>
    %3 = mpmd.fragment<mesh="m1", origin=["f"]> (%2) (%arg3: tensor<32x256xf32>) {
      mpmd.return %arg3 : tensor<32x256xf32>
    } : (!mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{?}, {"x"}]>>)
        -> !mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{?}, {"x"}]>>

    func.return %3, %1, %arg2 : !mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{?}, {"x"}]>>,
                                   !mpmd.mesh_tensor<"m2", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>,
                                   !mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>
  })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  enforceEquishardingConstraints(*module,
                                 {InputOutputEquishardingConstraint(0, 1),
                                  InputOutputEquishardingConstraint(1, 0)});

  FuncOp new_fn = GetMainFunction(*module);
  FunctionType new_type = new_fn.getFunctionType();
  EXPECT_EQ(new_type.getInput(0), new_type.getResult(1));
  EXPECT_EQ(new_type.getInput(1), new_type.getResult(0));
  EXPECT_EQ(new_type.getResult(2), new_type.getResult(2));
  Operation* terminator = new_fn.getBody().front().getTerminator();
  EXPECT_THAT(terminator->getOperand(0), IsTransferOp());
  EXPECT_THAT(terminator->getOperand(1), IsTransferOp());
}

TEST(EnforceEquishardingConstraints, FunctionIsNotEntrypoint) {
  MLIRContext context;
  loadAllRequiredDialects(&context);

  const std::string kProgram = R"mlir(
  func.func private @f(%arg0: !mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>,
                  %arg1: !mpmd.mesh_tensor<"m2", tensor<32x256xf32>>)
  -> (
    !mpmd.mesh_tensor<"m2", tensor<32x256xf32>>,
    !mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>
  ) attributes {"topology"=#mpmd.topology<<"m1": <["x"=4]>>, <"m2": <["x"=4]>>>} {
    func.return %arg1, %arg0 : !mpmd.mesh_tensor<"m2", tensor<32x256xf32>>,
                               !mpmd.mesh_tensor<"m1", tensor<32x256xf32>, sharding=<@mesh, [{"x"}, {?}]>>
  })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp orig_fn = dyn_cast_or_null<FuncOp>(module->lookupSymbol("f"));
  SDY_CHECK(orig_fn);
  FunctionType orig_type = orig_fn.getFunctionType();

  // Does not enforce constraints because @f is not an entry-point function.
  enforceEquishardingConstraints(*module,
                                 {InputOutputEquishardingConstraint(0, 0),
                                  InputOutputEquishardingConstraint(1, 1)});

  EXPECT_EQ(
      orig_type,
      dyn_cast_or_null<FuncOp>(module->lookupSymbol("f")).getFunctionType());
}

}  // namespace
}  // namespace mlir::mpmd
