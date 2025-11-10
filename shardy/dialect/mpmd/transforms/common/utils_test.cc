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

#include "shardy/dialect/mpmd/transforms/common/utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/register.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/optimize/utils.h"
#include <gtest/gtest.h>

namespace mlir::mpmd {
namespace {

using ::mlir::func::FuncOp;

TEST(MergeRegionOps, PreservesControlOperands) {
  const char kProgram[] = R"mlir(
    !mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_tensor) -> !mesh_tensor
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f0"(0)]> (%arg0)
           (%arg1: tensor<4x8xf32>) {
        mpmd.return %arg1 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %1 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0)
           (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %2 = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%arg0)
           (%arg3: tensor<4x8xf32>) {
        mpmd.return %arg3 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %3 = mpmd.fragment<mesh="m1", origin=["f3"(0)]> (%arg0)
           (%arg4: tensor<4x8xf32>) {
        mpmd.return %arg4 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      return %3 : !mesh_tensor
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  FragmentOp f0 = mlir::cast<FragmentOp>(*it++);
  FragmentOp f1 = mlir::cast<FragmentOp>(*it++);
  FragmentOp f2 = mlir::cast<FragmentOp>(*it++);
  FragmentOp f3 = mlir::cast<FragmentOp>(*it);

  AddControlDependency(f0, f1);
  AddControlDependency(f2, f3);

  ASSERT_EQ(f1.getNumOperands(), 2);
  ASSERT_EQ(f3.getNumOperands(), 2);

  IRRewriter rewriter(&context);
  rewriter.setInsertionPoint(f3);

  FragmentOp merged_fragment = MergeRegionOps(
      f1, f3, rewriter,
      /*num_static_args=*/0,
      /*replace_producer_use_in_consumer_block=*/
      [](OpOperand&, Value) {
        SDY_CHECK(false) << "Fragment ops shouldn't have free variables";
      },
      GetFragmentOriginUnion(f1, f3, rewriter), f1.getMeshNameAttr(),
      /*stage_id=*/f1.getStageIdAttr());

  ASSERT_TRUE(merged_fragment->hasAttr(kControlOperandStartIdxAttrName));
  int control_start_idx =
      merged_fragment
          ->getAttrOfType<IntegerAttr>(kControlOperandStartIdxAttrName)
          .getInt();

  EXPECT_EQ(merged_fragment.getNumOperands(), control_start_idx + 2);
  EXPECT_EQ(merged_fragment->getOperand(control_start_idx), f0->getResult(0));
  EXPECT_EQ(merged_fragment->getOperand(control_start_idx + 1),
            f2->getResult(0));
}

}  // namespace
}  // namespace mlir::mpmd
