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

#include "shardy/dialect/mpmd/transforms/export/utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/register.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::mlir::func::FuncOp;
using ::testing::Pair;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

namespace mlir::mpmd {
namespace {

MATCHER(OperationIsAFragmentOp, "") { return isa<FragmentOp>(arg); }

MATCHER(OperationIsAReturnOp, "") {
  return isa<func::ReturnOp>(arg);
}

const char kProgramWithUserMarkedDonation[] = R"mlir(
!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // %arg0 is marked to be donated by the user.
  func.func @main(%arg0: !mesh_1_tensor_4_8_f32 {tf.aliasing_output = 1 : i32}, %arg1: !mesh_1_tensor_4_8_f32 {jax.buffer_donor= true}, %arg2: !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
      "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  // %arg0 and %arg1 are used in this mpmd.fragment op last.
  %0, %1, %2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1, %arg2) (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>) {
    %0 = stablehlo.add %arg3, %arg4: tensor<4x8xf32>
    %1 = stablehlo.abs %0: tensor<4x8xf32>
    %2 = stablehlo.abs %arg5: tensor<4x8xf32>
    mpmd.return %0, %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32,!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32)
  // All three arguments are used in the return operation last.
  func.return %0, %1, %arg2 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}
  )mlir";

TEST(GetAliasedBlockArguments, ShouldReturnCorrectBlockArgsToAlias) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgramWithUserMarkedDonation, &context);
  FuncOp func_op = GetMainFunction(*module);

  DenseSet<BlockArgument> block_args_to_alias =
      GetAliasedBlockArguments(func_op);

  ASSERT_THAT(block_args_to_alias, SizeIs(1));
  EXPECT_THAT(block_args_to_alias.begin()->getArgNumber(), 0);
}

TEST(GetDonatedBlockArguments, ShouldReturnCorrectBlockArgsToDonate) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgramWithUserMarkedDonation, &context);
  FuncOp func_op = GetMainFunction(*module);

  DenseSet<BlockArgument> block_args_to_donate =
      GetDonatedBlockArguments(func_op);

  ASSERT_THAT(block_args_to_donate, SizeIs(1));
  EXPECT_THAT(block_args_to_donate.begin()->getArgNumber(), 1);
}

TEST(OperandsForDeletionMapping,
     ShouldReturnCorrectOperandsForDeletionMapping) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgramWithUserMarkedDonation, &context);
  FuncOp func_op = GetMainFunction(*module);

  DenseMap<Operation*, SmallVector<unsigned int>> operands_last_used_in_op =
      OperandsForDeletionMapping(func_op);

  EXPECT_THAT(operands_last_used_in_op,
              UnorderedElementsAre(
                  Pair(OperationIsAFragmentOp(), UnorderedElementsAre(0, 1)),
                  Pair(OperationIsAReturnOp(), UnorderedElementsAre(0, 1, 2))));
}

}  // namespace
}  // namespace mlir::mpmd
