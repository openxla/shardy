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

#include "shardy/dialect/mpmd/transforms/import/mesh_inference_utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/transforms/import/meshes_with_origins.h"
#include "stablehlo/dialect/StablehloOps.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::mlir::func::FuncOp;
using ::testing::UnorderedElementsAre;

namespace mlir::mpmd {
namespace {

TEST(UpdateTransitiveUses, UpdatesForBlockArgs) {
  constexpr StringRef kProgram = R"mlir(
  func.func @main(%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>){
    %1 = stablehlo.add %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} : tensor<4x8xf32>
    %2 = stablehlo.add %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : tensor<4x8xf32>
    func.return %2 : tensor<4x8xf32>
  })mlir";

  MLIRContext context;
  context.loadDialect<func::FuncDialect, stablehlo::StablehloDialect,
                      MpmdDialect>();
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);

  FuncOp main_func = dyn_cast<FuncOp>(module->lookupSymbol("main"));
  BlockArgument block_arg = main_func.front().getArgument(0);
  MeshesWithOrigins used_in_meshes;

  UpdateTransitiveUses(block_arg, used_in_meshes);

  EXPECT_THAT(used_in_meshes.MeshNamesOrEmpty(),
              UnorderedElementsAre("m1", "m2"));
}

TEST(UpdateTransitiveUses, UpdatesForOperation) {
  constexpr StringRef kProgram = R"mlir(
  func.func @main(%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>){
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4x8xf32>
    %1 = stablehlo.add %0, %0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : tensor<4x8xf32>
    %2 = stablehlo.add %0, %0 {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} : tensor<4x8xf32>
    %3 = stablehlo.add %0, %0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m3">} : tensor<4x8xf32>
    func.return %3 : tensor<4x8xf32>
  })mlir";

  MLIRContext context;
  context.loadDialect<func::FuncDialect, stablehlo::StablehloDialect,
                      MpmdDialect>();
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);

  FuncOp main_func = dyn_cast<FuncOp>(module->lookupSymbol("main"));
  Operation* op = &*main_func.getOps().begin();
  MeshesWithOrigins used_in_meshes;

  UpdateTransitiveUses(op->getResult(0), used_in_meshes);

  EXPECT_THAT(used_in_meshes.MeshNamesOrEmpty(),
              UnorderedElementsAre("m1", "m2", "m3"));
}

TEST(UpdateTransitiveUses, UpdatesForValue) {
  constexpr StringRef kProgram = R"mlir(
  func.func @main(%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>){
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4x8xf32>
    %1 = stablehlo.add %0, %0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1">} : tensor<4x8xf32>
    %2 = stablehlo.add %0, %0 {mpmd.use_set = #mpmd.meshes_with_origins<"m2">} : tensor<4x8xf32>
    %3 = stablehlo.add %0, %0 {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m3">} : tensor<4x8xf32>
    func.return %3 : tensor<4x8xf32>
  })mlir";

  MLIRContext context;
  context.loadDialect<func::FuncDialect, stablehlo::StablehloDialect,
                      MpmdDialect>();
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);

  FuncOp main_func = dyn_cast<FuncOp>(module->lookupSymbol("main"));
  Value v0 = main_func.getOps().begin()->getResult(0);
  MeshesWithOrigins used_in_meshes;

  UpdateTransitiveUses(v0, used_in_meshes);

  EXPECT_THAT(used_in_meshes.MeshNamesOrEmpty(),
              UnorderedElementsAre("m1", "m2", "m3"));
}

TEST(UpdateTransitiveUses, UpdatesWhenUsedInRegionOps) {
  constexpr StringRef kProgram = R"mlir(
  func.func @main(%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>){
    %0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
    %1 = stablehlo.constant dense<1> : tensor<i1>
    %region_op = stablehlo.while(%iterArg_0 = %1) : tensor<i1>
    attributes {mpmd.use_set = #mpmd.meshes_with_origins<"m1">}
    cond {
      "stablehlo.return"(%iterArg_0) : (tensor<i1>) -> ()
    } do {
      %8 = stablehlo.add %arg0, %0 : tensor<4x8xf32>
      "stablehlo.return"(%iterArg_0) : (tensor<i1>) -> ()
    }
    func.return %0 : tensor<4x8xf32>
  })mlir";

  MLIRContext context;
  context.loadDialect<func::FuncDialect, stablehlo::StablehloDialect,
                      MpmdDialect>();
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);

  FuncOp main_func = dyn_cast<FuncOp>(module->lookupSymbol("main"));

  BlockArgument block_arg = main_func.front().getArgument(0);
  MeshesWithOrigins block_arg_meshes;
  UpdateTransitiveUses(block_arg, block_arg_meshes);
  EXPECT_THAT(block_arg_meshes.MeshNamesOrEmpty(), UnorderedElementsAre("m1"));

  Operation* op = &*main_func.getOps().begin();
  MeshesWithOrigins op_meshes;
  UpdateTransitiveUses(op, op_meshes);
  EXPECT_THAT(op_meshes.MeshNamesOrEmpty(), UnorderedElementsAre("m1"));

  Value v0 = op->getResult(0);
  MeshesWithOrigins v0_meshes;
  UpdateTransitiveUses(v0, v0_meshes);
  EXPECT_THAT(v0_meshes.MeshNamesOrEmpty(), UnorderedElementsAre("m1"));
}

TEST(UpdateTransitiveUses, UpdatesWhenUsedInCallOps) {
  constexpr StringRef kProgram = R"mlir(
  func.func @main(%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>){
    %0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
    %call:2 = mpmd.call @call(%arg0, %0) :
      (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
    func.return %0 : tensor<4x8xf32>
  }

  func.func @call(
    %arg0: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1">},
    %arg1: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m2">}) ->
    (tensor<4x8xf32>){
    %1 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    %2 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    func.return %2 : tensor<4x8xf32>
  })mlir";

  MLIRContext context;
  context.loadDialect<func::FuncDialect, stablehlo::StablehloDialect,
                      MpmdDialect>();
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);

  FuncOp main_func = dyn_cast<FuncOp>(module->lookupSymbol("main"));

  BlockArgument block_arg = main_func.front().getArgument(0);
  MeshesWithOrigins used_in_meshes;
  UpdateTransitiveUses(block_arg, used_in_meshes);
  EXPECT_THAT(used_in_meshes.MeshNamesOrEmpty(), UnorderedElementsAre("m1"));

  Operation* op = &*main_func.getOps().begin();
  MeshesWithOrigins op_meshes;
  UpdateTransitiveUses(op, op_meshes);
  EXPECT_THAT(op_meshes.MeshNamesOrEmpty(), UnorderedElementsAre("m2"));

  Value v0 = op->getResult(0);
  MeshesWithOrigins v0_meshes;
  UpdateTransitiveUses(v0, v0_meshes);
  EXPECT_THAT(v0_meshes.MeshNamesOrEmpty(), UnorderedElementsAre("m2"));
}

}  // namespace

}  // namespace mlir::mpmd
