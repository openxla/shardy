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

#include "shardy/dialect/mpmd/transforms/optimize/utils.h"

#include <string>
#include <string_view>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/register.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/testing_utils.h"
#include <gtest/gtest.h>

using ::mlir::func::FuncOp;

namespace mlir::mpmd {
namespace {

TEST(CanRemat, ShouldReturnTrueIfAllConditionsAreMet) {
  const char kProgram[] = R"mlir(
    !mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
      -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      // Forward fragment, result %0 is used by the backward fragment below.
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      // A fragment that goes between the forward and backward fragments.
      %tmp = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      // Backward fragment.
      %1 = mpmd.fragment<mesh="m1", origin=["f2"(1)]> (%0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

      return %1 : !mesh_1_tensor_4_8_f32
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  Region::OpIterator it = func_op.getOps().begin();
  SDY_CHECK(it != func_op.getOps().end());
  FragmentOp fwd_fragment = cast<FragmentOp>(*it);
  it++;  // Now it should point to the fragment between the forward and backward
         // fragment.
  SDY_CHECK(it != func_op.getOps().end());
  it++;  // Now it should point to the backward fragment.
  SDY_CHECK(it != func_op.getOps().end());
  FragmentOp bwd_fragment = cast<FragmentOp>(*it);

  EXPECT_TRUE(CanRemat(fwd_fragment, bwd_fragment));
}

TEST(CanRemat, ShouldReturnFalseIfNotForward) {
  const char kProgram[] = R"mlir(
    !mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
      -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      // This is not a forward fragment because of the transpose count.
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(1)]> (%arg0)(%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      // A fragment that goes between the not-forward and backward fragments.
      %tmp = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      // Backward fragment.
      %1 = mpmd.fragment<mesh="m1", origin=["f2"(1)]> (%0)(%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

      return %1 : !mesh_1_tensor_4_8_f32
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  Region::OpIterator it = func_op.getOps().begin();
  SDY_CHECK(it != func_op.getOps().end());
  FragmentOp not_fwd_fragment = cast<FragmentOp>(*it);
  it++;  // Now it should point to the fragment between the not-forward and
         // backward fragment.
  SDY_CHECK(it != func_op.getOps().end());
  it++;  // Now it should point to the backward fragment.
  SDY_CHECK(it != func_op.getOps().end());
  FragmentOp bwd_fragment = cast<FragmentOp>(*it);

  EXPECT_FALSE(CanRemat(not_fwd_fragment, bwd_fragment));
}

TEST(CanRemat, ShouldReturnFalseIfNotBackward) {
  const char kProgram[] = R"mlir(
    !mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
      -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      // Forward fragment.
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0)(%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      // A fragment that goes between the forward and non-backward fragments.
      %tmp = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      // This is not a backward fragment because the transpose count is not 1.
      %1 = mpmd.fragment<mesh="m1", origin=["f2"(100)]> (%0)(%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

      return %1 : !mesh_1_tensor_4_8_f32
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  Region::OpIterator it = func_op.getOps().begin();
  SDY_CHECK(it != func_op.getOps().end());
  FragmentOp fwd_fragment = cast<FragmentOp>(*it);
  it++;  // Now it should point to the fragment between the forward and
         // non-backward fragment.
  SDY_CHECK(it != func_op.getOps().end());
  it++;  // Now it should point to the backward fragment.
  SDY_CHECK(it != func_op.getOps().end());
  FragmentOp not_bwd_fragment = cast<FragmentOp>(*it);

  EXPECT_FALSE(CanRemat(fwd_fragment, not_bwd_fragment));
}

TEST(CanRemat,
     ShouldReturnFalseIfCallCounterOfForwardAndBackwardFragmentsDonotMatch) {
  const char kProgram[] = R"mlir(
    !mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
      -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      // Forward fragment with call_counter 1.
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      // A fragment that goes between the forward and backward fragments.
      %tmp = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      // Backward fragment with call_counter 100, which does not match the forward fragment.
      %1 = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%0) {call_counter = 100 : ui32} (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

      return %1 : !mesh_1_tensor_4_8_f32
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  Region::OpIterator it = func_op.getOps().begin();
  SDY_CHECK(it != func_op.getOps().end());
  FragmentOp fwd_fragment = cast<FragmentOp>(*it);
  it++;  // Now it should point to the fragment between the forward and backward
         // fragment.
  SDY_CHECK(it != func_op.getOps().end());
  it++;  // Now it should point to the backward fragment.
  SDY_CHECK(it != func_op.getOps().end());
  FragmentOp bwd_fragment = cast<FragmentOp>(*it);

  EXPECT_FALSE(CanRemat(fwd_fragment, bwd_fragment));
}

TEST(CanRemat, ShouldReturnFalseIfBackwardFragmentsIsImmediatelyAfterForward) {
  const char kProgram[] = R"mlir(
    !mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
      -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      // Forward fragment.
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      // Backward fragment is immediately after the forward fragment.
      %1 = mpmd.fragment<mesh="m1", origin=["f2"(1)]> (%0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

      return %1 : !mesh_1_tensor_4_8_f32
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  Region::OpIterator it = func_op.getOps().begin();
  SDY_CHECK(it != func_op.getOps().end());
  FragmentOp fwd_fragment = cast<FragmentOp>(*it++);
  SDY_CHECK(it != func_op.getOps().end());
  FragmentOp bwd_fragment = cast<FragmentOp>(*it);

  EXPECT_FALSE(CanRemat(fwd_fragment, bwd_fragment));
}

TEST(CanRemat, ShouldReturnFalseIfStagesDontMatch) {
  const char kProgram[] = R"mlir(
    !mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
      -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      // Forward fragment, result %0 is used by the backward fragment below.
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(0)], stage=123> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      // A fragment that goes between the forward and backward fragments.
      %tmp = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%arg0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      // Backward fragment.
      %1 = mpmd.fragment<mesh="m1", origin=["f2"(1)], stage=321> (%0) {call_counter = 1 : ui32} (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

      return %1 : !mesh_1_tensor_4_8_f32
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  Region::OpIterator it = func_op.getOps().begin();
  SDY_CHECK(it != func_op.getOps().end());
  FragmentOp fwd_fragment = cast<FragmentOp>(*it);
  it++;  // Now it should point to the fragment between the forward and backward
         // fragment.
  SDY_CHECK(it != func_op.getOps().end());
  it++;  // Now it should point to the backward fragment.
  SDY_CHECK(it != func_op.getOps().end());
  FragmentOp bwd_fragment = cast<FragmentOp>(*it);

  EXPECT_FALSE(CanRemat(fwd_fragment, bwd_fragment));
}

bool TargetDependsOnSourceOpHelper(std::string_view program) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  SDY_CHECK(module);
  auto main_fn = GetMainFunction(*module);
  Operation* src_op = GetOpWithAttribute<Operation*>(main_fn, "source");
  Operation* tgt_op = GetOpWithAttribute<Operation*>(main_fn, "target");
  return TargetDependsOnSourceOp(src_op, tgt_op);
}

TEST(TargetDependsOnSourceOpTesting, ImmediateDependency) {
  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
      %0 = mpmd.named_computation<"f"> (%arg0) {source} (%arg1: tensor<16x32xf32>) {
        %1 = stablehlo.multiply %arg1, %arg1 : tensor<16x32xf32>
        mpmd.return %1 : tensor<16x32xf32>
      } : (tensor<16x32xf32>) -> tensor<16x32xf32>
      %2 = stablehlo.add %0, %0 {target} : tensor<16x32xf32>
      func.return %2 : tensor<16x32xf32>
    })mlir";
  EXPECT_TRUE(TargetDependsOnSourceOpHelper(program));
}

TEST(TargetDependsOnSourceOpTesting, InternalDependency) {
  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
      %0 = mpmd.named_computation<"f"> (%arg0) (%arg1: tensor<16x32xf32>) {
        %1 = stablehlo.multiply %arg1, %arg1 {source} : tensor<16x32xf32>
        %2 = stablehlo.add %1, %1 {target} : tensor<16x32xf32>
        mpmd.return %2 : tensor<16x32xf32>
      } : (tensor<16x32xf32>) -> tensor<16x32xf32>
      func.return %0 : tensor<16x32xf32>
    })mlir";
  EXPECT_TRUE(TargetDependsOnSourceOpHelper(program));
}

TEST(TargetDependsOnSourceOpTesting, NoDependencyTargetBeforeSource) {
  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
      %0 = mpmd.named_computation<"f"> (%arg0) {target} (%arg1: tensor<16x32xf32>) {
        %1 = stablehlo.multiply %arg1, %arg1 : tensor<16x32xf32>
        mpmd.return %1 : tensor<16x32xf32>
      } : (tensor<16x32xf32>) -> tensor<16x32xf32>
      %2 = stablehlo.add %0, %0 {source} : tensor<16x32xf32>
      func.return %2 : tensor<16x32xf32>
    })mlir";
  EXPECT_FALSE(TargetDependsOnSourceOpHelper(program));
}

TEST(TargetDependsOnSourceOpTesting, NoDependency) {
  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<256xf32>) -> tensor<f32> {
      %0 = stablehlo.add %arg0, %arg0 {source} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %1 = mpmd.named_computation<"f"> (%arg0) {target} (%arg1: tensor<256xf32>) {
        %2 = stablehlo.multiply %arg1, %arg1 : tensor<256xf32>
        mpmd.return %2 : tensor<256xf32>
      } : (tensor<256xf32>) -> tensor<256xf32>
      %3 = stablehlo.dot %1, %0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<f32>
      func.return %3 : tensor<f32>
    })mlir";

  EXPECT_FALSE(TargetDependsOnSourceOpHelper(program));
}

TEST(TargetDependsOnSourceOpTesting, DependencyWithInternalOp) {
  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<256xf32>) -> tensor<f32> {
      %0 = stablehlo.add %arg0, %arg0 {source} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %1 = mpmd.named_computation<"f"> (%0) {target} (%arg1: tensor<256xf32>) {
        %2 = stablehlo.multiply %arg1, %arg1 : tensor<256xf32>
        mpmd.return %2 : tensor<256xf32>
      } : (tensor<256xf32>) -> tensor<256xf32>
      %3 = stablehlo.dot %1, %0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<f32>
      func.return %3 : tensor<f32>
    })mlir";

  EXPECT_TRUE(TargetDependsOnSourceOpHelper(program));
}

TEST(TargetDependsOnSourceOpTesting, RecursiveDependencyWithInternalOps) {
  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<256xf32>) -> tensor<f32> {
      %0 = stablehlo.add %arg0, %arg0 {source} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %1 = mpmd.named_computation<"f"> (%0) {target} (%arg1: tensor<256xf32>) {
        %2 = stablehlo.multiply %arg1, %arg1 : tensor<256xf32>
        mpmd.return %2 : tensor<256xf32>
      } : (tensor<256xf32>) -> tensor<256xf32>
      %3 = mpmd.named_computation<"f"> (%1) {target} (%arg1: tensor<256xf32>) {
        %2 = stablehlo.multiply %arg1, %arg1 : tensor<256xf32>
        mpmd.return %2 : tensor<256xf32>
      } : (tensor<256xf32>) -> tensor<256xf32>
      %4 = stablehlo.dot %3, %3 {target} : (tensor<256xf32>, tensor<256xf32>) -> tensor<f32>
      func.return %4 : tensor<f32>
    })mlir";

  EXPECT_TRUE(TargetDependsOnSourceOpHelper(program));
}

}  // namespace
}  // namespace mlir::mpmd
