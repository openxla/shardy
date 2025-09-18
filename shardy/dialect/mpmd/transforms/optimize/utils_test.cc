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

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
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
#include "shardy/dialect/mpmd/transforms/common/utils.h"
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

// Test fixture for `GetDependencyPath` tests. It parses an MLIR program
// and provides access to the operations marked with "source" and "target"
// attributes, which correspond to the source and target operations in the
// dependency path query.
class GetDependencyPathTesting : public ::testing::Test {
 protected:
  void ParseAndGetOps(std::string_view program) {
    loadAllRequiredDialects(&context_);
    module_ = parseSourceString<ModuleOp>(program, &context_);
    SDY_CHECK(module_);
    FuncOp main_fn = GetMainFunction(*module_);
    src_op_ = GetOpWithAttribute<Operation*>(main_fn, "source");
    tgt_op_ = GetOpWithAttribute<Operation*>(main_fn, "target");
    ASSERT_NE(src_op_, nullptr);
    ASSERT_NE(tgt_op_, nullptr);
  }

  MLIRContext context_;
  OwningOpRef<ModuleOp> module_;
  Operation* src_op_ = nullptr;
  Operation* tgt_op_ = nullptr;
};

TEST_F(GetDependencyPathTesting, ImmediateDependency) {
  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
      %0 = mpmd.named_computation<"f"> (%arg0) {source} (%arg1: tensor<16x32xf32>) {
        %1 = stablehlo.multiply %arg1, %arg1 : tensor<16x32xf32>
        mpmd.return %1 : tensor<16x32xf32>
      } : (tensor<16x32xf32>) -> tensor<16x32xf32>
      %2 = stablehlo.add %0, %0 {target} : tensor<16x32xf32>
      func.return %2 : tensor<16x32xf32>
    })mlir";
  ParseAndGetOps(program);
  std::optional<SmallVector<Operation*>> path =
      GetDependencyPath(src_op_, tgt_op_);
  ASSERT_TRUE(path.has_value());
  ASSERT_EQ(path->size(), 2);
  EXPECT_EQ((*path)[0], src_op_);
  EXPECT_EQ((*path)[1], tgt_op_);
}

TEST_F(GetDependencyPathTesting, InternalDependency) {
  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
      %0 = mpmd.named_computation<"f"> (%arg0) (%arg1: tensor<16x32xf32>) {
        %1 = stablehlo.multiply %arg1, %arg1 {source} : tensor<16x32xf32>
        %2 = stablehlo.add %1, %1 {target} : tensor<16x32xf32>
        mpmd.return %2 : tensor<16x32xf32>
      } : (tensor<16x32xf32>) -> tensor<16x32xf32>
      func.return %0 : tensor<16x32xf32>
    })mlir";
  ParseAndGetOps(program);
  std::optional<SmallVector<Operation*>> path =
      GetDependencyPath(src_op_, tgt_op_);
  ASSERT_TRUE(path.has_value());
  ASSERT_EQ(path->size(), 2);
  EXPECT_EQ((*path)[0], src_op_);
  EXPECT_EQ((*path)[1], tgt_op_);
}

TEST_F(GetDependencyPathTesting, NoDependencyTargetBeforeSource) {
  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
      %0 = mpmd.named_computation<"f"> (%arg0) {target} (%arg1: tensor<16x32xf32>) {
        %1 = stablehlo.multiply %arg1, %arg1 : tensor<16x32xf32>
        mpmd.return %1 : tensor<16x32xf32>
      } : (tensor<16x32xf32>) -> tensor<16x32xf32>
      %2 = stablehlo.add %0, %0 {source} : tensor<16x32xf32>
      func.return %2 : tensor<16x32xf32>
    })mlir";
  ParseAndGetOps(program);
  EXPECT_FALSE(GetDependencyPath(src_op_, tgt_op_).has_value());
}

TEST_F(GetDependencyPathTesting, NoDependency) {
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
  ParseAndGetOps(program);
  EXPECT_FALSE(GetDependencyPath(src_op_, tgt_op_).has_value());
}

TEST_F(GetDependencyPathTesting, DependencyWithInternalOp) {
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
  ParseAndGetOps(program);
  std::optional<SmallVector<Operation*>> path =
      GetDependencyPath(src_op_, tgt_op_);
  ASSERT_TRUE(path.has_value());
  ASSERT_EQ(path->size(), 2);
  EXPECT_EQ((*path)[0], src_op_);
  EXPECT_EQ((*path)[1], tgt_op_);
}

TEST_F(GetDependencyPathTesting, RecursiveDependencyWithInternalOps) {
  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<256xf32>) -> tensor<f32> {
      %0 = stablehlo.add %arg0, %arg0 {source} : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
      %1 = mpmd.named_computation<"f"> (%0) (%arg1: tensor<256xf32>) {
        %2 = stablehlo.multiply %arg1, %arg1 : tensor<256xf32>
        mpmd.return %2 : tensor<256xf32>
      } : (tensor<256xf32>) -> tensor<256xf32>
      %3 = mpmd.named_computation<"g"> (%1) (%arg1: tensor<256xf32>) {
        %2 = stablehlo.multiply %arg1, %arg1 : tensor<256xf32>
        mpmd.return %2 : tensor<256xf32>
      } : (tensor<256xf32>) -> tensor<256xf32>
      %4 = stablehlo.dot %3, %3 {target} : (tensor<256xf32>, tensor<256xf32>) -> tensor<f32>
      func.return %4 : tensor<f32>
    })mlir";
  ParseAndGetOps(program);
  std::optional<SmallVector<Operation*>> path =
      GetDependencyPath(src_op_, tgt_op_);
  ASSERT_TRUE(path.has_value());
  ASSERT_EQ(path->size(), 4);
  EXPECT_EQ((*path)[0], src_op_);
  EXPECT_EQ(mlir::cast<NamedComputationOp>((*path)[1]).getName(), "f");
  EXPECT_EQ(mlir::cast<NamedComputationOp>((*path)[2]).getName(), "g");
  EXPECT_EQ((*path)[3], tgt_op_);
}

TEST(AddControlDependency, FirstDependency) {
  const char kProgram[] = R"mlir(
    !mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_tensor) -> !mesh_tensor
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0)
           (%arg1: tensor<4x8xf32>) {
        mpmd.return %arg1 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %1 = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%arg0)
           (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      return %1 : !mesh_tensor
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  FragmentOp frag0 = cast<FragmentOp>(*it++);
  FragmentOp frag1 = cast<FragmentOp>(*it);

  int64_t original_operand_count = frag1.getNumOperands();

  AddControlDependency(frag0, frag1);

  // Verify attribute was set
  ASSERT_TRUE(frag1->hasAttr(kControlOperandStartIdxAttrName));
  auto attr =
      frag1->getAttrOfType<IntegerAttr>(kControlOperandStartIdxAttrName);
  EXPECT_EQ(attr.getInt(), original_operand_count);

  // Verify operand was added
  EXPECT_EQ(frag1.getNumOperands(), original_operand_count + 1);
  EXPECT_EQ(frag1->getOperand(original_operand_count), frag0->getResult(0));
}

TEST(AddControlDependency, SecondDependency) {
  const char kProgram[] = R"mlir(
    !mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_tensor) -> !mesh_tensor
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0)
           (%arg1: tensor<4x8xf32>) {
        mpmd.return %arg1 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %1 = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%arg0)
           (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %2 = mpmd.fragment<mesh="m1", origin=["f3"(0)]> (%arg0)
           (%arg3: tensor<4x8xf32>) {
        mpmd.return %arg3 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      return %2 : !mesh_tensor
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  FragmentOp frag0 = mlir::cast<FragmentOp>(*it++);
  FragmentOp frag1 = mlir::cast<FragmentOp>(*it++);
  FragmentOp frag2 = mlir::cast<FragmentOp>(*it);

  int64_t original_operand_count = frag2.getNumOperands();

  // Add first dependency
  AddControlDependency(frag0, frag2);
  auto attr_after_first =
      frag2->getAttrOfType<IntegerAttr>(kControlOperandStartIdxAttrName);
  int64_t control_start_index = attr_after_first.getInt();

  // Add second dependency
  AddControlDependency(frag1, frag2);

  // Verify attribute value didn't change
  auto attr_after_second =
      frag2->getAttrOfType<IntegerAttr>(kControlOperandStartIdxAttrName);
  EXPECT_EQ(attr_after_second.getInt(), control_start_index);

  // Verify both operands were added
  EXPECT_EQ(frag2.getNumOperands(), original_operand_count + 2);
  EXPECT_EQ(frag2->getOperand(original_operand_count), frag0->getResult(0));
  EXPECT_EQ(frag2->getOperand(original_operand_count + 1), frag1->getResult(0));
}

TEST(AddControlDependency, WithExistingDataOperands) {
  const char kProgram[] = R"mlir(
    !mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_tensor, %arg1: !mesh_tensor) -> !mesh_tensor
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0)
           (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %1 = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%arg0, %arg1)
           (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
        mpmd.return %arg3 : tensor<4x8xf32>
      } : (!mesh_tensor, !mesh_tensor) -> !mesh_tensor

      return %1 : !mesh_tensor
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  FragmentOp frag0 = mlir::cast<FragmentOp>(*it++);
  FragmentOp frag1 = mlir::cast<FragmentOp>(*it);

  // frag1 has 2 data operands
  ASSERT_EQ(frag1.getNumOperands(), 2);
  Value data_operand0 = frag1->getOperand(0);
  Value data_operand1 = frag1->getOperand(1);

  AddControlDependency(frag0, frag1);

  // Verify attribute set to 2 (original operand count)
  auto attr =
      frag1->getAttrOfType<IntegerAttr>(kControlOperandStartIdxAttrName);
  EXPECT_EQ(attr.getInt(), 2);

  // Verify total operands = 3 (2 data + 1 control)
  EXPECT_EQ(frag1.getNumOperands(), 3);

  // Verify data operands unchanged
  EXPECT_EQ(frag1->getOperand(0), data_operand0);
  EXPECT_EQ(frag1->getOperand(1), data_operand1);

  // Verify control operand at end
  EXPECT_EQ(frag1->getOperand(2), frag0->getResult(0));
}

TEST(RemoveAllControlDependencies, SingleFragment) {
  const char kProgram[] = R"mlir(
    !mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_tensor) -> !mesh_tensor
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0)
           (%arg1: tensor<4x8xf32>) {
        mpmd.return %arg1 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %1 = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%arg0)
           (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %2 = mpmd.fragment<mesh="m1", origin=["f3"(0)]> (%arg0)
           (%arg3: tensor<4x8xf32>) {
        mpmd.return %arg3 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      return %2 : !mesh_tensor
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  FragmentOp frag0 = mlir::cast<FragmentOp>(*it++);
  FragmentOp frag1 = mlir::cast<FragmentOp>(*it++);
  FragmentOp frag2 = mlir::cast<FragmentOp>(*it);

  // Add 2 control dependencies to frag2
  AddControlDependency(frag0, frag2);
  AddControlDependency(frag1, frag2);

  int64_t original_operand_count = 1;    // frag2 originally had 1 operand
  ASSERT_EQ(frag2.getNumOperands(), 3);  // Now has 1 data + 2 control

  RemoveAllControlDependencies(func_op);

  // Verify operand count restored
  EXPECT_EQ(frag2.getNumOperands(), original_operand_count);

  // Verify attribute removed
  EXPECT_FALSE(frag2->hasAttr(kControlOperandStartIdxAttrName));
}

TEST(RemoveAllControlDependencies, MultipleFragments) {
  const char kProgram[] = R"mlir(
    !mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_tensor) -> !mesh_tensor
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0)
           (%arg1: tensor<4x8xf32>) {
        mpmd.return %arg1 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %1 = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%arg0)
           (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %2 = mpmd.fragment<mesh="m1", origin=["f3"(0)]> (%arg0)
           (%arg3: tensor<4x8xf32>) {
        mpmd.return %arg3 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      return %2 : !mesh_tensor
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  FragmentOp frag0 = mlir::cast<FragmentOp>(*it++);
  FragmentOp frag1 = mlir::cast<FragmentOp>(*it++);
  FragmentOp frag2 = mlir::cast<FragmentOp>(*it);

  // frag1: add 2 control dependencies
  AddControlDependency(frag0, frag1);
  AddControlDependency(frag0, frag1);

  // frag2: add 1 control dependency
  AddControlDependency(frag1, frag2);

  // frag0: no control dependencies

  ASSERT_EQ(frag0.getNumOperands(), 1);
  ASSERT_EQ(frag1.getNumOperands(), 3);  // 1 data + 2 control
  ASSERT_EQ(frag2.getNumOperands(), 2);  // 1 data + 1 control

  RemoveAllControlDependencies(func_op);

  // Verify all fragments restored
  EXPECT_EQ(frag0.getNumOperands(), 1);
  EXPECT_EQ(frag1.getNumOperands(), 1);
  EXPECT_EQ(frag2.getNumOperands(), 1);

  // Verify all attributes removed
  EXPECT_FALSE(frag0->hasAttr(kControlOperandStartIdxAttrName));
  EXPECT_FALSE(frag1->hasAttr(kControlOperandStartIdxAttrName));
  EXPECT_FALSE(frag2->hasAttr(kControlOperandStartIdxAttrName));
}

TEST(RemoveAllControlDependencies, NoControlDependencies) {
  const char kProgram[] = R"mlir(
    !mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_tensor) -> !mesh_tensor
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0)
           (%arg1: tensor<4x8xf32>) {
        mpmd.return %arg1 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %1 = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%arg0)
           (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      return %1 : !mesh_tensor
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  FragmentOp frag0 = mlir::cast<FragmentOp>(*it++);
  FragmentOp frag1 = mlir::cast<FragmentOp>(*it);

  // Record original state (no control dependencies)
  int64_t original_frag0_operands = frag0.getNumOperands();
  int64_t original_frag1_operands = frag1.getNumOperands();

  // Call RemoveAllControlDependencies on function with no control deps
  RemoveAllControlDependencies(func_op);

  // Verify no changes
  EXPECT_EQ(frag0.getNumOperands(), original_frag0_operands);
  EXPECT_EQ(frag1.getNumOperands(), original_frag1_operands);
  EXPECT_FALSE(frag0->hasAttr(kControlOperandStartIdxAttrName));
  EXPECT_FALSE(frag1->hasAttr(kControlOperandStartIdxAttrName));
}

}  // namespace
}  // namespace mlir::mpmd
