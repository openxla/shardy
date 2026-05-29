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

// ---------------------------------------------------------------------------
// A reusable two-mesh program for fragment query tests.
// Four fragments total: f0(m1), f1(m2), f2(m1), f3(m1).
// ---------------------------------------------------------------------------
constexpr char kTwoMeshProgram[] = R"mlir(
  !m1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  !m2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  func.func @main(%arg0: !m1_tensor, %arg1: !m2_tensor)
      -> (!m1_tensor, !m2_tensor)
    attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=4]>>>} {
    %0 = mpmd.fragment<mesh="m1", origin=["f0"(0)]> (%arg0)
         (%a0: tensor<4x8xf32>) {
      mpmd.return %a0 : tensor<4x8xf32>
    } : (!m1_tensor) -> !m1_tensor

    %1 = mpmd.fragment<mesh="m2", origin=["f1"(0)]> (%arg1)
         (%a1: tensor<4x8xf32>) {
      mpmd.return %a1 : tensor<4x8xf32>
    } : (!m2_tensor) -> !m2_tensor

    %2 = mpmd.fragment<mesh="m1", origin=["f2"(0)]> (%0)
         (%a2: tensor<4x8xf32>) {
      mpmd.return %a2 : tensor<4x8xf32>
    } : (!m1_tensor) -> !m1_tensor

    %3 = mpmd.fragment<mesh="m1", origin=["f3"(0)]> (%2)
         (%a3: tensor<4x8xf32>) {
      mpmd.return %a3 : tensor<4x8xf32>
    } : (!m1_tensor) -> !m1_tensor

    return %3, %1 : !m1_tensor, !m2_tensor
  }
)mlir";

// Helper: parse the two-mesh program and return (func, {f0, f1, f2, f3}).
struct ParsedTwoMeshProgram {
  OwningOpRef<ModuleOp> module;
  FuncOp func_op;
  FragmentOp f0;  // m1
  FragmentOp f1;  // m2
  FragmentOp f2;  // m1
  FragmentOp f3;  // m1
};

ParsedTwoMeshProgram ParseTwoMeshProgram(MLIRContext& context) {
  ParsedTwoMeshProgram p;
  p.module = parseSourceString<ModuleOp>(kTwoMeshProgram, &context);
  p.func_op = GetMainFunction(*p.module);
  auto it = p.func_op.getOps().begin();
  p.f0 = mlir::cast<FragmentOp>(*it++);
  p.f1 = mlir::cast<FragmentOp>(*it++);
  p.f2 = mlir::cast<FragmentOp>(*it++);
  p.f3 = mlir::cast<FragmentOp>(*it);
  return p;
}

// ===== FindLastFragmentOnMesh tests =====

TEST(FindLastFragmentOnMesh, ReturnsLastOnMatchingMesh) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  Block* block = &p.func_op.getBody().front();
  // m1 fragments are f0, f2, f3.  The last one should be f3.
  FragmentOp last = FindLastFragmentOnMesh(block, "m1");
  ASSERT_NE(last, nullptr);
  EXPECT_EQ(last, p.f3);
}

TEST(FindLastFragmentOnMesh, ReturnsNullptrForMissingMesh) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  Block* block = &p.func_op.getBody().front();
  FragmentOp result = FindLastFragmentOnMesh(block, "nonexistent");
  EXPECT_EQ(result, nullptr);
}

TEST(FindLastFragmentOnMesh, RespectsExcludeList) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  Block* block = &p.func_op.getBody().front();
  // Exclude f3 — the last m1 fragment should now be f2.
  FragmentOp last =
      FindLastFragmentOnMesh(block, "m1", {p.f3.getOperation()});
  ASSERT_NE(last, nullptr);
  EXPECT_EQ(last, p.f2);
}

TEST(FindLastFragmentOnMesh, ExcludeAllReturnsNullptr) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  Block* block = &p.func_op.getBody().front();
  // Exclude all m1 fragments.
  FragmentOp last = FindLastFragmentOnMesh(
      block, "m1",
      {p.f0.getOperation(), p.f2.getOperation(), p.f3.getOperation()});
  EXPECT_EQ(last, nullptr);
}

TEST(FindLastFragmentOnMesh, SingleMeshFragment) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  Block* block = &p.func_op.getBody().front();
  // m2 has only f1.
  FragmentOp last = FindLastFragmentOnMesh(block, "m2");
  ASSERT_NE(last, nullptr);
  EXPECT_EQ(last, p.f1);
}

// ===== FindFirstFragmentOnMesh tests =====

TEST(FindFirstFragmentOnMesh, ReturnsFirstOnMatchingMesh) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  Block* block = &p.func_op.getBody().front();
  // m1 fragments are f0, f2, f3.  The first one should be f0.
  FragmentOp first = FindFirstFragmentOnMesh(block, "m1");
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first, p.f0);
}

TEST(FindFirstFragmentOnMesh, ReturnsNullptrForMissingMesh) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  Block* block = &p.func_op.getBody().front();
  FragmentOp result = FindFirstFragmentOnMesh(block, "nonexistent");
  EXPECT_EQ(result, nullptr);
}

TEST(FindFirstFragmentOnMesh, RespectsExcludeList) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  Block* block = &p.func_op.getBody().front();
  // Exclude f0 — the first m1 fragment should now be f2.
  FragmentOp first =
      FindFirstFragmentOnMesh(block, "m1", {p.f0.getOperation()});
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first, p.f2);
}

TEST(FindFirstFragmentOnMesh, ExcludeAllReturnsNullptr) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  Block* block = &p.func_op.getBody().front();
  FragmentOp first = FindFirstFragmentOnMesh(
      block, "m1",
      {p.f0.getOperation(), p.f2.getOperation(), p.f3.getOperation()});
  EXPECT_EQ(first, nullptr);
}

// ===== FindLatestOperandProducer tests =====

TEST(FindLatestOperandProducer, ReturnsLatestProducer) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  // f3 takes %2 (result of f2) as operand.  f2 is the latest producer.
  Operation* latest = FindLatestOperandProducer(p.f3);
  ASSERT_NE(latest, nullptr);
  EXPECT_EQ(latest, p.f2.getOperation());
}

TEST(FindLatestOperandProducer, ReturnsNullptrForBlockArgOnly) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  // f0 takes %arg0 (a block argument) as operand — no defining op.
  Operation* latest = FindLatestOperandProducer(p.f0);
  EXPECT_EQ(latest, nullptr);
}

TEST(FindLatestOperandProducer, ReturnsNullptrForBlockArgOnly_M2) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  // f1 takes %arg1 (a block argument) as operand — no defining op.
  Operation* latest = FindLatestOperandProducer(p.f1);
  EXPECT_EQ(latest, nullptr);
}

TEST(FindLatestOperandProducer, WithMultipleOperands) {
  // Build a program where a fragment has operands from two different producers
  // at different positions in the block.
  const char kProgram[] = R"mlir(
    !mt = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mt) -> !mt
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["early"(0)]> (%arg0)
           (%a0: tensor<4x8xf32>) {
        mpmd.return %a0 : tensor<4x8xf32>
      } : (!mt) -> !mt

      %1 = mpmd.fragment<mesh="m1", origin=["late"(0)]> (%arg0)
           (%a1: tensor<4x8xf32>) {
        mpmd.return %a1 : tensor<4x8xf32>
      } : (!mt) -> !mt

      %2 = mpmd.fragment<mesh="m1", origin=["consumer"(0)]> (%0, %1)
           (%a2: tensor<4x8xf32>, %a3: tensor<4x8xf32>) {
        mpmd.return %a2 : tensor<4x8xf32>
      } : (!mt, !mt) -> !mt

      return %2 : !mt
    }
  )mlir";
  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  ++it;  // skip "early"
  Operation* late = &*it++;
  Operation* consumer = &*it;

  // The consumer uses results from both early and late.
  // late is the latest operand producer.
  Operation* latest = FindLatestOperandProducer(consumer);
  ASSERT_NE(latest, nullptr);
  EXPECT_EQ(latest, late);
}

// ===== CanMoveAfter tests =====

TEST(CanMoveAfter, CanMoveWhenNoIntermediateUsers) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  // f0's result is used by f2 (not between f0 and f1).
  // f1 is on m2 and doesn't use f0's result.
  // So f0 can be moved after f1 only if no user of f0 is at or before f1.
  // Actually: f0 -> f1 -> f2 -> f3.  f0's user is f2.
  // f2 is after f1, so f0 CAN be moved after f1.
  EXPECT_TRUE(CanMoveAfter(p.f0, p.f1));
}

TEST(CanMoveAfter, CannotMoveWhenUserIsTarget) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  // f0's result is used by f2. Trying to move f0 after f2 fails because
  // f2 itself uses f0's result (user == target).
  EXPECT_FALSE(CanMoveAfter(p.f0, p.f2));
}

TEST(CanMoveAfter, CannotMoveWhenUserIsBeforeTarget) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  // f0's result is used by f2.  Trying to move f0 after f3: f2 is before f3,
  // so the user f2 would be before the target f3.
  EXPECT_FALSE(CanMoveAfter(p.f0, p.f3));
}

TEST(CanMoveAfter, ReturnsFalseWhenNotBefore) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  // f3 is after f0 in block order — CanMoveAfter requires op_to_move to
  // precede target_op.
  EXPECT_FALSE(CanMoveAfter(p.f3, p.f0));
}

TEST(CanMoveAfter, ReturnsFalseForDifferentBlocks) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  // Operations inside different blocks (the body of f0 vs. the func block).
  // Get an op from inside f0's region.
  Operation* inner_op = &p.f0.getRegion().front().front();
  EXPECT_FALSE(CanMoveAfter(inner_op, p.f1));
}

// ===== EnsureAfter tests =====

TEST(EnsureAfter, NoOpWhenAlreadyAfter) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  // f2 is already after f0 — EnsureAfter should be a no-op returning true.
  EXPECT_TRUE(EnsureAfter(p.f2, p.f0));
  // f2 should still be after f0.
  EXPECT_TRUE(p.f0->isBeforeInBlock(p.f2));
}

TEST(EnsureAfter, NoOpWhenSameOp) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  // op == target — should be a no-op returning true.
  EXPECT_TRUE(EnsureAfter(p.f0, p.f0));
}

TEST(EnsureAfter, NoOpWhenTargetIsNull) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  // nullptr target — should return true.
  EXPECT_TRUE(EnsureAfter(p.f0, nullptr));
}

TEST(EnsureAfter, MovesWhenSafe) {
  // f1(m2) has no users in the block (only used by return).
  // We can move f1 after f2 safely since f1's result is only used by return
  // which comes after everything.
  const char kProgram[] = R"mlir(
    !m1 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    !m2 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
    func.func @main(%arg0: !m1, %arg1: !m2) -> (!m1, !m2)
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=4]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["a"(0)]> (%arg0)
           (%a: tensor<4x8xf32>) {
        mpmd.return %a : tensor<4x8xf32>
      } : (!m1) -> !m1

      %1 = mpmd.fragment<mesh="m2", origin=["b"(0)]> (%arg1)
           (%b: tensor<4x8xf32>) {
        mpmd.return %b : tensor<4x8xf32>
      } : (!m2) -> !m2

      %2 = mpmd.fragment<mesh="m1", origin=["c"(0)]> (%0)
           (%c: tensor<4x8xf32>) {
        mpmd.return %c : tensor<4x8xf32>
      } : (!m1) -> !m1

      return %2, %1 : !m1, !m2
    }
  )mlir";
  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  ++it;  // skip "a"
  FragmentOp b = mlir::cast<FragmentOp>(*it++);
  FragmentOp c = mlir::cast<FragmentOp>(*it);

  // b is before c; move b after c.
  ASSERT_TRUE(b->isBeforeInBlock(c));
  EXPECT_TRUE(EnsureAfter(b, c));
  // Now b should be after c.
  EXPECT_TRUE(c->isBeforeInBlock(b));
}

TEST(EnsureAfter, ReturnsFalseWhenUnsafe) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  // f2's result is used by f3. Moving f2 after f3 is unsafe.
  EXPECT_FALSE(EnsureAfter(p.f2, p.f3));
  // f2 should NOT have been moved — still before f3.
  EXPECT_TRUE(p.f2->isBeforeInBlock(p.f3));
}

// ===== SaveFragmentAttrs / RestoreFragmentAttrs tests =====

// A reusable single-mesh program with two fragments.
constexpr char kTwoFragmentProgram[] = R"mlir(
  !mt = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  func.func @main(%arg0: !mt) -> !mt
    attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
    %0 = mpmd.fragment<mesh="m1", origin=["f0"(0)]> (%arg0)
         (%a0: tensor<4x8xf32>) {
      mpmd.return %a0 : tensor<4x8xf32>
    } : (!mt) -> !mt

    %1 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%0)
         (%a1: tensor<4x8xf32>) {
      mpmd.return %a1 : tensor<4x8xf32>
    } : (!mt) -> !mt

    return %1 : !mt
  }
)mlir";

// A reusable single-mesh program with one fragment.
constexpr char kSingleFragmentProgram[] = R"mlir(
  !mt = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  func.func @main(%arg0: !mt) -> !mt
    attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
    %0 = mpmd.fragment<mesh="m1", origin=["f0"(0)]> (%arg0)
         (%a0: tensor<4x8xf32>) {
      mpmd.return %a0 : tensor<4x8xf32>
    } : (!mt) -> !mt
    return %0 : !mt
  }
)mlir";

TEST(SaveRestoreFragmentAttrs, SavesAndRestoresInferredByAndCallCounter) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kTwoFragmentProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  FragmentOp f0 = mlir::cast<FragmentOp>(*it++);
  FragmentOp f1 = mlir::cast<FragmentOp>(*it);

  OpBuilder builder(&context);

  // Set up f0 with inferred_by and call_counter attributes.
  SetInferredByAttr(f0, "pass_a", builder);
  f0->setAttr(kCallCounterAttrName, builder.getUI32IntegerAttr(42));

  // Save the attributes from f0.
  SavedFragmentAttrs saved = SaveFragmentAttrs(f0);
  ASSERT_NE(saved.inferred_by, nullptr);
  ASSERT_NE(saved.call_counter, nullptr);
  EXPECT_EQ(saved.call_counter.getValue().getZExtValue(), 42u);

  // Restore onto f1, adding "pass_b" to the inferred_by list.
  RestoreFragmentAttrs(f1, saved, "pass_b", builder);

  // Verify call_counter was restored.
  auto restored_counter =
      f1->getAttrOfType<IntegerAttr>(kCallCounterAttrName);
  ASSERT_NE(restored_counter, nullptr);
  EXPECT_EQ(restored_counter.getValue().getZExtValue(), 42u);

  // Verify inferred_by was restored with pass_b appended.
  auto restored_inferred_by =
      f1->getAttrOfType<ArrayAttr>(kInferredByAttr);
  ASSERT_NE(restored_inferred_by, nullptr);
  ASSERT_EQ(restored_inferred_by.size(), 2);
  EXPECT_EQ(mlir::cast<StringAttr>(restored_inferred_by[0]).getValue(),
            "pass_a");
  EXPECT_EQ(mlir::cast<StringAttr>(restored_inferred_by[1]).getValue(),
            "pass_b");
}

TEST(SaveRestoreFragmentAttrs, SavesNullWhenNoAttrs) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kSingleFragmentProgram, &context);
  FuncOp func_op = GetMainFunction(*module);
  FragmentOp f0 = mlir::cast<FragmentOp>(*func_op.getOps().begin());

  // f0 has no inferred_by or call_counter attrs.
  SavedFragmentAttrs saved = SaveFragmentAttrs(f0);
  EXPECT_EQ(saved.inferred_by, nullptr);
  EXPECT_EQ(saved.call_counter, nullptr);
}

TEST(SaveRestoreFragmentAttrs, RestoreWithNoSavedInferredByCreatesNew) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kSingleFragmentProgram, &context);
  FuncOp func_op = GetMainFunction(*module);
  FragmentOp f0 = mlir::cast<FragmentOp>(*func_op.getOps().begin());

  OpBuilder builder(&context);

  // Saved attrs have no inferred_by.
  SavedFragmentAttrs saved = {/*inferred_by=*/nullptr,
                              /*call_counter=*/nullptr};
  RestoreFragmentAttrs(f0, saved, "my_pass", builder);

  // Should create an inferred_by list with just "my_pass".
  auto inferred_by = f0->getAttrOfType<ArrayAttr>(kInferredByAttr);
  ASSERT_NE(inferred_by, nullptr);
  ASSERT_EQ(inferred_by.size(), 1);
  EXPECT_EQ(mlir::cast<StringAttr>(inferred_by[0]).getValue(), "my_pass");

  // No call_counter should be set.
  EXPECT_FALSE(f0->hasAttr(kCallCounterAttrName));
}

}  // namespace
}  // namespace mlir::mpmd
