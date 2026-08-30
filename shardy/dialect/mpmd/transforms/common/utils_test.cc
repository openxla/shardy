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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/register.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/optimize/utils.h"
#include <gtest/gtest.h>

namespace mlir::mpmd {
namespace {

using ::mlir::func::FuncOp;

TEST(MergeFragments, BasicMerge) {
  const char kProgram[] = R"mlir(
    !mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_tensor) -> (!mesh_tensor, !mesh_tensor)
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f0"(0)]> (%arg0)
           (%arg1: tensor<4x8xf32>) {
        mpmd.return %arg1 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %1 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0)
           (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      return %0, %1 : !mesh_tensor, !mesh_tensor
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  FragmentOp f0 = mlir::cast<FragmentOp>(*it++);
  FragmentOp f1 = mlir::cast<FragmentOp>(*it);

  IRRewriter rewriter(&context);
  rewriter.setInsertionPoint(f1);

  FragmentOp merged = MergeFragments(f0, f1, rewriter);

  // Both fragments had 1 operand each but share %arg0, so the merged
  // fragment should have 1 deduplicated operand.
  EXPECT_EQ(merged.getNumOperands(), 1);
  // The merged fragment should produce 2 results (one from each original).
  EXPECT_EQ(merged.getNumResults(), 2);
  // Original ops should have been erased — only the merged fragment + return
  // should remain.
  int num_ops = 0;
  for (auto& op : func_op.getOps()) {
    (void)op;
    ++num_ops;
  }
  // merged fragment + return
  EXPECT_EQ(num_ops, 2);
}

TEST(MergeFragments, SharedOperandsAreDeduplicated) {
  const char kProgram[] = R"mlir(
    !mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_tensor, %arg1: !mesh_tensor)
        -> (!mesh_tensor, !mesh_tensor)
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f0"(0)]> (%arg0, %arg1)
           (%a: tensor<4x8xf32>, %b: tensor<4x8xf32>) {
        mpmd.return %a : tensor<4x8xf32>
      } : (!mesh_tensor, !mesh_tensor) -> !mesh_tensor

      %1 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0, %arg1)
           (%c: tensor<4x8xf32>, %d: tensor<4x8xf32>) {
        mpmd.return %d : tensor<4x8xf32>
      } : (!mesh_tensor, !mesh_tensor) -> !mesh_tensor

      return %0, %1 : !mesh_tensor, !mesh_tensor
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  FragmentOp f0 = mlir::cast<FragmentOp>(*it++);
  FragmentOp f1 = mlir::cast<FragmentOp>(*it);

  IRRewriter rewriter(&context);
  rewriter.setInsertionPoint(f1);

  FragmentOp merged = MergeFragments(f0, f1, rewriter);

  // Both fragments take (%arg0, %arg1). The shared operands should be
  // deduplicated, so the merged fragment should have exactly 2 operands.
  ASSERT_EQ(merged.getNumOperands(), 2);
  EXPECT_EQ(merged->getOperand(0), func_op.getArgument(0));
  EXPECT_EQ(merged->getOperand(1), func_op.getArgument(1));
}

TEST(MergeFragments, ProducerResultUsedByConsumer) {
  const char kProgram[] = R"mlir(
    !mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_tensor) -> !mesh_tensor
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f0"(0)]> (%arg0)
           (%arg1: tensor<4x8xf32>) {
        mpmd.return %arg1 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %1 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%0)
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
  FragmentOp f0 = mlir::cast<FragmentOp>(*it++);
  FragmentOp f1 = mlir::cast<FragmentOp>(*it);

  IRRewriter rewriter(&context);
  rewriter.setInsertionPoint(f1);

  FragmentOp merged = MergeFragments(f0, f1, rewriter);

  // The producer's result (%0) was the consumer's only operand. After merging,
  // the merged fragment should only take %arg0 (the producer's input).
  ASSERT_EQ(merged.getNumOperands(), 1);
  EXPECT_EQ(merged->getOperand(0), func_op.getArgument(0));
  // The merged fragment should produce 1 result (the consumer's output).
  EXPECT_EQ(merged.getNumResults(), 1);
}

TEST(MergeFragments, PreservesControlOperands) {
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

  FragmentOp merged_fragment = MergeFragments(f1, f3, rewriter);

  ASSERT_TRUE(merged_fragment->hasAttr(kControlOperandStartIdxAttrName));
  int control_start_idx =
      merged_fragment
          ->getAttrOfType<IntegerAttr>(kControlOperandStartIdxAttrName)
          .getInt();

  ASSERT_EQ(merged_fragment.getNumOperands(), control_start_idx + 2);
  EXPECT_EQ(merged_fragment->getOperand(control_start_idx), f0->getResult(0));
  EXPECT_EQ(merged_fragment->getOperand(control_start_idx + 1),
            f2->getResult(0));
}

TEST(MergeAttributes, NoAttributes) {
  const char kProgram[] = R"mlir(
    !mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_tensor) -> (!mesh_tensor, !mesh_tensor)
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f0"(0)]> (%arg0)
           (%arg1: tensor<4x8xf32>) {
        mpmd.return %arg1 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %1 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0)
           (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      return %0, %1 : !mesh_tensor, !mesh_tensor
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  FragmentOp f0 = mlir::cast<FragmentOp>(*it++);
  FragmentOp f1 = mlir::cast<FragmentOp>(*it);

  auto attrs = MergeAttributes(f0, f1);
  EXPECT_TRUE(attrs.empty());
}

TEST(MergeAttributes, MergesMatchingCallCounters) {
  const char kProgram[] = R"mlir(
    !mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_tensor) -> (!mesh_tensor, !mesh_tensor)
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f0"(0)]> (%arg0)
           {call_counter = 5 : ui32}
           (%arg1: tensor<4x8xf32>) {
        mpmd.return %arg1 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %1 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0)
           {call_counter = 5 : ui32}
           (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      return %0, %1 : !mesh_tensor, !mesh_tensor
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  FragmentOp f0 = mlir::cast<FragmentOp>(*it++);
  FragmentOp f1 = mlir::cast<FragmentOp>(*it);

  auto attrs = MergeAttributes(f0, f1);
  ASSERT_EQ(attrs.size(), 1);
  EXPECT_EQ(attrs[0].first, kCallCounterAttrName);
  EXPECT_EQ(mlir::cast<IntegerAttr>(attrs[0].second).getUInt(), 5);
}

TEST(MergeAttributes, OnlyProducerHasCallCounter) {
  const char kProgram[] = R"mlir(
    !mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_tensor) -> (!mesh_tensor, !mesh_tensor)
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f0"(0)]> (%arg0)
           {call_counter = 3 : ui32}
           (%arg1: tensor<4x8xf32>) {
        mpmd.return %arg1 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %1 = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0)
           (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      return %0, %1 : !mesh_tensor, !mesh_tensor
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  FragmentOp f0 = mlir::cast<FragmentOp>(*it++);
  FragmentOp f1 = mlir::cast<FragmentOp>(*it);

  auto attrs = MergeAttributes(f0, f1);
  ASSERT_EQ(attrs.size(), 1);
  EXPECT_EQ(attrs[0].first, kCallCounterAttrName);
  EXPECT_EQ(mlir::cast<IntegerAttr>(attrs[0].second).getUInt(), 3);
}

TEST(MergeAttributes, MergesInferredByForInferredFragments) {
  const char kProgram[] = R"mlir(
    !mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_tensor) -> (!mesh_tensor, !mesh_tensor)
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
           {mpmd.inferred_by = ["pass_a"]}
           (%arg1: tensor<4x8xf32>) {
        mpmd.return %arg1 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %1 = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
           {mpmd.inferred_by = ["pass_b"]}
           (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      return %0, %1 : !mesh_tensor, !mesh_tensor
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  FragmentOp f0 = mlir::cast<FragmentOp>(*it++);
  FragmentOp f1 = mlir::cast<FragmentOp>(*it);

  auto attrs = MergeAttributes(f0, f1);
  ASSERT_EQ(attrs.size(), 1);
  EXPECT_EQ(attrs[0].first, kInferredByAttr);
  auto merged_array = mlir::cast<ArrayAttr>(attrs[0].second);
  ASSERT_EQ(merged_array.size(), 2);
  EXPECT_EQ(mlir::cast<StringAttr>(merged_array[0]).getValue(), "pass_a");
  EXPECT_EQ(mlir::cast<StringAttr>(merged_array[1]).getValue(), "pass_b");
}

TEST(MergeAttributes, SkipsInferredByForMixedUserAndInferredFragments) {
  const char kProgram[] = R"mlir(
    !mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_tensor) -> (!mesh_tensor, !mesh_tensor)
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f0"(0)]> (%arg0)
           (%arg1: tensor<4x8xf32>) {
        mpmd.return %arg1 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %1 = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
           {mpmd.inferred_by = ["pass_a"]}
           (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      return %0, %1 : !mesh_tensor, !mesh_tensor
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  FragmentOp f0 = mlir::cast<FragmentOp>(*it++);
  FragmentOp f1 = mlir::cast<FragmentOp>(*it);

  // One user fragment and one inferred fragment — inferred_by should not be
  // preserved since the merged result will be a user fragment.
  auto attrs = MergeAttributes(f0, f1);
  EXPECT_TRUE(attrs.empty());
}

TEST(MergeAttributes, BothCallCounterAndInferredBy) {
  const char kProgram[] = R"mlir(
    !mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_tensor) -> (!mesh_tensor, !mesh_tensor)
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
           {call_counter = 2 : ui32, mpmd.inferred_by = ["pass_a"]}
           (%arg1: tensor<4x8xf32>) {
        mpmd.return %arg1 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      %1 = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
           {call_counter = 2 : ui32, mpmd.inferred_by = ["pass_b"]}
           (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_tensor) -> !mesh_tensor

      return %0, %1 : !mesh_tensor, !mesh_tensor
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  auto it = func_op.getOps().begin();
  FragmentOp f0 = mlir::cast<FragmentOp>(*it++);
  FragmentOp f1 = mlir::cast<FragmentOp>(*it);

  auto attrs = MergeAttributes(f0, f1);
  ASSERT_EQ(attrs.size(), 2);
  ASSERT_EQ(attrs[0].first, kCallCounterAttrName);
  EXPECT_EQ(mlir::cast<IntegerAttr>(attrs[0].second).getUInt(), 2);
  ASSERT_EQ(attrs[1].first, kInferredByAttr);
  auto merged_array = mlir::cast<ArrayAttr>(attrs[1].second);
  ASSERT_EQ(merged_array.size(), 2);
  EXPECT_EQ(mlir::cast<StringAttr>(merged_array[0]).getValue(), "pass_a");
  EXPECT_EQ(mlir::cast<StringAttr>(merged_array[1]).getValue(), "pass_b");
}

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

TEST(FindLastFragmentOnMesh, ReturnsNullptrWhenNoEarlierSameMeshFragment) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  // f0 is the first m1 fragment. Other m1 fragments (f2, f3) exist after it,
  // but the search is backward-only.
  FragmentOp last = FindLastFragmentOnMesh(p.f0);
  EXPECT_EQ(last, nullptr);
}

TEST(FindLastFragmentOnMesh, FindsNearestPrecedingSameMeshFragment) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  // f3 is the last m1 fragment. Searching backward from f3 finds f2,
  // skipping f1 (which is on m2).
  FragmentOp last = FindLastFragmentOnMesh(p.f3);
  ASSERT_NE(last, nullptr);
  EXPECT_EQ(last, p.f2);
}

TEST(FindLatestOperandProducer, ReturnsLatestProducer) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  // f3's operand is %2 (from f2). The latest operand producer should be f2.
  Operation* latest = FindLatestOperandProducer(p.f3);
  ASSERT_NE(latest, nullptr);
  EXPECT_EQ(latest, p.f2.getOperation());
}

TEST(FindLatestOperandProducer, ReturnsNullptrForBlockArgOnly) {
  MLIRContext context;
  loadAllRequiredDialects(&context);
  auto p = ParseTwoMeshProgram(context);

  // f0's operand is %arg0 (a block argument). Should return nullptr.
  EXPECT_EQ(FindLatestOperandProducer(p.f0), nullptr);
}

TEST(FindLatestOperandProducer, WithMultipleOperands) {
  // Build a program where a fragment takes two op-defined operands.
  const char kProgram[] = R"mlir(
    !mt = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mt) -> !mt
      attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["a"(0)]> (%arg0)
           (%a: tensor<4x8xf32>) {
        mpmd.return %a : tensor<4x8xf32>
      } : (!mt) -> !mt

      %1 = mpmd.fragment<mesh="m1", origin=["b"(0)]> (%0)
           (%b: tensor<4x8xf32>) {
        mpmd.return %b : tensor<4x8xf32>
      } : (!mt) -> !mt

      %2 = mpmd.fragment<mesh="m1", origin=["c"(0)]> (%0, %1)
           (%c: tensor<4x8xf32>, %d: tensor<4x8xf32>) {
        mpmd.return %c : tensor<4x8xf32>
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
  ++it;  // skip "a"
  Operation* b = &*it++;
  Operation* c = &*it;

  // c takes (%0, %1). b is later than a, so latest producer should be b.
  Operation* latest = FindLatestOperandProducer(c);
  ASSERT_NE(latest, nullptr);
  EXPECT_EQ(latest, b);
}

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

}  // namespace
}  // namespace mlir::mpmd
