/* Copyright 2026 The Shardy Authors.

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

#include <cstdint>
#include <iterator>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/transforms/common/util.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"  // IWYU pragma: keep
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_EXPLICITGATHERSCATTERBATCHINGPASS
#include "shardy/dialect/sdy/transforms/import/passes.h.inc"

namespace {

// Helper function to slice out a specific index component from an index
// vector and concatenate the remaining parts. This is used when an implicit
// batch dimension is being removed from the index map.
Value sliceAndConcatenateIndices(PatternRewriter& rewriter, Location loc,
                                 Value indices, RankedTensorType indicesType,
                                 int64_t indexVectorDim, int64_t removedIdx) {
  int64_t rank = indicesType.getRank();
  auto currentShape = indicesType.getShape();
  int64_t total = currentShape[indexVectorDim];

  // If start_indices is already a ConcatenateOp along the index vector dim,
  // simply remove the component at removedIdx.
  if (auto concatOp = indices.getDefiningOp<stablehlo::ConcatenateOp>()) {
    if (static_cast<int64_t>(concatOp.getDimension()) == indexVectorDim &&
        static_cast<int64_t>(concatOp.getInputs().size()) == total) {
      SmallVector<Value> inputs = llvm::to_vector(concatOp.getInputs());
      inputs.erase(inputs.begin() + removedIdx);

      // If only one component remains, return it directly.
      if (inputs.size() == 1) return inputs[0];

      auto concatShape = llvm::to_vector(currentShape);
      concatShape[indexVectorDim] -= 1;
      auto concatType =
          RankedTensorType::get(concatShape, indicesType.getElementType());
      return stablehlo::ConcatenateOp::create(rewriter, loc, concatType, inputs,
                                              indexVectorDim);
    }
  }

  SmallVector<Value> keptParts;
  auto addPart = [&](int64_t startIdx, int64_t endIdx) {
    if (startIdx >= endIdx) return;

    SmallVector<int64_t> start(rank, 0);
    SmallVector<int64_t> limit(currentShape.begin(), currentShape.end());
    SmallVector<int64_t> strides(rank, 1);
    start[indexVectorDim] = startIdx;
    limit[indexVectorDim] = endIdx;

    auto sliceShape = llvm::to_vector(currentShape);
    sliceShape[indexVectorDim] = endIdx - startIdx;
    auto sliceType =
        RankedTensorType::get(sliceShape, indicesType.getElementType());

    keptParts.push_back(stablehlo::SliceOp::create(
        rewriter, loc, sliceType, indices, rewriter.getDenseI64ArrayAttr(start),
        rewriter.getDenseI64ArrayAttr(limit),
        rewriter.getDenseI64ArrayAttr(strides)));
  };

  // Collect indices before and after the removed iota component.
  addPart(0, removedIdx);
  addPart(removedIdx + 1, total);

  if (keptParts.size() == 1) return keptParts[0];

  auto resultShape = llvm::to_vector(currentShape);
  resultShape[indexVectorDim] -= 1;
  auto resultType =
      RankedTensorType::get(resultShape, indicesType.getElementType());
  return stablehlo::ConcatenateOp::create(rewriter, loc, resultType, keptParts,
                                          indexVectorDim);
}

// Helper function to unwrap values through common StableHLO operations
// like Reshape, BroadcastInDim, Select, and Clamp to find the original
// source.
Value unwrapReshapes(Value v) {
  while (Operation* op = v.getDefiningOp()) {
    Value nextV =
        llvm::TypeSwitch<Operation*, Value>(op)
            .Case<stablehlo::ReshapeOp, stablehlo::BroadcastInDimOp,
                  stablehlo::ClampOp>([](auto op) { return op.getOperand(); })
            .Case([](stablehlo::SelectOp select) {
              return matchPattern(select.getOnFalse(),
                                  m_Op<stablehlo::IotaOp>())
                         ? select.getOnFalse()
                         : select.getOnTrue();
            })
            .Default([](Operation*) { return Value(); });

    if (!nextV) break;
    v = nextV;
  }
  return v;
}

// Verify that the index component at removedIndexVectorIdx originates
// from an iota for identity mapping. Shared between gather and scatter.
bool verifyIotaOrigin(Value indicesVal, int64_t indexVectorDim,
                      int64_t removedIndexVectorIdx) {
  Value potentialIota = indicesVal;
  if (auto concatOp = potentialIota.getDefiningOp<stablehlo::ConcatenateOp>()) {
    if (static_cast<int64_t>(concatOp.getDimension()) != indexVectorDim ||
        removedIndexVectorIdx >=
            static_cast<int64_t>(concatOp.getInputs().size())) {
      return false;
    }
    potentialIota = concatOp.getInputs()[removedIndexVectorIdx];
  }

  Value unwrapped = unwrapReshapes(potentialIota);
  auto iota = dyn_cast_or_null<stablehlo::IotaOp>(unwrapped.getDefiningOp());
  return iota && iota.getIotaDimension() == 0;
}

struct ExplicitGatherBatchingPattern
    : public OpRewritePattern<stablehlo::GatherOp> {
  using OpRewritePattern<stablehlo::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::GatherOp op,
                                PatternRewriter& rewriter) const override {
    auto dimNums = op.getDimensionNumbers();

    // 1. If the gather already has explicit batching dims, no rewrite
    // needed.
    if (!dimNums.getOperandBatchingDims().empty()) {
      return failure();
    }

    auto operandType = op.getOperand().getType();
    auto startIndicesType = op.getStartIndices().getType();

    // Skip if either operand or start_indices are scalars.
    if (operandType.getRank() == 0 || startIndicesType.getRank() == 0) {
      return failure();
    }

    // The current implementation can only identify dimensions 0 in
    // operand and start_indices as implicit batch. As such, we check
    // whether dimensions 0 in operand and start_indices have matching
    // sizes, and whether the values in start_indices used to index
    // dimension 0 in operand originate from iota.
    int64_t indexVectorDim = dimNums.getIndexVectorDim();

    // We cannot reduce the index vector dimension further if the dimension size
    // is 1.
    if (indexVectorDim >= startIndicesType.getRank() ||
        startIndicesType.getDimSize(indexVectorDim) <= 1) {
      return failure();
    }

    int64_t batchDimSize = operandType.getDimSize(0);
    if (startIndicesType.getDimSize(0) != batchDimSize) {
      return failure();
    }

    // Ensure the first dimension is mapped in the start_index_map.
    auto startIndexMap = llvm::to_vector(dimNums.getStartIndexMap());
    auto* it = llvm::find(startIndexMap, 0);
    if (it == startIndexMap.end()) {
      return failure();
    }

    // Locate the index in the index vector corresponding to the
    // batch dimension.
    int64_t removedIndexVectorIdx = std::distance(startIndexMap.begin(), it);

    // 2. Recognize some code patterns where the index component at
    // removedIndexVectorIdx originates from iota for identity
    // mapping.
    if (!verifyIotaOrigin(op.getStartIndices(), indexVectorDim,
                          removedIndexVectorIdx)) {
      return failure();
    }

    // 3. Remove the implicit batch dimension from start_index_map
    // and collapsed_slice_dims, and explicitly add it to the batching
    // dimensions.
    startIndexMap.erase(it);

    auto collapsedSliceDims = llvm::to_vector(dimNums.getCollapsedSliceDims());
    auto* collapsedIt = llvm::find(collapsedSliceDims, 0);
    if (collapsedIt == collapsedSliceDims.end()) {
      return failure();
    }
    collapsedSliceDims.erase(collapsedIt);

    // 4. Construct the new start_indices tensor by slicing out the
    // component at removedIndexVectorIdx.
    Value newStartIndices = sliceAndConcatenateIndices(
        rewriter, op.getLoc(), op.getStartIndices(), startIndicesType,
        indexVectorDim, removedIndexVectorIdx);

    // 5. Create the new GatherDimensionNumbersAttr with explicit
    // batching.
    SmallVector<int64_t> operandBatchingDims = {0};
    SmallVector<int64_t> startIndicesBatchingDims = {0};

    auto newDimNums = stablehlo::GatherDimensionNumbersAttr::get(
        rewriter.getContext(), dimNums.getOffsetDims(), collapsedSliceDims,
        operandBatchingDims, startIndicesBatchingDims, startIndexMap,
        indexVectorDim);

    auto newGatherOp = stablehlo::GatherOp::create(
        rewriter, op.getLoc(), op.getType(), op.getOperand(),
        newStartIndices, newDimNums, op.getSliceSizes(),
        op.getIndicesAreSorted());
    copyAttributes(op, newGatherOp, {"dimension_numbers",
                                     "indices_are_sorted"});
    rewriter.replaceOp(op, newGatherOp.getResult());

    return success();
  }
};

struct ExplicitScatterBatchingPattern
    : public OpRewritePattern<stablehlo::ScatterOp> {
  using OpRewritePattern<stablehlo::ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ScatterOp op,
                                PatternRewriter& rewriter) const override {
    auto dimNums = op.getScatterDimensionNumbers();

    // 1. If scatter already has explicit batching dims, no rewrite
    // needed.
    if (!dimNums.getInputBatchingDims().empty()) {
      return failure();
    }

    auto inputType = cast<RankedTensorType>(op.getInputs().front().getType());
    auto scatterIndicesType = op.getScatterIndices().getType();

    // Skip if either input or scatter_indices are scalars.
    if (inputType.getRank() == 0 || scatterIndicesType.getRank() == 0) {
      return failure();
    }

    // The current implementation can only identify dimensions 0 in
    // operand and start_indices as implicit batch. As such, we check
    // whether dimensions 0 in operand and start_indices have matching
    // sizes, and whether the values in start_indices used to index
    // dimension 0 in operand originate from iota.
    int64_t indexVectorDim = dimNums.getIndexVectorDim();

    // We cannot reduce the index vector dimension further if the dimension size
    // is 1.
    if (indexVectorDim >= scatterIndicesType.getRank() ||
        scatterIndicesType.getDimSize(indexVectorDim) <= 1) {
      return failure();
    }

    int64_t batchDimSize = inputType.getDimSize(0);
    if (scatterIndicesType.getDimSize(0) != batchDimSize) {
      return failure();
    }

    auto scatterDimsToOperandDims =
        llvm::to_vector(dimNums.getScatterDimsToOperandDims());
    auto* it = llvm::find(scatterDimsToOperandDims, 0);
    if (it == scatterDimsToOperandDims.end()) {
      return failure();
    }

    int64_t removedIndexVectorIdx =
        std::distance(scatterDimsToOperandDims.begin(), it);

    // Verify iota origin using shared helper.
    if (!verifyIotaOrigin(op.getScatterIndices(), indexVectorDim,
                          removedIndexVectorIdx)) {
      return failure();
    }

    // 3. Update attributes for scatter.
    scatterDimsToOperandDims.erase(it);

    auto insertedWindowDims = llvm::to_vector(dimNums.getInsertedWindowDims());
    auto* insertedIt = llvm::find(insertedWindowDims, 0);
    if (insertedIt == insertedWindowDims.end()) {
      return failure();
    }
    insertedWindowDims.erase(insertedIt);

    // 4. Slice out the iota index component.
    Value newScatterIndices = sliceAndConcatenateIndices(
        rewriter, op.getLoc(), op.getScatterIndices(), scatterIndicesType,
        indexVectorDim, removedIndexVectorIdx);

    // 5. Construct new scatter dim numbers.
    SmallVector<int64_t> inputBatchingDims = {0};
    SmallVector<int64_t> scatterIndicesBatchingDims = {0};

    auto newScatterDimNums = stablehlo::ScatterDimensionNumbersAttr::get(
        rewriter.getContext(), dimNums.getUpdateWindowDims(),
        insertedWindowDims, inputBatchingDims, scatterIndicesBatchingDims,
        scatterDimsToOperandDims, indexVectorDim);

    auto newScatterOp = stablehlo::ScatterOp::create(
        rewriter, op.getLoc(), op.getResultTypes(), op.getInputs(),
        newScatterIndices, op.getUpdates(), newScatterDimNums,
        op.getIndicesAreSorted(), op.getUniqueIndices());
    copyAttributes(op, newScatterOp, {"scatter_dimension_numbers",
                                       "indices_are_sorted",
                                       "unique_indices"});

    rewriter.inlineRegionBefore(op.getUpdateComputation(),
                                newScatterOp.getUpdateComputation(),
                                newScatterOp.getUpdateComputation().end());

    rewriter.replaceOp(op, newScatterOp.getResults());

    return success();
  }
};

class ExplicitGatherScatterBatchingPass
    : public impl::ExplicitGatherScatterBatchingPassBase<
          ExplicitGatherScatterBatchingPass> {
 public:
  using ExplicitGatherScatterBatchingPassBase::
      ExplicitGatherScatterBatchingPassBase;

 protected:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExplicitGatherBatchingPattern>(&getContext());
    patterns.add<ExplicitScatterBatchingPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
