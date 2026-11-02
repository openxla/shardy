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

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
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

// Returns the corresponding dimension in `inType` if `currDim` in `outType`
// maps 1-to-1 without splitting or merging non-unit dimensions.
std::optional<int64_t> matchNonUnitReshapeDim(RankedTensorType inType,
                                              RankedTensorType outType,
                                              int64_t currDim) {
  if (outType.getDimSize(currDim) == 1) {
    return std::nullopt;
  }
  auto getNonUnitDims = [](RankedTensorType type) {
    SmallVector<int64_t> nonUnit;
    for (int64_t s : type.getShape()) {
      if (s != 1) nonUnit.push_back(s);
    }
    return nonUnit;
  };
  if (getNonUnitDims(inType) != getNonUnitDims(outType)) {
    return std::nullopt;
  }
  int64_t nonUnitBeforeOut = 0;
  for (int64_t i = 0; i < currDim; ++i) {
    if (outType.getDimSize(i) != 1) ++nonUnitBeforeOut;
  }
  int64_t nonUnitBeforeIn = 0;
  for (int64_t i = 0; i < inType.getRank(); ++i) {
    if (inType.getDimSize(i) != 1) {
      if (nonUnitBeforeIn == nonUnitBeforeOut) {
        if (inType.getDimSize(i) == outType.getDimSize(currDim)) return i;
        return std::nullopt;
      }
      ++nonUnitBeforeIn;
    }
  }
  return std::nullopt;
}

// Checks if dimension `dim` in `val` evaluates to coord[dim] originating
// from an IotaOp along that dimension.
bool tracesToIota(Value val, int64_t dim) {
  int64_t currDim = dim;
  while (val) {
    if (auto iota = val.getDefiningOp<stablehlo::IotaOp>()) {
      return static_cast<int64_t>(iota.getIotaDimension()) == currDim;
    }
    if (auto convert = val.getDefiningOp<stablehlo::ConvertOp>()) {
      val = convert.getOperand();
      continue;
    }
    if (auto concat = val.getDefiningOp<stablehlo::ConcatenateOp>()) {
      if (concat.getInputs().size() == 1) {
        val = concat.getInputs().front();
        continue;
      }
      return false;
    }
    if (auto bcast = val.getDefiningOp<stablehlo::BroadcastInDimOp>()) {
      auto bcastDims = bcast.getBroadcastDimensions();
      const auto* it = llvm::find(bcastDims, currDim);
      if (it == bcastDims.end()) return false;
      currDim = std::distance(bcastDims.begin(), it);
      val = bcast.getOperand();
      continue;
    }
    if (auto reshape = val.getDefiningOp<stablehlo::ReshapeOp>()) {
      auto inType = reshape.getOperand().getType();
      auto outType = reshape.getType();
      auto matchingDim = matchNonUnitReshapeDim(inType, outType, currDim);
      if (!matchingDim) return false;
      currDim = *matchingDim;
      val = reshape.getOperand();
      continue;
    }
    break;
  }
  return false;
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
  return tracesToIota(potentialIota, 0);
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
        rewriter, op.getLoc(), op.getType(), op.getOperand(), newStartIndices,
        newDimNums, op.getSliceSizes(), op.getIndicesAreSorted());
    copyAttributes(op, newGatherOp,
                   {"dimension_numbers", "indices_are_sorted"});
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
    copyAttributes(
        op, newScatterOp,
        {"scatter_dimension_numbers", "indices_are_sorted", "unique_indices"});

    rewriter.inlineRegionBefore(op.getUpdateComputation(),
                                newScatterOp.getUpdateComputation(),
                                newScatterOp.getUpdateComputation().end());

    rewriter.replaceOp(op, newScatterOp.getResults());

    return success();
  }
};

// Returns true if indexVectorDim is absent or size <= 1.
bool isDegenerateIndexVector(int64_t indexVectorDim, RankedTensorType type) {
  return indexVectorDim >= type.getRank() ||
         type.getDimSize(indexVectorDim) <= 1;
}

// Returns the dimension in `v` that traces back to an identity iota dimension
// of size `expectedDimSize`.
// `v` may be wrapped in Concatenate, Reshape, or BroadcastInDim.
std::optional<int64_t> findIdentityIotaDim(Value v, int64_t expectedDimSize,
                                           int64_t indexVectorDim) {
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType) return std::nullopt;
  int64_t rank = tensorType.getRank();

  for (int64_t d = 0; d < rank; ++d) {
    if (d == indexVectorDim && indexVectorDim < rank) continue;
    if (tensorType.getDimSize(d) == expectedDimSize && tracesToIota(v, d)) {
      return d;
    }
  }

  return std::nullopt;
}

bool isSupportedCombiner(Operation* op) {
  return isa<stablehlo::AddOp, stablehlo::MulOp, stablehlo::MaxOp,
             stablehlo::MinOp, stablehlo::OrOp, stablehlo::AndOp>(op);
}

// Returns the binary combiner operation in the update computation block if the
// block computes a valid binary reduction over its two arguments.
Operation* getCombinerOp(Region& updateComputation) {
  if (updateComputation.empty() || updateComputation.front().empty()) {
    return nullptr;
  }
  Block& block = updateComputation.front();
  if (block.getNumArguments() != 2) return nullptr;
  auto returnOp = dyn_cast<stablehlo::ReturnOp>(block.getTerminator());
  if (!returnOp || returnOp.getResults().size() != 1) return nullptr;
  Operation* op = returnOp.getResults().front().getDefiningOp();
  if (!op || op->getNumOperands() != 2 || !isSupportedCombiner(op)) {
    return nullptr;
  }
  Value arg0 = block.getArgument(0);
  Value arg1 = block.getArgument(1);
  bool matchesArgs = (op->getOperand(0) == arg0 && op->getOperand(1) == arg1) ||
                     (op->getOperand(0) == arg1 && op->getOperand(1) == arg0);
  return matchesArgs ? op : nullptr;
}

// Returns the identity (neutral) constant for reduction with `combinerOp`.
Value getCombinerIdentityConstant(PatternRewriter& rewriter, Location loc,
                                  Operation* combinerOp,
                                  RankedTensorType scalarType) {
  Type elemType = scalarType.getElementType();
  Attribute attr;
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    const llvm::fltSemantics& sem = floatType.getFloatSemantics();
    if (isa<stablehlo::AddOp>(combinerOp)) {
      attr = DenseElementsAttr::get(scalarType, APFloat::getZero(sem));
    } else if (isa<stablehlo::MulOp>(combinerOp)) {
      attr = DenseElementsAttr::get(scalarType, APFloat(sem, 1));
    } else if (isa<stablehlo::MaxOp>(combinerOp)) {
      attr = DenseElementsAttr::get(scalarType,
                                    APFloat::getInf(sem, /*Negative=*/true));
    } else if (isa<stablehlo::MinOp>(combinerOp)) {
      attr = DenseElementsAttr::get(scalarType,
                                    APFloat::getInf(sem, /*Negative=*/false));
    }
  } else if (auto intType = dyn_cast<IntegerType>(elemType)) {
    unsigned width = intType.getWidth();
    bool isUnsigned = intType.isUnsigned();
    if (isa<stablehlo::AddOp, stablehlo::OrOp>(combinerOp)) {
      attr = DenseElementsAttr::get(scalarType, APInt::getZero(width));
    } else if (isa<stablehlo::MulOp>(combinerOp)) {
      attr = DenseElementsAttr::get(scalarType, APInt(width, 1));
    } else if (isa<stablehlo::AndOp>(combinerOp)) {
      attr = DenseElementsAttr::get(scalarType, APInt::getAllOnes(width));
    } else if (isa<stablehlo::MaxOp>(combinerOp)) {
      attr = DenseElementsAttr::get(
          scalarType, isUnsigned ? APInt::getMinValue(width)
                                 : APInt::getSignedMinValue(width));
    } else if (isa<stablehlo::MinOp>(combinerOp)) {
      attr = DenseElementsAttr::get(
          scalarType, isUnsigned ? APInt::getMaxValue(width)
                                 : APInt::getSignedMaxValue(width));
    }
  }
  return attr ? stablehlo::ConstantOp::create(rewriter, loc, scalarType,
                                              cast<ElementsAttr>(attr))
              : Value();
}

struct DegenerateGatherToBroadcastPattern
    : public OpRewritePattern<stablehlo::GatherOp> {
  using OpRewritePattern<stablehlo::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::GatherOp op,
                                PatternRewriter& rewriter) const override {
    auto dimNums = op.getDimensionNumbers();
    if (!dimNums.getOperandBatchingDims().empty()) {
      return failure();
    }

    auto operandType = op.getOperand().getType();
    auto startIndicesType = op.getStartIndices().getType();
    if (!operandType || !startIndicesType) {
      return failure();
    }

    int64_t operandRank = operandType.getRank();
    int64_t startIndicesRank = startIndicesType.getRank();
    if (operandRank == 0 || startIndicesRank == 0) {
      return failure();
    }

    int64_t indexVectorDim = dimNums.getIndexVectorDim();
    if (!isDegenerateIndexVector(indexVectorDim, startIndicesType)) {
      return failure();
    }

    auto startIndexMap = dimNums.getStartIndexMap();
    if (startIndexMap.size() != 1) {
      return failure();
    }
    int64_t operandDim = startIndexMap[0];
    if (operandDim < 0 || operandDim >= operandRank) {
      return failure();
    }

    auto sliceSizes = op.getSliceSizes();
    if (static_cast<int64_t>(sliceSizes.size()) != operandRank) {
      return failure();
    }
    auto collapsedSliceDims = dimNums.getCollapsedSliceDims();
    llvm::DenseSet<int64_t> collapsedDimsSet(collapsedSliceDims.begin(),
                                             collapsedSliceDims.end());
    if (sliceSizes[operandDim] != 1 || !collapsedDimsSet.contains(operandDim)) {
      return failure();
    }
    for (int64_t d = 0; d < operandRank; ++d) {
      if (d == operandDim) continue;
      if (sliceSizes[d] != operandType.getDimSize(d) ||
          collapsedDimsSet.contains(d)) {
        return failure();
      }
    }

    std::optional<int64_t> iotaDim =
        findIdentityIotaDim(op.getStartIndices(),
                            operandType.getDimSize(operandDim), indexVectorDim);
    if (!iotaDim.has_value()) {
      return failure();
    }

    auto offsetDims = dimNums.getOffsetDims();
    llvm::DenseSet<int64_t> offsetDimsSet(offsetDims.begin(), offsetDims.end());
    SmallVector<int64_t> startIndicesDimToResultDim(startIndicesRank, -1);
    int64_t currentResultDim = 0;
    for (int64_t d = 0; d < startIndicesRank; ++d) {
      if (d == indexVectorDim && indexVectorDim < startIndicesRank) {
        continue;
      }
      while (offsetDimsSet.contains(currentResultDim)) {
        ++currentResultDim;
      }
      startIndicesDimToResultDim[d] = currentResultDim++;
    }

    SmallVector<int64_t> broadcastDimensions(operandRank, -1);
    broadcastDimensions[operandDim] = startIndicesDimToResultDim[*iotaDim];

    int64_t offsetIdx = 0;
    for (int64_t d = 0; d < operandRank; ++d) {
      if (d == operandDim) continue;
      if (offsetIdx >= static_cast<int64_t>(offsetDims.size())) {
        return failure();
      }
      broadcastDimensions[d] = offsetDims[offsetIdx++];
    }

    auto resultType = op.getType();
    for (int64_t d = 0; d < operandRank; ++d) {
      int64_t resDim = broadcastDimensions[d];
      if (resDim < 0 || resDim >= resultType.getRank() ||
          resultType.getDimSize(resDim) != operandType.getDimSize(d)) {
        return failure();
      }
    }

    auto broadcastOp = stablehlo::BroadcastInDimOp::create(
        rewriter, op.getLoc(), op.getType(), op.getOperand(),
        rewriter.getDenseI64ArrayAttr(broadcastDimensions));
    copyAttributes(op, broadcastOp,
                   {"dimension_numbers", "slice_sizes", "indices_are_sorted"});
    rewriter.replaceOp(op, broadcastOp.getResult());
    return success();
  }
};

struct DegenerateScatterToReducePattern
    : public OpRewritePattern<stablehlo::ScatterOp> {
  using OpRewritePattern<stablehlo::ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ScatterOp op,
                                PatternRewriter& rewriter) const override {
    auto dimNums = op.getScatterDimensionNumbers();
    if (!dimNums.getInputBatchingDims().empty()) {
      return failure();
    }

    if (op.getInputs().size() != 1 || op.getUpdates().size() != 1) {
      return failure();
    }
    Value operand = op.getInputs().front();
    Value update = op.getUpdates().front();
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    auto updateType = dyn_cast<RankedTensorType>(update.getType());
    auto scatterIndicesType = op.getScatterIndices().getType();
    if (!operandType || !updateType || !scatterIndicesType) {
      return failure();
    }

    int64_t operandRank = operandType.getRank();
    int64_t updateRank = updateType.getRank();
    int64_t scatterIndicesRank = scatterIndicesType.getRank();
    if (operandRank == 0 || updateRank == 0 || scatterIndicesRank == 0) {
      return failure();
    }

    int64_t indexVectorDim = dimNums.getIndexVectorDim();
    if (!isDegenerateIndexVector(indexVectorDim, scatterIndicesType)) {
      return failure();
    }

    auto scatterDimsToOperandDims = dimNums.getScatterDimsToOperandDims();
    if (scatterDimsToOperandDims.size() != 1) {
      return failure();
    }
    int64_t operandDim = scatterDimsToOperandDims[0];
    if (operandDim < 0 || operandDim >= operandRank) {
      return failure();
    }

    auto insertedWindowDims = dimNums.getInsertedWindowDims();
    if (insertedWindowDims.size() != 1 || insertedWindowDims[0] != operandDim) {
      return failure();
    }

    std::optional<int64_t> iotaDim =
        findIdentityIotaDim(op.getScatterIndices(),
                            operandType.getDimSize(operandDim), indexVectorDim);
    if (!iotaDim.has_value()) {
      return failure();
    }

    SmallVector<int64_t> scatterIndicesDimsExceptIndexVector;
    for (int64_t d = 0; d < scatterIndicesRank; ++d) {
      if (d != indexVectorDim || indexVectorDim >= scatterIndicesRank) {
        scatterIndicesDimsExceptIndexVector.push_back(d);
      }
    }

    llvm::DenseSet<int64_t> updateWindowDimsSet(
        dimNums.getUpdateWindowDims().begin(),
        dimNums.getUpdateWindowDims().end());
    SmallVector<int64_t> updateScatterDims;
    for (int64_t d = 0; d < updateRank; ++d) {
      if (!updateWindowDimsSet.contains(d)) {
        updateScatterDims.push_back(d);
      }
    }
    if (scatterIndicesDimsExceptIndexVector.size() !=
        updateScatterDims.size()) {
      return failure();
    }

    SmallVector<int64_t> updateDimsToReduce;
    for (size_t k = 0; k < scatterIndicesDimsExceptIndexVector.size(); ++k) {
      if (scatterIndicesDimsExceptIndexVector[k] != *iotaDim) {
        updateDimsToReduce.push_back(updateScatterDims[k]);
      }
    }

    Operation* combinerOp = getCombinerOp(op.getUpdateComputation());
    if (!combinerOp) return failure();

    llvm::DenseSet<int64_t> reduceDimsSet(updateDimsToReduce.begin(),
                                          updateDimsToReduce.end());

    Value reducedUpdate;
    if (!updateDimsToReduce.empty()) {
      Type elemType = operandType.getElementType();
      RankedTensorType scalarType = RankedTensorType::get({}, elemType);
      Value initValue = getCombinerIdentityConstant(rewriter, op.getLoc(),
                                                    combinerOp, scalarType);
      if (!initValue) {
        return failure();
      }

      SmallVector<int64_t> reducedShape;
      for (int64_t d = 0; d < updateRank; ++d) {
        if (!reduceDimsSet.contains(d)) {
          reducedShape.push_back(updateType.getDimSize(d));
        }
      }
      auto reducedType = RankedTensorType::get(reducedShape, elemType);

      auto reduceOp = stablehlo::ReduceOp::create(
          rewriter, op.getLoc(), TypeRange{reducedType}, ValueRange{update},
          ValueRange{initValue},
          rewriter.getDenseI64ArrayAttr(updateDimsToReduce));
      rewriter.cloneRegionBefore(op.getUpdateComputation(), reduceOp.getBody(),
                                 reduceOp.getBody().end());
      reducedUpdate = reduceOp.getResult(0);
    } else {
      reducedUpdate = update;
    }

    SmallVector<int64_t> survivingUpdateDims;
    for (int64_t d = 0; d < updateRank; ++d) {
      if (!reduceDimsSet.contains(d)) {
        survivingUpdateDims.push_back(d);
      }
    }
    if (survivingUpdateDims.size() != static_cast<size_t>(operandRank)) {
      return failure();
    }

    SmallVector<int64_t> nonInsertedOperandDims;
    for (int64_t d = 0; d < operandRank; ++d) {
      if (d != operandDim) {
        nonInsertedOperandDims.push_back(d);
      }
    }

    auto updateWindowDims = dimNums.getUpdateWindowDims();
    if (updateWindowDims.size() != nonInsertedOperandDims.size()) {
      return failure();
    }
    for (size_t w = 0; w < updateWindowDims.size(); ++w) {
      if (updateType.getDimSize(updateWindowDims[w]) !=
          operandType.getDimSize(nonInsertedOperandDims[w])) {
        return failure();
      }
    }

    SmallVector<int64_t> operandTargetDim(operandRank, -1);
    for (int64_t i = 0; i < operandRank; ++i) {
      int64_t dUpdate = survivingUpdateDims[i];
      const auto* it = llvm::find(updateWindowDims, dUpdate);
      if (it != updateWindowDims.end()) {
        int64_t w = std::distance(updateWindowDims.begin(), it);
        if (w >= static_cast<int64_t>(nonInsertedOperandDims.size())) {
          return failure();
        }
        operandTargetDim[i] = nonInsertedOperandDims[w];
      } else {
        operandTargetDim[i] = operandDim;
      }
    }

    SmallVector<int64_t> permutation(operandRank, -1);
    for (int64_t inDim = 0; inDim < operandRank; ++inDim) {
      int64_t outDim = operandTargetDim[inDim];
      if (outDim < 0 || outDim >= operandRank || permutation[outDim] != -1) {
        return failure();
      }
      permutation[outDim] = inDim;
    }

    auto reducedUpdateType = cast<RankedTensorType>(reducedUpdate.getType());
    if (!llvm::equal(permutation, llvm::seq<int64_t>(0, operandRank))) {
      SmallVector<int64_t> transposedShape(operandRank);
      for (int64_t i = 0; i < operandRank; ++i) {
        transposedShape[i] = reducedUpdateType.getDimSize(permutation[i]);
      }
      auto transposedType =
          RankedTensorType::get(transposedShape, operandType.getElementType());
      reducedUpdate = stablehlo::TransposeOp::create(
          rewriter, op.getLoc(), transposedType, reducedUpdate,
          rewriter.getDenseI64ArrayAttr(permutation));
    }

    if (reducedUpdate.getType() != operandType) {
      return failure();
    }

    OperationState state(op.getLoc(), combinerOp->getName());
    state.addOperands({operand, reducedUpdate});
    state.addTypes(operandType);
    Operation* combinedOp = rewriter.create(state);

    copyAttributes(
        op, combinedOp,
        {"scatter_dimension_numbers", "indices_are_sorted", "unique_indices"});
    rewriter.replaceOp(op, combinedOp->getResults());
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
    patterns.add<DegenerateGatherToBroadcastPattern>(&getContext());
    patterns.add<DegenerateScatterToReducePattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
