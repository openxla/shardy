/* Copyright 2024 The Shardy Authors.

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

#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Threading.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

namespace {

bool isTranspose(stablehlo::Transpose transpose) {
  switch (transpose) {
    case stablehlo::Transpose::TRANSPOSE:
    case stablehlo::Transpose::ADJOINT:
      return true;
    case stablehlo::Transpose::NO_TRANSPOSE:
    case stablehlo::Transpose::TRANSPOSE_INVALID:
      return false;
  }
  llvm_unreachable("unknown stablehlo::Transpose");
}

// Returns a vector with `numInputs` copies of `inputDim`, followed by a single
// `indicesDim`, then `numInputs` copies of `updateDim`, which matches the order
// and quantity of scatter operands.
SmallVector<int64_t> createOperandDimsForScatter(int64_t numInputs,
                                                 int64_t inputDim,
                                                 int64_t indicesDim,
                                                 int64_t updateDim) {
  SmallVector<int64_t> operandDims;
  operandDims.reserve(2 * numInputs + 1);

  operandDims.insert(operandDims.end(), numInputs, inputDim);
  operandDims.push_back(indicesDim);
  operandDims.insert(operandDims.end(), numInputs, updateDim);
  return operandDims;
}

using GatherScatterAddFactorFn =
    std::function<void(int64_t inputDim, int64_t indicesDim, int64_t slicesDim,
                       int64_t factorSize)>;

// Adds factors for either a gather or scatter op, as they have a similar
// structure.
void addGatherScatterFactors(RankedTensorType inputType,
                             RankedTensorType slicesType, int64_t indicesRank,
                             int64_t indexVectorDim,
                             ArrayRef<int64_t> offsetDims,
                             ArrayRef<int64_t> collapsedSliceDims,
                             ArrayRef<int64_t> inputBatchingDims,
                             ArrayRef<int64_t> indicesBatchingDims,
                             GatherScatterAddFactorFn addFactorFn) {
  int64_t inputDim = 0;
  int64_t batchDimPos = 0;
  for (auto [slicesDim, slicesDimSize] :
       llvm::enumerate(slicesType.getShape())) {
    if (llvm::is_contained(offsetDims, slicesDim)) {
      // `dim` is an offset dimension.
      // We must now look up the next non-collapsed/batching input dimension
      // that corresponds to this slices offset dimension.
      while (llvm::is_contained(collapsedSliceDims, inputDim) ||
             llvm::is_contained(inputBatchingDims, inputDim)) {
        ++inputDim;
      }
      assert(inputDim < inputType.getRank());
      if (inputType.getDimSize(inputDim) == slicesDimSize) {
        // We only propagate through unsliced dimensions.
        addFactorFn(inputDim, /*indicesDim=*/kNullDim, slicesDim,
                    slicesDimSize);
      }
      ++inputDim;
    } else {
      // `dim` is a batch dimension.
      // We must now look up which one it is in `indicesType`.
      auto indicesDim =
          batchDimPos < indexVectorDim ? batchDimPos : batchDimPos + 1;
      assert(indicesDim < indicesRank);

      // If `indicesDim` is in `indicesBatchingDims`, This is an explicit batch
      // dimension across input, indices, and result. Otherwise, it is an
      // implicit batch dimension across input and result only.
      const auto* batchingDimIt = llvm::find(indicesBatchingDims, indicesDim);
      int64_t inputBatchDim =
          batchingDimIt == indicesBatchingDims.end()
              ? kNullDim
              : inputBatchingDims[batchingDimIt - indicesBatchingDims.begin()];
      addFactorFn(inputBatchDim, indicesDim, slicesDim, slicesDimSize);
      ++batchDimPos;
    }
  }
}

}  // namespace

OpShardingRuleAttr getOrCreateShardingRule(Operation* op,
                                           bool conservativePropagation) {
  if (auto shardingRule =
          op->getAttrOfType<OpShardingRuleAttr>(kShardingRuleAttr)) {
    return shardingRule;
  }
  OpShardingRuleAttr shardingRule =
      createOpShardingRule(op, conservativePropagation);
  if (shardingRule) {
    op->setAttr(kShardingRuleAttr, shardingRule);
  }
  return shardingRule;
}

OpShardingRuleAttr createOpShardingRule(Operation* op,
                                        const bool conservativePropagation) {
  return TypeSwitch<Operation*, OpShardingRuleAttr>(op)
      .Case<ShardingConstraintOp, stablehlo::AbsOp, stablehlo::AddOp,
            stablehlo::AllGatherOp, stablehlo::AllReduceOp,
            stablehlo::AllToAllOp, stablehlo::AndOp, stablehlo::Atan2Op,
            stablehlo::CbrtOp, stablehlo::CeilOp, stablehlo::ClzOp,
            stablehlo::CollectivePermuteOp, stablehlo::CompareOp,
            stablehlo::ComplexOp, stablehlo::ConvertOp, stablehlo::CosineOp,
            stablehlo::CrossReplicaSumOp, stablehlo::DivOp, stablehlo::ExpOp,
            stablehlo::Expm1Op, stablehlo::FloorOp, stablehlo::ImagOp,
            stablehlo::IsFiniteOp, stablehlo::Log1pOp, stablehlo::LogOp,
            stablehlo::LogisticOp, stablehlo::MaxOp, stablehlo::MinOp,
            stablehlo::MulOp, stablehlo::NegOp, stablehlo::NotOp,
            stablehlo::OrOp, stablehlo::PopulationCountOp, stablehlo::PowOp,
            stablehlo::RealOp, stablehlo::ReducePrecisionOp,
            stablehlo::ReduceScatterOp, stablehlo::RemOp,
            stablehlo::RoundNearestEvenOp, stablehlo::RoundOp,
            stablehlo::RsqrtOp, stablehlo::ShiftLeftOp,
            stablehlo::ShiftRightArithmeticOp, stablehlo::ShiftRightLogicalOp,
            stablehlo::SignOp, stablehlo::SineOp, stablehlo::SqrtOp,
            stablehlo::SubtractOp, stablehlo::TanhOp, stablehlo::XorOp>(
          [](Operation* pointwiseOp) {
            return OpShardingRuleBuilder::buildPointwise(pointwiseOp);
          })
      // The following case is for ops that are only pointwise for the sake of
      // propagation, but would require communication for the result to be
      // sharded like the operand along a specific dimensions. For example, if
      // the operand of an `stablehlo::SortOp` is sharded along the sorted
      // dimension, we would want to propagate that sharding to the
      // corresponding dimension of the result, even though that would require
      // communication as all elements are needed for sorting.
      .Case<stablehlo::CholeskyOp, stablehlo::ReverseOp, stablehlo::SortOp>(
          [](Operation* pointwiseOp) {
            return OpShardingRuleBuilder::buildPointwise(pointwiseOp);
          })
      //===----------------------------------------------------------------===//
      // NOTE: Please keep the order of cases alphabetical.
      //===----------------------------------------------------------------===//
      .Case<stablehlo::BitcastConvertOp>(
          [](stablehlo::BitcastConvertOp bitcastConvert) {
            ArrayRef<int64_t> inShape =
                getTensorShape(bitcastConvert.getOperand());
            ArrayRef<int64_t> outShape =
                getTensorShape(bitcastConvert.getResult());
            // Can shard on any dimension other than the additional
            // innermost dimension of the shape with the smaller element
            // bitwidth.
            ArrayRef<int64_t> shape =
                inShape.size() < outShape.size() ? inShape : outShape;
            return OpShardingRuleBuilder(bitcastConvert)
                .addPointwise(shape)
                .build();
          })
      .Case<stablehlo::BroadcastInDimOp>(
          [](stablehlo::BroadcastInDimOp broadcast) {
            OpShardingRuleBuilder builder(broadcast);

            RankedTensorType inType = broadcast.getOperand().getType();
            RankedTensorType outType = broadcast.getType();

            // We can shard any dimension of the output, and the dimension map
            // tells us how.
            SmallVector<int64_t> outDimToInDim(outType.getRank(), kNullDim);
            for (auto [inDim, outDim] :
                 llvm::enumerate(broadcast.getBroadcastDimensions())) {
              outDimToInDim[outDim] = inDim;
            }

            for (auto [outDim, outDimSize] :
                 llvm::enumerate(outType.getShape())) {
              int64_t inDim = outDimToInDim[outDim];
              if (inDim != kNullDim) {
                int64_t inDimSize = inType.getDimSize(inDim);
                if (inDimSize == 1 && outDimSize != 1) {
                  // `inDim` is expanded in-place in `outDim`.
                  builder.addFactor(inDim, kNullDim, 1);
                  inDim = kNullDim;
                } else {
                  // `inDim` and `outDim` are identical, thus they should be
                  // sharded in the same way.
                  assert(outDimSize == inDimSize);
                }
              }
              // Otherwise, `inDim == kNullDim`, which means `outDim` is
              // broadcasted.

              builder.addFactor(inDim, outDim, outDimSize);
            }
            return builder.build();
          })
      .Case<stablehlo::ClampOp>([](stablehlo::ClampOp clamp) {
        // The `min` and `max` operands may be scalars.
        return OpShardingRuleBuilder(clamp)
            .addPointwise(getTensorShape(clamp.getOperand()))
            .build();
      })
      .Case<stablehlo::ConcatenateOp>(
          [conservativePropagation](stablehlo::ConcatenateOp concat) {
            // If `conservativePropagation` is false, we propagate through
            // concat dimension, even though that would require communication.
            // TODO(tomnatan): once strided-view is supported, consider adding
            // compound factors using GCD.
            return OpShardingRuleBuilder(concat)
                .addPointwiseIf(getTensorShape(concat.getResult()),
                                [&](int64_t dim) {
                                  return !conservativePropagation ||
                                         dim != concat.getDimension();
                                })
                .build();
          })
      .Case<stablehlo::ConvolutionOp>([conservativePropagation](
                                          stablehlo::ConvolutionOp conv) {
        stablehlo::ConvDimensionNumbersAttr dimNums =
            conv.getDimensionNumbers();

        RankedTensorType lhsType = conv.getLhs().getType();
        RankedTensorType rhsType = conv.getRhs().getType();
        RankedTensorType outType = conv.getType();

        OpShardingRuleBuilder builder(
            conv,
            /*reserveNumFactors=*/outType.getRank() +
                dimNums.getInputSpatialDimensions().size() + 1);

        if (conv.getBatchGroupCount() > 1) {
          assert(conv.getFeatureGroupCount() == 1);
          // Add the number of batch groups factor.
          builder.addFactor({dimNums.getInputBatchDimension(),
                             dimNums.getKernelOutputFeatureDimension()},
                            dimNums.getOutputFeatureDimension(),
                            conv.getBatchGroupCount());
        }

        // Add the per-group batch size factor.
        builder.addFactor(
            {dimNums.getInputBatchDimension(), kNullDim},
            dimNums.getOutputBatchDimension(),
            outType.getDimSize(dimNums.getOutputBatchDimension()));

        if (!conservativePropagation) {
          // Only add a factor for spatial dimensions if we are not in
          // conservative mode.
          for (auto [lhsDim, rhsDim, outDim] :
               llvm::zip_equal(dimNums.getInputSpatialDimensions(),
                               dimNums.getKernelSpatialDimensions(),
                               dimNums.getOutputSpatialDimensions())) {
            // The input spatial dimension can be sharded along either the
            // number of windows (corresponds to the output spatial dimension)
            // or the window size (corresponds to the kernel spatial dimension),
            // so we create a factor for each. However, we need to decide which
            // factor is the major-most, which will determine whether to
            // propagate the input dimension sharding to the kernel or output
            // (or both if the sharding size is greater than the major-most
            // factor). We prefer the factor with the greater size.
            int64_t numWindows = outType.getDimSize(outDim);
            int64_t windowSize = rhsType.getDimSize(rhsDim);
            int64_t remainingLhsSize = lhsType.getDimSize(lhsDim);
            auto addSpatialFactor =
                [&, lhsDim = lhsDim, rhsDim = rhsDim, outDim = outDim](
                    int64_t factorSize, bool addRhs, bool addOut) {
                  if (factorSize = std::min(remainingLhsSize, factorSize);
                      factorSize > 1) {
                    builder.addFactor({lhsDim, addRhs ? rhsDim : kNullDim},
                                      addOut ? outDim : kNullDim, factorSize);
                    remainingLhsSize /= factorSize;
                  }
                };
            auto addNumWindowsFactor = [&]() {
              addSpatialFactor(numWindows, /*addRhs=*/false, /*addOut=*/true);
            };
            auto addWindowSizeFactor = [&]() {
              addSpatialFactor(windowSize, /*addRhs=*/true, /*addOut=*/false);
            };
            if (numWindows >= windowSize) {
              addNumWindowsFactor();
              addWindowSizeFactor();
            } else {
              addWindowSizeFactor();
              addNumWindowsFactor();
            }
          }
        }

        if (conv.getFeatureGroupCount() > 1) {
          assert(conv.getBatchGroupCount() == 1);
          // Add the number of feature groups factor.
          builder.addFactor({dimNums.getInputFeatureDimension(),
                             dimNums.getKernelOutputFeatureDimension()},
                            dimNums.getOutputFeatureDimension(),
                            conv.getFeatureGroupCount());
        }

        // Add the per-group input feature size factor.
        builder.addFactor(
            {dimNums.getInputFeatureDimension(),
             dimNums.getKernelInputFeatureDimension()},
            kNullDim,
            rhsType.getDimSize(dimNums.getKernelInputFeatureDimension()));

        // Add the output feature size factor.
        builder.addFactor(
            {kNullDim, dimNums.getKernelOutputFeatureDimension()},
            dimNums.getOutputFeatureDimension(),
            outType.getDimSize(dimNums.getOutputFeatureDimension()) /
                (conv.getBatchGroupCount() * conv.getFeatureGroupCount()));

        return builder.build();
      })
      .Case<stablehlo::CustomCallOp>([](stablehlo::CustomCallOp customCall) {
        StringRef callTargetName = customCall.getCallTargetName();
        // TODO(b/327191011): output unregistered op stats instead.
        if (callTargetName == "sdy_testonly" ||
            callTargetName == "tpu_custom_call") {
          return OpShardingRuleAttr();
        }
        if (callTargetName == "X64Combine" ||
            callTargetName == "MoveToDevice" ||
            callTargetName == "MoveToHost") {
          return OpShardingRuleBuilder::buildPointwise(customCall);
        }
        if (callTargetName == "Eigh") {
          assert(customCall.getNumOperands() == 1 &&
                 customCall.getNumResults() == 2);
          // See `jax.lax.linalg.eigh` for more information.
          //
          // All but the last two dimensions of the input are batch dimensions,
          // but we can also propagate through the non-batch dimensions as they
          // correspond between input and results, even though that would
          // require communication. The 2nd result (eigenvalues) has a single
          // non-batch dimension that corresponds to the last dimension of the
          // input and 1st result (eigenvectors).
          ArrayRef<int64_t> inShape = getTensorShape(customCall.getOperand(0));
          int64_t nonBatchDim1 = inShape.size() - 2;
          int64_t nonBatchDim2 = inShape.size() - 1;
          return OpShardingRuleBuilder(customCall)
              .addPointwise(inShape.drop_back(2))
              .addFactor(nonBatchDim1, {nonBatchDim1, kNullDim},
                         inShape[nonBatchDim1])
              .addFactor(nonBatchDim2, {nonBatchDim2, nonBatchDim1},
                         inShape[nonBatchDim2])
              .build();
        }
        if (callTargetName == "Qr") {
          assert(customCall.getNumOperands() == 1 &&
                 customCall.getNumResults() == 2);
          // See `jax.lax.linalg.qr` for more information.
          //
          // All but the last two dimensions of the input are batch dimensions,
          // but we can also propagate through the non-batch dimensions as they
          // correspond between input and 1st result, even though that would
          // require communication. The 2nd result has a single non-batch
          // dimension that has size equal to the minimum between the two
          // non-batch dimensions of the input, but we wouldn't benefit from
          // sharding it in the same way, given how QR decomposition is
          // computed.
          ArrayRef<int64_t> inShape = getTensorShape(customCall.getOperand(0));
          int64_t nonBatchDim1 = inShape.size() - 2;
          int64_t nonBatchDim2 = inShape.size() - 1;
          return OpShardingRuleBuilder(customCall)
              .addPointwise(inShape.drop_back(2))
              .addFactor(nonBatchDim1, {nonBatchDim1, kNullDim},
                         inShape[nonBatchDim1])
              .addFactor(nonBatchDim2, {nonBatchDim2, kNullDim},
                         inShape[nonBatchDim2])
              .build();
        }
        if (callTargetName == "ProductOfElementaryHouseholderReflectors") {
          // See `jax.lax.linalg.householder_product` for more information.
          //
          // All but the last two dimensions of the input are batch dimensions,
          // but we can also propagate through the non-batch dimensions as they
          // correspond between the 1st input and result, even though that
          // would require communication. The 2nd input (taus) has a single
          // non-batch dimension that doesn't correspond to any dimension in the
          // other tensors.
          ArrayRef<int64_t> inShape = getTensorShape(customCall.getOperand(0));
          int64_t nonBatchDim1 = inShape.size() - 2;
          int64_t nonBatchDim2 = inShape.size() - 1;
          return OpShardingRuleBuilder(customCall)
              .addPointwise(inShape.drop_back(2))
              .addFactor({nonBatchDim1, kNullDim}, nonBatchDim1,
                         inShape[nonBatchDim1])
              .addFactor({nonBatchDim2, kNullDim}, nonBatchDim2,
                         inShape[nonBatchDim2])
              .build();
        }
        if (callTargetName == "ApproxTopK" ||
            callTargetName == "PartialReduce") {
          assert(customCall.getNumOperands() == 4 &&
                 customCall.getNumResults() == 2);
          // See `jax.lax.approx_max_k` for more information.
          //
          // Operands: [operand, iota, init_val (scalar), init_arg (scalar)]
          // Results: [values, indices]
          ArrayRef<int64_t> inputShape =
              getTensorShape(customCall.getOperand(0));
          ArrayRef<int64_t> resultShape =
              getTensorShape(customCall.getResult(0));
          int64_t numInputs = 2, numResults = 2;
          SmallVector<int64_t> operandDims(customCall->getNumOperands(),
                                           kNullDim);
          SmallVector<int64_t> resultDims(customCall->getNumResults(),
                                          kNullDim);
          return OpShardingRuleBuilder(customCall)
              .addPointwiseIfDimSizesMatch(
                  inputShape, resultShape,
                  /*alwaysAddFactor=*/false,
                  /*onMismatchFn=*/
                  [&](int64_t dim, OpShardingRuleBuilder& builder) {
                    std::fill_n(operandDims.begin(), numInputs, dim);
                    resultDims.assign(numResults, kNullDim);
                    builder.addFactor(operandDims, resultDims, inputShape[dim]);
                    resultDims.assign(numResults, dim);
                    std::fill_n(operandDims.begin(), numInputs, kNullDim);
                    builder.addFactor(operandDims, resultDims,
                                      resultShape[dim]);
                  })
              .build();
        }
        // TODO(b/327191011): output unregistered op stats instead.
        static llvm::once_flag onceFlag;
        emitOpWarningOnce(
            onceFlag, customCall,
            llvm::formatv(
                "custom call @{0} is unknown to SDY sharding rule registry",
                customCall.getCallTargetName())
                .str());
        return OpShardingRuleAttr();
      })
      .Case<stablehlo::DotGeneralOp>([](stablehlo::DotGeneralOp dotGeneral) {
        stablehlo::DotDimensionNumbersAttr dimNums =
            dotGeneral.getDotDimensionNumbers();
        ArrayRef<int64_t> lhsBatchingDims = dimNums.getLhsBatchingDimensions();
        ArrayRef<int64_t> rhsBatchingDims = dimNums.getRhsBatchingDimensions();
        ArrayRef<int64_t> lhsContractingDims =
            dimNums.getLhsContractingDimensions();
        ArrayRef<int64_t> rhsContractingDims =
            dimNums.getRhsContractingDimensions();

        RankedTensorType lhsType = dotGeneral.getLhs().getType();
        RankedTensorType rhsType = dotGeneral.getRhs().getType();

        const int64_t lhsRank = lhsType.getRank();
        const int64_t rhsRank = rhsType.getRank();

        OpShardingRuleBuilder builder(dotGeneral,
                                      /*reserveNumFactors=*/lhsRank + rhsRank -
                                          lhsBatchingDims.size() -
                                          lhsContractingDims.size());

        int64_t outputDim = 0;

        for (auto [lhsDim, rhsDim] :
             llvm::zip_equal(lhsBatchingDims, rhsBatchingDims)) {
          builder.addFactor({lhsDim, rhsDim}, outputDim++,
                            lhsType.getDimSize(lhsDim));
        }

        for (int64_t i = 0; i < lhsRank; i++) {
          if (!llvm::is_contained(lhsContractingDims, i) &&
              !llvm::is_contained(lhsBatchingDims, i)) {
            builder.addFactor({i, kNullDim}, outputDim++,
                              lhsType.getDimSize(i));
          }
        }
        for (int64_t i = 0; i < rhsRank; i++) {
          if (!llvm::is_contained(rhsContractingDims, i) &&
              !llvm::is_contained(rhsBatchingDims, i)) {
            builder.addFactor({kNullDim, i}, outputDim++,
                              rhsType.getDimSize(i));
          }
        }
        for (auto [lhsDim, rhsDim] :
             llvm::zip_equal(lhsContractingDims, rhsContractingDims)) {
          builder.addFactor({lhsDim, rhsDim}, kNullDim,
                            lhsType.getDimSize(lhsDim));
        }

        return builder.build();
      })
      .Case<stablehlo::DotOp>([](stablehlo::DotOp dot) {
        OpShardingRuleBuilder builder(dot);
        RankedTensorType lhsType = dot.getLhs().getType();
        RankedTensorType rhsType = dot.getRhs().getType();

        bool isLhsMatrix = lhsType.getRank() == 2;
        bool isRhsMatrix = rhsType.getRank() == 2;

        // LHS non-contracting.
        if (isLhsMatrix) {
          builder.addFactor({0, kNullDim}, 0, lhsType.getDimSize(0));
        }

        // RHS non-contracting.
        if (isRhsMatrix) {
          builder.addFactor({kNullDim, 1}, isLhsMatrix ? 1 : 0,
                            rhsType.getDimSize(1));
        }

        // Contracting dimension.
        builder.addFactor({isLhsMatrix ? 1 : 0, 0}, kNullDim,
                          rhsType.getDimSize(0));

        return builder.build();
      })
      .Case<stablehlo::DynamicSliceOp>(
          [](stablehlo::DynamicSliceOp dynamicSlice) {
            return OpShardingRuleBuilder(dynamicSlice)
                .addPointwiseIfDimSizesMatch(
                    getTensorShape(dynamicSlice.getOperand()),
                    getTensorShape(dynamicSlice.getResult()))
                .build();
          })
      .Case<stablehlo::DynamicUpdateSliceOp>(
          [](stablehlo::DynamicUpdateSliceOp dynamicUpdateSlice) {
            ArrayRef<int64_t> operandShape =
                getTensorShape(dynamicUpdateSlice.getOperand());
            ArrayRef<int64_t> updateShape =
                getTensorShape(dynamicUpdateSlice.getUpdate());
            SmallVector<int64_t> operandDims(
                dynamicUpdateSlice->getNumOperands(), kNullDim);
            return OpShardingRuleBuilder(dynamicUpdateSlice)
                .addPointwiseIfDimSizesMatch(
                    operandShape, updateShape,
                    /*alwaysAddFactor=*/false,
                    /*onMismatchFn=*/
                    [&](int64_t dim, OpShardingRuleBuilder& builder) {
                      operandDims[0] = dim;
                      builder.addFactor(operandDims, dim, operandShape[dim]);
                    })
                .build();
          })
      .Case<stablehlo::FftOp>([](stablehlo::FftOp fft) {
        ArrayRef<int64_t> inShape = getTensorShape(fft.getOperand());
        ArrayRef<int64_t> outShape = getTensorShape(fft.getResult());
        // The `FftOp` computes the Fourier transform across the trailing
        // `fft.getFftLength().size()` dimensions of its operand, which means
        // reductions are performed across those dimensions, the other
        // dimensions are batch dimensions. We still want to propagate through
        // the trailing dimensions even though that would require communication.
        bool isLastDimTruncated = inShape.back() != outShape.back();
        return OpShardingRuleBuilder(fft)
            .addPointwise(inShape.drop_back(isLastDimTruncated ? 1 : 0))
            .build();
      })
      .Case<stablehlo::GatherOp>([](stablehlo::GatherOp gather) {
        OpShardingRuleBuilder builder(gather);

        RankedTensorType inputType = gather.getOperand().getType();
        RankedTensorType slicesType = gather.getType();
        int64_t indicesRank = getTensorRank(gather.getStartIndices());
        stablehlo::GatherDimensionNumbersAttr dimNums =
            gather.getDimensionNumbers();

        addGatherScatterFactors(
            inputType, slicesType, indicesRank, dimNums.getIndexVectorDim(),
            dimNums.getOffsetDims(), dimNums.getCollapsedSliceDims(),
            dimNums.getOperandBatchingDims(),
            dimNums.getStartIndicesBatchingDims(),
            [&](int64_t inputDim, int64_t indicesDim, int64_t slicesDim,
                int64_t factorSize) {
              builder.addFactor({inputDim, indicesDim}, slicesDim, factorSize);
            });

        return builder.build();
      })
      .Case<stablehlo::PadOp>([conservativePropagation](stablehlo::PadOp pad) {
        // If `conservativePropagation` is false, we propagate through padded
        // dimensions, even though that would require communication.
        return OpShardingRuleBuilder(pad)
            .addPointwiseIfDimSizesMatch(
                getTensorShape(pad.getOperand()),
                getTensorShape(pad.getResult()),
                /*alwaysAddFactor=*/!conservativePropagation)
            .build();
      })
      .Case<stablehlo::ReduceOp>([](stablehlo::ReduceOp reduce) {
        OpShardingRuleBuilder builder(reduce);
        // Since all inputs and results have compatible shapes, we can look at
        // the first.
        ArrayRef<int64_t> inputShape =
            getTensorShape(reduce.getInputs().front());
        auto resultType =
            cast<RankedTensorType>(reduce.getResultTypes().front());
        // NOTE: resultType is only used by vanilla C++ asserts, so during opt
        // builds it will be marked as unused.
        // TODO(bartchr): define our own asserts that are kept during opt
        // builds.
        (void)resultType;

        ArrayRef<int64_t> dimensions = reduce.getDimensions();
        size_t numInputs = reduce.getInputs().size();

        int64_t outDim = 0;
        SmallVector<int64_t> operandDims(reduce->getNumOperands(), kNullDim);
        SmallVector<int64_t> resultDims(numInputs, kNullDim);
        for (auto [inDim, dimSize] : llvm::enumerate(inputShape)) {
          // The first `numInputs` operands are the inputs and the next
          // `numInputs` operands are the init values.
          std::fill_n(operandDims.begin(), numInputs, inDim);

          if (llvm::is_contained(dimensions, inDim)) {
            // Dimension that is being reduced. Can have a mapping for the
            // inputs.
            resultDims.assign(numInputs, kNullDim);
          } else {
            // Not a reduced dimension. So have a mapping b/w the operand and
            // result.
            assert(resultType.getDimSize(outDim) == dimSize);
            resultDims.assign(numInputs, outDim++);
          }
          builder.addFactor(operandDims, resultDims, dimSize);
        }
        assert(outDim == resultType.getRank());
        return builder.build();
      })
      .Case<stablehlo::ReduceWindowOp>(
          [conservativePropagation](stablehlo::ReduceWindowOp reduceWindow) {
            // Since all results have compatible shapes, we can look at the
            // first. The size of each result dimension is the number of input
            // windows reduced along that dimension. The corresponding input
            // dimension size can be sharded along the number of windows,
            // therefore we add a factor with that size.
            //
            // In conservative mode, we only add a factor if the input and
            // output dimension sizes are equal.
            // TODO(tomnatan): should the reduced factor be compound?
            return OpShardingRuleBuilder(reduceWindow)
                .addPointwiseIfDimSizesMatch(
                    getTensorShape(reduceWindow.getResult(0)),
                    getTensorShape(reduceWindow.getInputs().front()),
                    /*alwaysAddFactor=*/!conservativePropagation)
                .build();
          })
      .Case<stablehlo::ReshapeOp>([](stablehlo::ReshapeOp reshape) {
        RankedTensorType inType = reshape.getOperand().getType();
        RankedTensorType outType = reshape.getType();

        OpShardingRuleBuilder builder(reshape);

        if (inType.getNumElements() == 0) {
          // This reshape has a dimension with size 0, in which case we return
          // an empty rule as the algorithm can't handle it. There is no point
          // in propagating through an op with 0 elements.
          return OpShardingRuleAttr();
        }

        int64_t inRank = inType.getRank();
        int64_t outRank = outType.getRank();

        int64_t inDim = 0;
        int64_t outDim = 0;

        int64_t prodDimSizesIn = 1;
        int64_t prodDimSizesOut = 1;

        int64_t prodFactorsIn = 1;
        int64_t prodFactorsOut = 1;

        while (inDim < inRank || outDim < outRank) {
          if (inDim < inRank && inType.getDimSize(inDim) == 1) {
            builder.addFactor(inDim++, kNullDim, 1);
            continue;
          }
          if (outDim < outRank && outType.getDimSize(outDim) == 1) {
            builder.addFactor(kNullDim, outDim++, 1);
            continue;
          }

          assert(inDim < inRank && outDim < outRank);
          if (prodDimSizesIn == prodFactorsIn) {
            prodDimSizesIn *= inType.getDimSize(inDim);
          }
          if (prodDimSizesOut == prodFactorsOut) {
            prodDimSizesOut *= outType.getDimSize(outDim);
          }

          int64_t nextInFactor = prodDimSizesIn / prodFactorsIn;
          int64_t nextOutFactor = prodDimSizesOut / prodFactorsOut;
          assert(nextInFactor > 1 && nextOutFactor > 1);

          int64_t nextFactorGcd = std::gcd(nextInFactor, nextOutFactor);

          if (nextFactorGcd > 1 && prodFactorsIn == prodFactorsOut) {
            // The next in and out factors have a GCD greater than 1, and the
            // current in and out accumulated factors match, therefore we can
            // add the GCD as a common factor (same factor index for both input
            // and output).
            builder.addFactor(inDim, outDim, nextFactorGcd);
            prodFactorsIn *= nextFactorGcd;
            prodFactorsOut *= nextFactorGcd;
          } else {
            // Otherwise, we add the next factors as unique factors (different
            // factor indices for the input and output), and we wouldn't be able
            // to add a common factor until the in and out accumulated factors
            // match again (they might have already diverged).
            nextInFactor = nextInFactor > nextFactorGcd
                               ? nextInFactor / nextFactorGcd
                               : nextFactorGcd;
            builder.addFactor(inDim, kNullDim, nextInFactor);
            prodFactorsIn *= nextInFactor;

            nextOutFactor = nextOutFactor > nextFactorGcd
                                ? nextOutFactor / nextFactorGcd
                                : nextFactorGcd;
            builder.addFactor(kNullDim, outDim, nextOutFactor);
            prodFactorsOut *= nextOutFactor;
          }

          if (prodDimSizesIn == prodFactorsIn) {
            inDim++;
          }
          if (prodDimSizesOut == prodFactorsOut) {
            outDim++;
          }
        }

        return builder.build();
      })
      .Case<stablehlo::ScatterOp>([](stablehlo::ScatterOp scatter) {
        OpShardingRuleBuilder builder(scatter);

        // Since all inputs and results have compatible shapes, we can look at
        // the first.
        auto inputType =
            cast<RankedTensorType>(scatter.getInputs()[0].getType());
        auto slicesType =
            cast<RankedTensorType>(scatter.getUpdates()[0].getType());
        int64_t indicesRank = getTensorRank(scatter.getScatterIndices());
        size_t numInputs = scatter.getInputs().size();
        stablehlo::ScatterDimensionNumbersAttr dimNums =
            scatter.getScatterDimensionNumbers();

        addGatherScatterFactors(
            inputType, slicesType, indicesRank, dimNums.getIndexVectorDim(),
            /*offsetDims=*/dimNums.getUpdateWindowDims(),
            /*collapsedSliceDims=*/dimNums.getInsertedWindowDims(),
            dimNums.getInputBatchingDims(),
            dimNums.getScatterIndicesBatchingDims(),
            [&](int64_t inputDim, int64_t indicesDim, int64_t slicesDim,
                int64_t factorSize) {
              builder.addFactor(
                  createOperandDimsForScatter(numInputs, inputDim, indicesDim,
                                              /*updateDim=*/slicesDim),
                  /*resultDims=*/SmallVector<int64_t>(numInputs, inputDim),
                  factorSize);
            });
        return builder.build();
      })
      .Case<stablehlo::SelectAndScatterOp>(
          [conservativePropagation](
              stablehlo::SelectAndScatterOp selectAndScatter) {
            // The size of each source dimension is the number of input windows
            // reduced along that dimension. The corresponding input dimension
            // size can be sharded along the number of windows, therefore we add
            // a factor with that size.
            //
            // In conservative mode, we only add a factor if the input and
            // source dimension sizes are equal.
            // TODO(tomnatan): should the reduced factor be compound?
            return OpShardingRuleBuilder(selectAndScatter)
                .addPointwiseIfDimSizesMatch(
                    getTensorShape(selectAndScatter.getSource()),
                    getTensorShape(selectAndScatter.getOperand()),
                    /*alwaysAddFactor=*/!conservativePropagation)
                .build();
          })
      .Case<stablehlo::SelectOp>([](stablehlo::SelectOp select) {
        // Case 1: `pred` is a scalar in which case it is broadcasted and must
        //   therefore not be partitioned. The other two inputs behave like
        //   pointwise ops.
        // Case 2: all three inputs have the same shape and behave like
        //   pointwise ops.
        return OpShardingRuleBuilder(select)
            .addPointwise(getTensorShape(select.getResult()))
            .build();
      })
      .Case<stablehlo::SliceOp>(
          [conservativePropagation](stablehlo::SliceOp slice) {
            // If `conservativePropagation` is false, we propagate through
            // sliced dimensions, even though that would require communication.
            //
            // This is different from `DynamicSliceOp`, where we don't
            // propagate through sliced dimensions regardless of
            // `conservativePropagation`, and the reason is that for `SliceOp`
            // the start indices are static, so we know how to shift the data
            // to keep the sliced dimension sharded.
            return OpShardingRuleBuilder(slice)
                .addPointwiseIfDimSizesMatch(
                    getTensorShape(slice.getOperand()),
                    getTensorShape(slice.getResult()),
                    /*alwaysAddFactor=*/!conservativePropagation)
                .build();
          })
      .Case<stablehlo::TransposeOp>([](stablehlo::TransposeOp transpose) {
        OpShardingRuleBuilder builder(transpose);
        RankedTensorType inType = transpose.getOperand().getType();
        for (auto [outDim, inDim] :
             llvm::enumerate(transpose.getPermutation())) {
          builder.addFactor(inDim, outDim, inType.getDimSize(inDim));
        }
        return builder.build();
      })
      .Case<stablehlo::TriangularSolveOp>(
          [](stablehlo::TriangularSolveOp triangularSolve) {
            OpShardingRuleBuilder builder(triangularSolve);
            ArrayRef<int64_t> aShape = getTensorShape(triangularSolve.getA());
            ArrayRef<int64_t> bShape = getTensorShape(triangularSolve.getB());
            // All dimensions except the last two are batch dimensions.
            builder.addPointwise(aShape.drop_back(2));

            int64_t dim1 = aShape.size() - 2;
            int64_t dim2 = aShape.size() - 1;
            bool isATransposed = isTranspose(triangularSolve.getTransposeA());
            if (triangularSolve.getLeftSide()) {
              // The equation is `op(a) @ result = b`, where op(a) is determined
              // by `isATransposed`.
              int64_t aNonContractingDim = isATransposed ? dim2 : dim1;
              int64_t aContractingDim = isATransposed ? dim1 : dim2;
              builder
                  // A non-contracting dim
                  .addFactor({aNonContractingDim, dim1}, kNullDim,
                             aShape[aNonContractingDim])
                  // Result non-contracting dim
                  .addFactor({kNullDim, dim2}, dim2, bShape[dim2])
                  // Contracting dim
                  .addFactor({aContractingDim, kNullDim}, dim1,
                             aShape[aContractingDim]);
            } else {
              // The equation is `result @ op(a) = b`, where op(a) is determined
              // by `isATransposed`.
              int64_t aNonContractingDim = isATransposed ? dim1 : dim2;
              int64_t aContractingDim = isATransposed ? dim2 : dim1;
              builder
                  // Result non-contracting dim
                  .addFactor({kNullDim, dim1}, dim1, bShape[dim1])
                  // A non-contracting dim
                  .addFactor({aNonContractingDim, dim2}, kNullDim,
                             aShape[aNonContractingDim])
                  // Contracting dim
                  .addFactor({aContractingDim, kNullDim}, dim2,
                             aShape[aContractingDim]);
            }
            return builder.build();
          })
      // Ops that shouldn't be registered as they are either handled separately
      // (e.g., `stablehlo::WhileOp`) or don't require any propagation
      // (`stablehlo::ConstantOp`).
      // TODO(b/327191011): output unregistered op stats instead.
      .Case<ModuleOp, func::FuncOp, func::ReturnOp, ConstantOp, DataFlowEdgeOp,
            ManualComputationOp, MeshOp, PropagationBarrierOp, ReturnOp,
            ShardableRegionOpInterface, stablehlo::CaseOp,
            stablehlo::ConstantOp, stablehlo::CreateTokenOp,
            stablehlo::GetTupleElementOp, stablehlo::IotaOp,
            stablehlo::OutfeedOp, stablehlo::OptimizationBarrierOp,
            stablehlo::PartitionIdOp, stablehlo::ReturnOp,
            stablehlo::RngBitGeneratorOp, stablehlo::WhileOp>(
          [](Operation*) { return OpShardingRuleAttr(); })
      .Default([](Operation* op) {
        static llvm::once_flag onceFlag;
        emitOpWarningOnce(
            onceFlag, op,
            llvm::formatv("op '{0}' is unknown to SDY sharding rule registry",
                          op->getName())
                .str());
        return OpShardingRuleAttr();
      });
}

}  // namespace sdy
}  // namespace mlir
