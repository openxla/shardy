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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/transforms/export/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_PADFORDIVISIBILITYPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

// Computes the padded type for a given type with sharding.
Type getPaddedType(Type type, TensorShardingAttr sharding,
                   const SymbolTable& symbolTable) {
  auto rankedType = dyn_cast<RankedTensorType>(type);
  if (!rankedType || isFullyReplicated(sharding)) {
    return type;
  }
  MeshAttr mesh = sharding.getMesh(symbolTable);
  if (!mesh) {
    return type;
  }

  SmallVector<int64_t> paddedShape = llvm::to_vector(rankedType.getShape());
  bool changed = false;
  for (auto [dim, dimSharding] : llvm::enumerate(sharding.getDimShardings())) {
    int64_t dimSize = paddedShape[dim];
    if (dimSize == ShapedType::kDynamic) continue;

    int64_t shardCount = dimSharding.getShardedSize(mesh);
    if (shardCount > 1 && dimSize % shardCount != 0) {
      paddedShape[dim] = ((dimSize + shardCount - 1) / shardCount) * shardCount;
      changed = true;
    }
  }

  if (!changed) {
    return type;
  }
  return RankedTensorType::get(paddedShape, rankedType.getElementType());
}

class PaddedTypeConverter : public TypeConverter {
 public:
  PaddedTypeConverter(const SymbolTable& symbolTable)
      : symbolTable(symbolTable) {
    addConversion([](Type type) { return type; });

    addConversion([&](Value value) -> std::optional<Type> {
      if (auto type = dyn_cast<RankedTensorType>(value.getType())) {
        return getPaddedType(type, getSharding(value), symbolTable);
      }
      return std::nullopt;
    });

    auto materialize = [](OpBuilder& b, Type t, ValueRange inputs,
                          Location loc) -> Value {
      return UnrealizedConversionCastOp::create(b, loc, t, inputs).getResult(0);
    };
    addSourceMaterialization(materialize);
    addTargetMaterialization(materialize);
  }

  const SymbolTable& getSymbolTable() const { return symbolTable; }

 private:
  const SymbolTable& symbolTable;
};

// Known padding value kinds for generated padding values.
enum class PaddingValueKind { kZero, kOne };

// Returns true if the operation has custom padding handling implemented in
// this file and should be excluded from GenericOpPattern.
bool hasCustomPadHandling(Operation* op) {
  return isa<stablehlo::SliceOp, stablehlo::DotGeneralOp>(op);
}

class PaddingCache {
 public:
  // Registers the padding kind for a value. The conversion pattern
  // rewriter processes operations in topological order, so each value should
  // only be registered once.
  void setPadding(Value value, PaddingValueKind kind) {
    if (!cache.insert({value, kind}).second) {
      SDY_CHECK(false) << "Padding value already set for "
                       << valueToString(value);
    }
  }

  std::optional<PaddingValueKind> getPadding(Value value) const {
    auto it = cache.find(value);
    if (it != cache.end()) {
      return it->second;
    }
    return std::nullopt;
  }

 private:
  DenseMap<Value, PaddingValueKind> cache;
};

// Returns a constant for the given PaddingValueKind.
Value createConstant(OpBuilder& b, Location loc, Type elementType,
                     PaddingValueKind kind) {
  auto type = RankedTensorType::get({}, elementType);
  switch (kind) {
    case PaddingValueKind::kZero:
      return stablehlo::ConstantOp::create(b, loc, b.getZeroAttr(type));
    case PaddingValueKind::kOne:
      if (auto floatType = dyn_cast<FloatType>(elementType)) {
        return stablehlo::ConstantOp::create(
            b, loc,
            DenseElementsAttr::get(type, b.getFloatAttr(elementType, 1.0)));
      }
      return stablehlo::ConstantOp::create(
          b, loc,
          DenseElementsAttr::get(type, b.getIntegerAttr(elementType, 1)));
  }
  llvm_unreachable("invalid PaddingValueKind");
}

// Creates a padded value for 'value' with 'paddedType' and 'paddingKind',
// and adds the padded value to the PaddingCache.
Value createPaddedValue(RankedTensorType paddedType, Value value,
                        PaddingValueKind paddingKind,
                        const SymbolTable& symbolTable,
                        ConversionPatternRewriter& rewriter,
                        PaddingCache& cache) {
  Location loc = value.getLoc();
  auto origType = cast<RankedTensorType>(value.getType());
  SDY_CHECK(paddedType != origType);

  Value padding =
      createConstant(rewriter, loc, paddedType.getElementType(), paddingKind);

  SmallVector<int64_t> edgePaddingHigh;
  for (int i = 0; i < origType.getRank(); ++i) {
    edgePaddingHigh.push_back(paddedType.getDimSize(i) -
                              origType.getDimSize(i));
  }

  Value padOp = stablehlo::PadOp::create(
      rewriter, loc, paddedType, value, padding,
      rewriter.getDenseI64ArrayAttr(
          SmallVector<int64_t>(origType.getRank(), 0)),
      rewriter.getDenseI64ArrayAttr(edgePaddingHigh),
      rewriter.getDenseI64ArrayAttr(
          SmallVector<int64_t>(origType.getRank(), 0)));
  if (auto sharding = getSharding(value)) {
    setSharding(padOp, sharding);
  }
  cache.setPadding(padOp, paddingKind);
  return padOp;
}

// Returns 'inputVal' if the value is not padded or the value already has
// 'requiredKind' as PaddingValueKind. Otherwise, uses compare-and-select to
// produce a new padded value from inputVal with the requiredKind padding and
// returns the new value.
//
// We ensure all dimensions that require padding are padded with requireKind
// unless dimsToEnforce is provided, in which case only the specified
// dimensions are padded.
Value ensurePadding(
    Value inputVal, RankedTensorType origType, PaddingValueKind requiredKind,
    OpBuilder& b, Location loc, PaddingCache& cache,
    std::optional<ArrayRef<int64_t>> dimsToEnforce = std::nullopt) {
  // Return early if no padding is applied or the cached padding already
  // matches.
  auto paddedType = cast<RankedTensorType>(inputVal.getType());
  if (origType == paddedType) {
    return inputVal;
  }
  std::optional<PaddingValueKind> currentKind = cache.getPadding(inputVal);
  if (currentKind && *currentKind == requiredKind) {
    return inputVal;
  }

  // Build a mask that is `true` for the original (unpadded) data region.
  // An element is in the original region if its index along each padded
  // dimension is less than the original unpadded size (index < original_size).
  Value validDataMask;
  for (auto [dim, origSize] : llvm::enumerate(origType.getShape())) {
    if (origSize == paddedType.getDimSize(dim) ||
        (dimsToEnforce && !llvm::is_contained(*dimsToEnforce, dim))) {
      continue;
    }
    auto iotaType =
        RankedTensorType::get(paddedType.getShape(), b.getI32Type());
    Value iota = stablehlo::IotaOp::create(b, loc, iotaType, dim);
    Value limit = stablehlo::ConstantOp::create(
        b, loc,
        DenseElementsAttr::get(RankedTensorType::get({}, b.getI32Type()),
                               b.getI32IntegerAttr(origSize)));
    Value broadcastLimit = stablehlo::BroadcastInDimOp::create(
        b, loc, iotaType, limit, b.getDenseI64ArrayAttr({}));
    Value mask = stablehlo::CompareOp::create(
        b, loc, iota, broadcastLimit, stablehlo::ComparisonDirection::LT);
    validDataMask = validDataMask
                        ? stablehlo::AndOp::create(b, loc, validDataMask, mask)
                        : mask;
  }

  if (!validDataMask) {
    return inputVal;
  }

  // Create the constant with the new padding value and broadcast it to the
  // same shape as 'inputVal'.
  Value newPaddingScalar =
      createConstant(b, loc, paddedType.getElementType(), requiredKind);
  Value newPaddingValue = stablehlo::BroadcastInDimOp::create(
      b, loc, paddedType, newPaddingScalar, b.getDenseI64ArrayAttr({}));

  // Keep the original data from 'inputVal' (where mask is true), and replace
  // the padded region with 'newPaddingValue' (where mask is false).
  Value select = stablehlo::SelectOp::create(b, loc, validDataMask, inputVal,
                                             newPaddingValue);
  if (!dimsToEnforce) {
    cache.setPadding(select, requiredKind);
  }
  return select;
}

// Converts op to its local version by replacing its operands with the already
// converted operands.
LogicalResult padGenericOp(Operation* op, ValueRange operands,
                           ConversionPatternRewriter& rewriter,
                           const PaddedTypeConverter* typeConverter) {
  SmallVector<Value> shardableOperands;
  for (Value operand : operands) {
    shardableOperands.push_back(getShardableValue(operand));
  }

  // Compute padded shapes for results.
  SmallVector<Type> inferredTypes;
  if (auto inferTypeOp = dyn_cast<InferTypeOpInterface>(op)) {
    if (failed(inferTypeOp.inferReturnTypes(
            op->getContext(), op->getLoc(), shardableOperands,
            op->getAttrDictionary(), op->getPropertiesStorage(),
            op->getRegions(), inferredTypes))) {
      inferredTypes.clear();
    }
  }

  SmallVector<Type> newResultTypes;
  for (int i = 0; i < op->getNumResults(); ++i) {
    Value result = op->getResult(i);
    Type paddedType = getPaddedType(result.getType(), getSharding(result),
                                    typeConverter->getSymbolTable());
    if (inferredTypes.empty()) {
      newResultTypes.push_back(paddedType);
    } else {
      if (auto inferredShaped = dyn_cast<RankedTensorType>(inferredTypes[i])) {
        auto paddedShaped = cast<RankedTensorType>(paddedType);
        SmallVector<int64_t> reconciledShape;
        for (int d = 0; d < inferredShaped.getRank(); ++d) {
          reconciledShape.push_back(std::max(inferredShaped.getDimSize(d),
                                             paddedShaped.getDimSize(d)));
        }
        newResultTypes.push_back(RankedTensorType::get(
            reconciledShape, inferredShaped.getElementType()));
      } else {
        newResultTypes.push_back(paddedType);
      }
    }
  }
  SDY_CHECK(newResultTypes.size() == op->getNumResults());
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(shardableOperands);
  state.addTypes(newResultTypes);
  state.addAttributes(op->getAttrs());
  state.addSuccessors(op->getSuccessors());
  for (int i = 0; i < op->getNumRegions(); ++i) {
    state.addRegion();
  }

  Operation* newOp = rewriter.create(state);

  for (auto [oldRegion, newRegion] :
       llvm::zip(op->getRegions(), newOp->getRegions())) {
    rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.end());
  }

  rewriter.replaceOp(op, newOp->getResults());
  return success();
}

// Pattern for ops that just need operands updates and result type updates to
// match padded shape.
class GenericOpPattern : public ConversionPattern {
 public:
  GenericOpPattern(TypeConverter& converter, MLIRContext* ctx)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    Dialect* dialect = op->getDialect();
    if ((dialect && dialect->getNamespace() != "stablehlo" &&
         !isa<sdy::ReturnOp>(op)) ||
        hasCustomPadHandling(op)) {
      return failure();
    }
    return padGenericOp(op, operands, rewriter,
                        static_cast<const PaddedTypeConverter*>(typeConverter));
  }
};

class FuncOpPattern : public OpConversionPattern<func::FuncOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const PaddedTypeConverter*>(getTypeConverter());
    const SymbolTable& symbolTable = converter->getSymbolTable();

    for (auto [index, arg] : llvm::enumerate(op.getArguments())) {
      if (getPaddedType(arg.getType(), getSharding(arg), symbolTable) !=
          arg.getType()) {
        return op.emitOpError()
               << "argument #" << index << " has a non-divisible sharding. "
               << "Shardy expects function IO to be divisible.";
      }
    }

    for (int i = 0; i < op.getNumResults(); ++i) {
      Type resultType = op.getResultTypes()[i];
      if (getPaddedType(resultType, getFuncResultSharding(op, i),
                        symbolTable) != resultType) {
        return op.emitOpError()
               << "result #" << i << " has a non-divisible sharding. "
               << "Shardy expects function IO to be divisible.";
      }
    }

    return failure();
  }
};

class AllGatherOpPattern : public OpConversionPattern<sdy::AllGatherOp> {
 public:
  AllGatherOpPattern(TypeConverter& converter, MLIRContext* ctx,
                     PaddingCache& cache)
      : OpConversionPattern(converter, ctx), cache(cache) {}

  LogicalResult matchAndRewrite(
      sdy::AllGatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const PaddedTypeConverter*>(getTypeConverter());
    const SymbolTable& symbolTable = converter->getSymbolTable();

    SmallVector<Value> shardableOperands;
    for (Value operand : adaptor.getOperands()) {
      shardableOperands.push_back(getShardableValue(operand));
    }

    SmallVector<Type> inferredTypes;
    if (failed(op.inferReturnTypes(op.getContext(), op.getLoc(),
                                   shardableOperands, op->getAttrDictionary(),
                                   op->getPropertiesStorage(), op->getRegions(),
                                   inferredTypes))) {
      return failure();
    }

    SmallVector<Type> newResultTypes;
    for (int i = 0; i < op->getNumResults(); ++i) {
      Value result = op->getResult(i);
      Type paddedType =
          getPaddedType(result.getType(), getSharding(result), symbolTable);
      if (inferredTypes.empty()) {
        newResultTypes.push_back(paddedType);
      } else {
        if (auto inferredShaped =
                dyn_cast<RankedTensorType>(inferredTypes[i])) {
          auto paddedShaped = cast<RankedTensorType>(paddedType);
          SmallVector<int64_t> reconciledShape;
          for (int d = 0; d < inferredShaped.getRank(); ++d) {
            reconciledShape.push_back(std::max(inferredShaped.getDimSize(d),
                                               paddedShaped.getDimSize(d)));
          }
          newResultTypes.push_back(RankedTensorType::get(
              reconciledShape, inferredShaped.getElementType()));
        } else {
          newResultTypes.push_back(paddedType);
        }
      }
    }

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(shardableOperands);
    state.addTypes(newResultTypes);
    state.addAttributes(op->getAttrs());
    Operation* newOp = rewriter.create(state);

    std::optional<PaddingValueKind> paddingKind;
    Value input = adaptor.getOperands()[0];
    paddingKind = cache.getPadding(input);
    if (paddingKind) {
      for (Value res : newOp->getResults()) {
        cache.setPadding(res, *paddingKind);
      }
    }

    SmallVector<Value> replacements;
    ::llvm::ArrayRef<AxisRefListAttr> gatheringAxes = op.getGatheringAxes();

    for (int i = 0; i < op->getNumResults(); ++i) {
      Value res = newOp->getResult(i);
      Type origType = op->getResult(i).getType();
      auto origRanked = dyn_cast<RankedTensorType>(origType);
      auto newRanked = dyn_cast<RankedTensorType>(res.getType());

      if (origRanked && newRanked) {
        bool needsSlice = false;
        SmallVector<int64_t> limitIndices;
        for (int d = 0; d < origRanked.getRank(); ++d) {
          if (!gatheringAxes[d].empty() &&
              newRanked.getDimSize(d) > origRanked.getDimSize(d)) {
            limitIndices.push_back(origRanked.getDimSize(d));
            needsSlice = true;
          } else {
            limitIndices.push_back(newRanked.getDimSize(d));
          }
        }

        if (needsSlice) {
          SmallVector<int64_t> starts(origRanked.getRank(), 0);
          SmallVector<int64_t> strides(origRanked.getRank(), 1);
          auto sliceOp = stablehlo::SliceOp::create(
              rewriter, op.getLoc(),
              RankedTensorType::get(limitIndices, origRanked.getElementType()),
              res, rewriter.getDenseI64ArrayAttr(starts),
              rewriter.getDenseI64ArrayAttr(limitIndices),
              rewriter.getDenseI64ArrayAttr(strides));
          setSharding(sliceOp.getResult(), getSharding(res));
          Value sliceRes = sliceOp.getResult();
          if (paddingKind) {
            cache.setPadding(sliceRes, *paddingKind);
          }
          replacements.push_back(sliceRes);
          continue;
        }
      }
      replacements.push_back(res);
    }

    rewriter.replaceOp(op, replacements);
    return success();
  }

 private:
  PaddingCache& cache;
};

// Pattern for sdy.all_slice. Slices the input along sharded dimensions. If a
// sharded dimension is not divisible by its shard count, the input is padded.
// The padding kind (e.g. kZero) is propagated to the output and registered in
// the PaddingCache, allowing downstream operations to reuse the padding.
class AllSliceOpPattern : public OpConversionPattern<sdy::AllSliceOp> {
 public:
  AllSliceOpPattern(TypeConverter& converter, MLIRContext* ctx,
                    PaddingCache& cache)
      : OpConversionPattern(converter, ctx), cache(cache) {}

  LogicalResult matchAndRewrite(
      sdy::AllSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const PaddedTypeConverter*>(getTypeConverter());
    const SymbolTable& symbolTable = converter->getSymbolTable();

    Value input = adaptor.getOperands()[0];
    auto rankedInputType = dyn_cast<RankedTensorType>(input.getType());
    if (!rankedInputType) {
      return failure();
    }

    TensorShardingAttr outSharding = op.getOutSharding();
    Type paddedInputType =
        getPaddedType(rankedInputType, outSharding, symbolTable);

    std::optional<PaddingValueKind> paddingKind = cache.getPadding(input);
    Value newInput = input;

    if (paddedInputType != rankedInputType) {
      paddingKind = PaddingValueKind::kZero;
      newInput =
          createPaddedValue(cast<RankedTensorType>(paddedInputType), input,
                            *paddingKind, symbolTable, rewriter, cache);
    }

    OperationState state(op->getLoc(), op->getName());
    state.addOperands({newInput});
    // sdy.all_slice has the SameOperandsAndResultType trait, so the output
    // will always have the same padded shape as the input.
    state.addTypes({paddedInputType});
    state.addAttributes(op->getAttrs());
    Operation* newOp = rewriter.create(state);

    if (paddingKind) {
      cache.setPadding(newOp->getResult(0), *paddingKind);
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

 private:
  PaddingCache& cache;
};

class StablehloSliceOpPattern : public OpConversionPattern<stablehlo::SliceOp> {
 public:
  StablehloSliceOpPattern(TypeConverter& converter, MLIRContext* ctx)
      : OpConversionPattern(converter, ctx) {}

  LogicalResult matchAndRewrite(
      stablehlo::SliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const PaddedTypeConverter*>(getTypeConverter());
    TensorShardingAttr sharding = getSharding(op.getResult());
    RankedTensorType resultType = op.getResult().getType();
    Type paddedType =
        getPaddedType(resultType, sharding, converter->getSymbolTable());

    if (paddedType == resultType) {
      return padGenericOp(op, adaptor.getOperands(), rewriter, converter);
    }

    auto paddedRankedType = cast<RankedTensorType>(paddedType);
    ArrayRef<int64_t> paddedShape = paddedRankedType.getShape();

    // Update limit_indices to expand the slice to match padded shape.
    ArrayRef<int64_t> limitIndices = op.getLimitIndices();
    SmallVector<int64_t> newLimits = llvm::to_vector(limitIndices);

    TensorShardingAttr operandSharding = getSharding(adaptor.getOperand());
    MeshAttr mesh = sharding.getMesh(converter->getSymbolTable());

    for (int i = 0; i < paddedShape.size(); ++i) {
      newLimits[i] = op.getStartIndices()[i] + paddedShape[i];
      SDY_CHECK(newLimits[i] == limitIndices[i] ||
                isCommunicationFreeSliceDim(i, op, operandSharding, mesh));
    }

    auto newOp = stablehlo::SliceOp::create(
        rewriter, op.getLoc(), paddedType, adaptor.getOperand(),
        rewriter.getDenseI64ArrayAttr(op.getStartIndices()),
        rewriter.getDenseI64ArrayAttr(newLimits),
        rewriter.getDenseI64ArrayAttr(op.getStrides()));

    // Copy sharding attribute to the new result.
    setSharding(newOp.getResult(), sharding);

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

class StablehloDotGeneralOpPattern
    : public OpConversionPattern<stablehlo::DotGeneralOp> {
 public:
  StablehloDotGeneralOpPattern(TypeConverter& converter, MLIRContext* ctx,
                               PaddingCache& cache)
      : OpConversionPattern(converter, ctx), cache(cache) {}

  LogicalResult matchAndRewrite(
      stablehlo::DotGeneralOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const PaddedTypeConverter*>(getTypeConverter());

    Location loc = op.getLoc();
    stablehlo::DotDimensionNumbersAttr dimNums = op.getDotDimensionNumbers();

    Value lhs = adaptor.getOperands()[0];
    auto lhsOrigType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!lhsOrigType) {
      return failure();
    }
    Value paddedLhs =
        ensurePadding(lhs, lhsOrigType, PaddingValueKind::kZero, rewriter, loc,
                      cache, dimNums.getLhsContractingDimensions());

    Value rhs = adaptor.getOperands()[1];
    auto rhsOrigType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    if (!rhsOrigType) {
      return failure();
    }
    Value paddedRhs =
        ensurePadding(rhs, rhsOrigType, PaddingValueKind::kZero, rewriter, loc,
                      cache, dimNums.getRhsContractingDimensions());

    return padGenericOp(op, {paddedLhs, paddedRhs}, rewriter, converter);
  }

 private:
  PaddingCache& cache;
};

struct PadForDivisibilityPass
    : public impl::PadForDivisibilityPassBase<PadForDivisibilityPass> {
  using PadForDivisibilityPassBase::PadForDivisibilityPassBase;

 protected:
  void runOnOperation() final {
    // FuncOpPattern enforces that function inputs and outputs are always fully
    // divisible by sharding requirements. Consequently, padded values never
    // escape the local function scope. This isolation guarantees the cache can
    // be stack-allocated per function.
    PaddingCache paddingCache;
    func::FuncOp funcOp = getOperation();
    ModuleOp module = funcOp->getParentOfType<ModuleOp>();
    SymbolTable symbolTable(module);

    PaddedTypeConverter typeConverter(symbolTable);
    RewritePatternSet patterns(&getContext());
    patterns.add<FuncOpPattern, GenericOpPattern, StablehloSliceOpPattern>(
        typeConverter, &getContext());
    // Sharing the padding cache reference across pattern instances is safe from
    // data races because pattern application within a function is sequential.
    patterns.add<AllSliceOpPattern, StablehloDotGeneralOpPattern,
                 AllGatherOpPattern>(typeConverter, &getContext(),
                                     paddingCache);
    ConversionTarget target(getContext());

    auto isLegalType = [&](Type type, TensorShardingAttr sharding) {
      return getPaddedType(type, sharding, symbolTable) == type;
    };
    auto isLegalValue = [&](Value value) {
      return isLegalType(value.getType(), getSharding(value));
    };

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return llvm::all_of(op.getArguments(), isLegalValue) &&
             llvm::all_of(llvm::seq<int>(0, op.getNumResults()), [&](int i) {
               return isLegalType(op.getResultTypes()[i],
                                  getFuncResultSharding(op, i));
             });
    });
    target.addDynamicallyLegalDialect<stablehlo::StablehloDialect>(
        [&](Operation* op) {
          return llvm::all_of(op->getResults(), isLegalValue) &&
                 llvm::all_of(op->getOperands(), isLegalValue);
        });

    target.addDynamicallyLegalDialect<SdyDialect>([&](Operation* op) {
      if (auto allSliceOp = dyn_cast<AllSliceOp>(op)) {
        return isLegalType(allSliceOp.getOperand().getType(),
                           allSliceOp.getOutSharding());
      }
      if (auto allGatherOp = dyn_cast<AllGatherOp>(op)) {
        return llvm::all_of(op->getOperands(), isLegalValue);
      }
      return true;
    });

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace sdy
}  // namespace mlir
