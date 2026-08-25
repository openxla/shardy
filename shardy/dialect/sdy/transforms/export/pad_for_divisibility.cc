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
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/TypeInference.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_PADFORDIVISIBILITYPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

class PaddedTypeConverter : public TypeConverter {
 public:
  explicit PaddedTypeConverter(const SymbolTable& symbolTable)
      : symbolTable(symbolTable) {
    addConversion([](Type type) { return type; });

    addConversion([&](Value value) -> std::optional<Type> {
      if (auto type = dyn_cast<RankedTensorType>(value.getType())) {
        return getDivisiblePaddedType(type, getSharding(value), symbolTable);
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

constexpr PaddingValueKind kDefaultPaddingValueKind = PaddingValueKind::kZero;

// Returns true if the operation has custom padding handling implemented in
// this file and should be excluded from GenericOpPattern.
bool hasCustomPadHandling(Operation* op) {
  return isa<stablehlo::SliceOp, stablehlo::DotGeneralOp, stablehlo::PadOp,
             stablehlo::ConvolutionOp, stablehlo::ReshapeOp>(op);
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
// and adds the padded value to the PaddingCache if registeredKind is present.
Value createPaddedValue(RankedTensorType paddedType, Value value,
                        PaddingValueKind paddingKind,
                        std::optional<PaddingValueKind> registeredKind,
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
  if (registeredKind) {
    cache.setPadding(padOp, *registeredKind);
  }
  return padOp;
}

// Returns 'inputVal' if the value is not padded or the value already has
// 'padVal'. Otherwise, uses compare-and-select to produce a new padded
// value from inputVal with 'padVal' and returns the new value.
// This ensures elements in 'inputVal' at indices >= 'origType' dimensions
// are masked with 'padVal'.
//
// We ensure all dimensions that require padding are padded with 'padVal'
// unless dimsToEnforce is provided, in which case only the specified
// dimensions are padded.
Value ensurePaddingWithValue(
    Value inputVal, RankedTensorType origType, Value padVal, OpBuilder& b,
    Location loc,
    std::optional<ArrayRef<int64_t>> dimsToEnforce = std::nullopt) {
  auto paddedType = cast<RankedTensorType>(inputVal.getType());
  if (origType == paddedType) {
    return inputVal;
  }
  TensorShardingAttr sharding = getSharding(inputVal);

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
    if (sharding) {
      setSharding(iota, sharding);
    }
    Value limit = stablehlo::ConstantOp::create(
        b, loc,
        DenseElementsAttr::get(RankedTensorType::get({}, b.getI32Type()),
                               b.getI32IntegerAttr(origSize)));
    Value broadcastLimit = stablehlo::BroadcastInDimOp::create(
        b, loc, iotaType, limit, b.getDenseI64ArrayAttr({}));
    if (sharding) {
      setSharding(broadcastLimit, sharding);
    }
    Value mask = stablehlo::CompareOp::create(
        b, loc, iota, broadcastLimit, stablehlo::ComparisonDirection::LT);
    if (sharding) {
      setSharding(mask, sharding);
    }
    if (validDataMask) {
      validDataMask = stablehlo::AndOp::create(b, loc, validDataMask, mask);
      if (sharding) {
        setSharding(validDataMask, sharding);
      }
    } else {
      validDataMask = mask;
    }
  }

  if (!validDataMask) {
    return inputVal;
  }
  Value bcastPadVal = stablehlo::BroadcastInDimOp::create(
      b, loc, paddedType, padVal, b.getDenseI64ArrayAttr({}));
  if (sharding) {
    setSharding(bcastPadVal, sharding);
  }

  Value select =
      stablehlo::SelectOp::create(b, loc, validDataMask, inputVal, bcastPadVal);
  if (sharding) {
    setSharding(select, sharding);
  }
  return select;
}

// This routine is similar to ensurePaddingWithValue, but it uses a
// PaddingValueKind to determine the padding value instead of a user-provided
// value and may put the result into the PaddingCache.
Value ensurePaddingWithKind(
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
  Value newPaddingScalar =
      createConstant(b, loc, paddedType.getElementType(), requiredKind);
  Value select = ensurePaddingWithValue(inputVal, origType, newPaddingScalar, b,
                                        loc, dimsToEnforce);
  if (select != inputVal && !dimsToEnforce) {
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
    shardableOperands.push_back(sdy::getShardableValue(operand));
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
    Type paddedType = getDivisiblePaddedType(
        result.getType(), getSharding(result), typeConverter->getSymbolTable());
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

  // For now, generic ops do not propagate padding kinds, so they remain
  // unregistered in the cache.

  rewriter.replaceOp(op, newOp->getResults());
  return success();
}

// Slices/trims `res` back to its original sizes on `trimDims`.
Value trimOutputForDims(Value res, Type origType, ArrayRef<int64_t> trimDims,
                        TensorShardingAttr outSharding,
                        ConversionPatternRewriter& rewriter,
                        std::optional<PaddingValueKind> paddingKind,
                        PaddingCache& cache) {
  auto origRanked = dyn_cast<RankedTensorType>(origType);
  auto newRanked = dyn_cast<RankedTensorType>(res.getType());
  if (!origRanked || !newRanked) {
    return res;
  }

  SmallVector<int64_t> limitIndices;
  limitIndices.reserve(origRanked.getRank());
  bool needsSlice = false;
  for (int d = 0; d < origRanked.getRank(); ++d) {
    int64_t origSize = origRanked.getDimSize(d);
    int64_t newSize = newRanked.getDimSize(d);
    if (llvm::is_contained(trimDims, d) && newSize > origSize) {
      limitIndices.push_back(origSize);
      needsSlice = true;
    } else {
      limitIndices.push_back(newSize);
    }
  }

  if (!needsSlice) {
    return res;
  }

  SmallVector<int64_t> starts(origRanked.getRank(), 0);
  SmallVector<int64_t> strides(origRanked.getRank(), 1);
  auto sliceOp = stablehlo::SliceOp::create(
      rewriter, res.getLoc(),
      RankedTensorType::get(limitIndices, origRanked.getElementType()), res,
      rewriter.getDenseI64ArrayAttr(starts),
      rewriter.getDenseI64ArrayAttr(limitIndices),
      rewriter.getDenseI64ArrayAttr(strides));
  setSharding(sliceOp.getResult(), outSharding);
  Value sliceRes = sliceOp.getResult();
  if (paddingKind) {
    cache.setPadding(sliceRes, *paddingKind);
  }
  return sliceRes;
}

// Pads `input` to `paddedInputType` if shapes differ, creates a new collective
// operation with the padded shape, and registers its result padding in the
// cache.
Operation* padCollectiveOp(Operation* op, Value input, Value inputOrig,
                           RankedTensorType paddedInputType,
                           const SymbolTable& symbolTable,
                           ConversionPatternRewriter& rewriter,
                           PaddingCache& cache) {
  auto rankedInput = cast<RankedTensorType>(input.getType());
  Value padOp = input;
  std::optional<PaddingValueKind> paddingKind = cache.getPadding(input);
  bool isPaddedBefore = input.getType() != inputOrig.getType();

  if (paddedInputType != rankedInput) {
    PaddingValueKind constantKind = (isPaddedBefore && paddingKind)
                                        ? *paddingKind
                                        : kDefaultPaddingValueKind;
    std::optional<PaddingValueKind> registeredKind = std::nullopt;
    if (!isPaddedBefore) {
      registeredKind = kDefaultPaddingValueKind;
    } else if (paddingKind) {
      registeredKind = paddingKind;
    }
    padOp = createPaddedValue(paddedInputType, input, constantKind,
                              registeredKind, symbolTable, rewriter, cache);
    paddingKind = registeredKind;
  }

  OperationState state(op->getLoc(), op->getName());
  state.addOperands({padOp});
  state.addTypes({paddedInputType});
  state.addAttributes(op->getAttrs());
  Operation* newOp = rewriter.create(state);
  if (paddingKind) {
    cache.setPadding(newOp->getResult(0), *paddingKind);
  }
  return newOp;
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
         !isa<sdy::ReturnOp, sdy::AllReduceOp, sdy::ShardedToUnreducedOp,
              sdy::ReplicatedToUnreducedOp>(op)) ||
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
      if (getDivisiblePaddedType(arg.getType(), getSharding(arg),
                                 symbolTable) != arg.getType()) {
        return op.emitOpError()
               << "argument #" << index << " has a non-divisible sharding. "
               << "Shardy expects function IO to be divisible.";
      }
    }

    for (int i = 0; i < op.getNumResults(); ++i) {
      Type resultType = op.getResultTypes()[i];
      if (getDivisiblePaddedType(resultType, getFuncResultSharding(op, i),
                                 symbolTable) != resultType) {
        return op.emitOpError()
               << "result #" << i << " has a non-divisible sharding. "
               << "Shardy expects function IO to be divisible.";
      }
    }

    return failure();
  }
};

// Returns the dimension indices of `op` that are gathered across devices and
// need to be trimmed (sliced back to their original size) after the op.
SmallVector<int64_t> getTrimDims(sdy::AllGatherOp op) {
  SmallVector<int64_t> trimDims;
  for (auto [d, axes] : llvm::enumerate(op.getGatheringAxes())) {
    if (!axes.empty()) {
      trimDims.push_back(d);
    }
  }
  return trimDims;
}

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
      shardableOperands.push_back(sdy::getShardableValue(operand));
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
      Type paddedType = getDivisiblePaddedType(
          result.getType(), getSharding(result), symbolTable);
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

    Value input = adaptor.getOperands()[0];
    std::optional<PaddingValueKind> paddingKind = cache.getPadding(input);
    if (paddingKind) {
      for (Value res : newOp->getResults()) {
        cache.setPadding(res, *paddingKind);
      }
    }

    SmallVector<Value> replacements;
    replacements.reserve(op->getNumResults());
    SmallVector<int64_t> trimDims = getTrimDims(op);

    for (int i = 0; i < op->getNumResults(); ++i) {
      Value res = newOp->getResult(i);
      Value trimmed =
          trimOutputForDims(res, op->getResult(i).getType(), trimDims,
                            getSharding(res), rewriter, paddingKind, cache);
      replacements.push_back(trimmed);
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
    Value inputOrig = op->getOperand(0);
    auto rankedInputType = dyn_cast<RankedTensorType>(input.getType());
    if (!rankedInputType) {
      return failure();
    }

    TensorShardingAttr outSharding = op.getOutSharding();
    RankedTensorType paddedInputType = cast<RankedTensorType>(
        getDivisiblePaddedType(rankedInputType, outSharding, symbolTable));

    Operation* newOp = padCollectiveOp(op, input, inputOrig, paddedInputType,
                                       symbolTable, rewriter, cache);

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

 private:
  PaddingCache& cache;
};

class ReduceScatterOpPattern
    : public OpConversionPattern<sdy::ReduceScatterOp> {
 public:
  ReduceScatterOpPattern(TypeConverter& converter, MLIRContext* ctx,
                         PaddingCache& cache)
      : OpConversionPattern(converter, ctx), cache(cache) {}

  LogicalResult matchAndRewrite(
      sdy::ReduceScatterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const PaddedTypeConverter*>(getTypeConverter());
    const SymbolTable& symbolTable = converter->getSymbolTable();

    Value input = adaptor.getOperands()[0];
    Value inputOrig = op->getOperand(0);
    auto rankedInputType = dyn_cast<RankedTensorType>(input.getType());
    if (!rankedInputType) {
      return failure();
    }

    TensorShardingAttr outSharding = op.getOutSharding();
    RankedTensorType paddedInputType = cast<RankedTensorType>(
        getDivisiblePaddedType(rankedInputType, outSharding, symbolTable));

    Operation* newOp = padCollectiveOp(op, input, inputOrig, paddedInputType,
                                       symbolTable, rewriter, cache);

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

 private:
  PaddingCache& cache;
};

// Returns the source dimensions (dims where input is gathered/combined).
SmallVector<int64_t> getTrimDims(sdy::AllToAllOp op) {
  SmallVector<int64_t> trimDims;
  trimDims.reserve(op.getParams().size());
  for (AllToAllParamAttr param : op.getParams()) {
    trimDims.push_back(param.getSrcDim());
  }
  return trimDims;
}

class AllToAllOpPattern : public OpConversionPattern<sdy::AllToAllOp> {
 public:
  AllToAllOpPattern(TypeConverter& converter, MLIRContext* ctx,
                    PaddingCache& cache)
      : OpConversionPattern(converter, ctx), cache(cache) {}

  LogicalResult matchAndRewrite(
      sdy::AllToAllOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const PaddedTypeConverter*>(getTypeConverter());
    const SymbolTable& symbolTable = converter->getSymbolTable();

    Value input = adaptor.getOperands()[0];
    Value inputOrig = op->getOperand(0);
    auto rankedInputType = dyn_cast<RankedTensorType>(input.getType());
    if (!rankedInputType) {
      return failure();
    }

    TensorShardingAttr outSharding = op.getOutSharding();
    RankedTensorType paddedInputType = cast<RankedTensorType>(
        getDivisiblePaddedType(rankedInputType, outSharding, symbolTable));

    std::optional<PaddingValueKind> paddingKind = cache.getPadding(input);
    Operation* newOp = padCollectiveOp(op, input, inputOrig, paddedInputType,
                                       symbolTable, rewriter, cache);

    Value res = newOp->getResult(0);

    SmallVector<int64_t> trimDims = getTrimDims(op);
    res = trimOutputForDims(res, inputOrig.getType(), trimDims, outSharding,
                            rewriter, paddingKind, cache);

    rewriter.replaceOp(op, {res});
    return success();
  }

 private:
  PaddingCache& cache;
};

class StablehloReshapeOpPattern
    : public OpConversionPattern<stablehlo::ReshapeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::ReshapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const PaddedTypeConverter*>(getTypeConverter());
    TensorShardingAttr inSharding = getSharding(op.getOperand());
    TensorShardingAttr outSharding = getSharding(op.getResult());
    MeshAttr mesh =
        inSharding ? inSharding.getMesh(converter->getSymbolTable()) : nullptr;

    if (!isCommunicationFreeReshape(op, inSharding, outSharding, mesh,
                                    op.getOperand().getType(), op.getType())) {
      return op.emitOpError(
          "participating reshape dimensions are not divisible. Reshape "
          "sharding "
          "should have been resolved by resolve-permutation-factors.");
    }

    return padGenericOp(op, adaptor.getOperands(), rewriter, converter);
  }
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
    RankedTensorType paddedType = cast<RankedTensorType>(getDivisiblePaddedType(
        resultType, sharding, converter->getSymbolTable()));

    if (paddedType == resultType) {
      return padGenericOp(op, adaptor.getOperands(), rewriter, converter);
    }

    ArrayRef<int64_t> paddedShape = paddedType.getShape();

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

class StablehloPadOpPattern : public OpConversionPattern<stablehlo::PadOp> {
 public:
  explicit StablehloPadOpPattern(TypeConverter& converter, MLIRContext* ctx)
      : OpConversionPattern(converter, ctx) {}

  LogicalResult matchAndRewrite(
      stablehlo::PadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const PaddedTypeConverter*>(getTypeConverter());
    const SymbolTable& symbolTable = converter->getSymbolTable();

    Value input = adaptor.getOperands()[0];
    auto rankedInputType = dyn_cast<RankedTensorType>(input.getType());
    if (!rankedInputType) {
      return failure();
    }

    Value result = op.getResult();
    RankedTensorType paddedType = cast<RankedTensorType>(getDivisiblePaddedType(
        result.getType(), getSharding(result), symbolTable));

    ArrayRef<int64_t> low = op.getEdgePaddingLow();
    ArrayRef<int64_t> high = op.getEdgePaddingHigh();
    ArrayRef<int64_t> interior = op.getInteriorPadding();

    // We adjust edge-padding-high to ensure that the new pad op has the
    // expected result type. We may increase this trailing padding region
    // if the original result size is not divisible. We may also decrease
    // this trailing padding region, if there is non-trivial interior
    // padding and the original input is not divisible. This is because
    // padding the input for divisibility can increases the size of the new
    // pad op to beyond the original padded result size.
    //
    // We use this formula to calculate `high` as the new edge-padding-high to
    // the padd op:
    // paddedResultSize = paddedInputSize + low + high +
    //   std::max(0, paddedInputSize - 1) * interior
    SmallVector<int64_t> newHigh;
    newHigh.reserve(rankedInputType.getRank());
    bool highPaddingReduced = false;
    for (int d = 0; d < rankedInputType.getRank(); ++d) {
      int64_t paddedInputSize = rankedInputType.getDimSize(d);
      int64_t paddedResultSize = paddedType.getDimSize(d);
      int64_t dLow = low[d];
      int64_t dInterior = interior[d];
      int64_t dHigh = paddedResultSize - paddedInputSize - dLow -
                      std::max<int64_t>(0, paddedInputSize - 1) * dInterior;
      newHigh.push_back(dHigh);
      if (dHigh < high[d]) {
        highPaddingReduced = true;
      }
    }

    if (highPaddingReduced) {
      // When high padding is reduced, the input's padding region takes the
      // place of the trailing padding value for the pad op. As such, we need
      // to ensure the input is padded with the correct value.
      input = ensurePaddingWithValue(input, op.getOperand().getType(),
                                     adaptor.getOperands()[1], rewriter,
                                     op.getLoc());
    }

    auto newHighAttr = rewriter.getDenseI64ArrayAttr(newHigh);

    // Create the new pad operation.
    auto newPadOp = stablehlo::PadOp::create(
        rewriter, op.getLoc(), paddedType, input, adaptor.getOperands()[1],
        op.getEdgePaddingLowAttr(), newHighAttr, op.getInteriorPaddingAttr());

    // Copy sharding attribute to the new result.
    setSharding(newPadOp.getResult(), getSharding(result));

    // TODO(b/537380378): We could parse the padding value from the original pad
    // op to cache the padding kind and avoid downstream selects.
    rewriter.replaceOp(op, newPadOp.getResult());
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
    Value rhs = adaptor.getOperands()[1];
    Value lhsOrig = op->getOperand(0);
    Value rhsOrig = op->getOperand(1);
    auto lhsOrigType = dyn_cast<RankedTensorType>(lhsOrig.getType());
    auto rhsOrigType = dyn_cast<RankedTensorType>(rhsOrig.getType());
    if (!lhsOrigType || !rhsOrigType) {
      return failure();
    }
    Value paddedLhs = ensurePaddingWithKind(
        lhs, lhsOrigType, PaddingValueKind::kZero, rewriter, loc, cache,
        dimNums.getLhsContractingDimensions());
    Value paddedRhs = ensurePaddingWithKind(
        rhs, rhsOrigType, PaddingValueKind::kZero, rewriter, loc, cache,
        dimNums.getRhsContractingDimensions());

    return padGenericOp(op, {paddedLhs, paddedRhs}, rewriter, converter);
  }

 private:
  PaddingCache& cache;
};

class StablehloConvolutionOpPattern
    : public OpConversionPattern<stablehlo::ConvolutionOp> {
 public:
  StablehloConvolutionOpPattern(TypeConverter& converter, MLIRContext* ctx,
                                PaddingCache& cache)
      : OpConversionPattern(converter, ctx), cache(cache) {}

  LogicalResult matchAndRewrite(
      stablehlo::ConvolutionOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const PaddedTypeConverter*>(getTypeConverter());

    Location loc = op.getLoc();
    stablehlo::ConvDimensionNumbersAttr dimNums = op.getDimensionNumbers();

    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];
    Value lhsOrig = op->getOperand(0);
    Value rhsOrig = op->getOperand(1);
    auto lhsOrigType = dyn_cast<RankedTensorType>(lhsOrig.getType());
    auto rhsOrigType = dyn_cast<RankedTensorType>(rhsOrig.getType());
    if (!lhsOrigType || !rhsOrigType) {
      return failure();
    }

    SmallVector<int64_t> lhsEnforceDims;
    lhsEnforceDims.push_back(dimNums.getInputFeatureDimension());
    llvm::append_range(lhsEnforceDims, dimNums.getInputSpatialDimensions());
    SmallVector<int64_t> rhsEnforceDims;
    rhsEnforceDims.push_back(dimNums.getKernelInputFeatureDimension());
    llvm::append_range(rhsEnforceDims, dimNums.getKernelSpatialDimensions());

    // Enforce zero-padding on dimensions that need to be padded, except for
    // pass-through dimensions (like batch or kernel output feature) which are
    // left out of lhsEnforceDims and rhsEnforceDims.
    Value paddedLhs =
        ensurePaddingWithKind(lhs, lhsOrigType, PaddingValueKind::kZero,
                              rewriter, loc, cache, lhsEnforceDims);
    Value paddedRhs =
        ensurePaddingWithKind(rhs, rhsOrigType, PaddingValueKind::kZero,
                              rewriter, loc, cache, rhsEnforceDims);

    SmallVector<ShapedTypeComponents> inferredReturnShapes;
    if (failed(mlir::hlo::inferConvolutionOp(
            op->getLoc(), paddedLhs.getType(), paddedRhs.getType(),
            op.getWindowStrides(), op.getPaddingAttr(), op.getLhsDilation(),
            op.getRhsDilation(), op.getWindowReversal(),
            dimNums.getInputBatchDimension(),
            dimNums.getInputFeatureDimension(),
            dimNums.getInputSpatialDimensions(),
            dimNums.getKernelInputFeatureDimension(),
            dimNums.getKernelOutputFeatureDimension(),
            dimNums.getKernelSpatialDimensions(),
            dimNums.getOutputBatchDimension(),
            dimNums.getOutputFeatureDimension(),
            dimNums.getOutputSpatialDimensions(), op.getFeatureGroupCount(),
            op.getBatchGroupCount(), op.getPrecisionConfig(),
            inferredReturnShapes))) {
      return failure();
    }

    auto inferredResultType = RankedTensorType::get(
        inferredReturnShapes[0].getDims(), lhsOrigType.getElementType());
    Value result = op.getResult();
    Type paddedResultType = getDivisiblePaddedType(
        result.getType(), getSharding(result), converter->getSymbolTable());
    auto paddedResultShaped = cast<RankedTensorType>(paddedResultType);

    // Assert that the spatial/window dimensions do not require post-op padding
    // (i.e. the mathematically inferred size is greater than or equal to the
    // target padded size). Any indivisibility that requires post-op padding
    // must be resolved (made divisible or replicated) by
    // resolve-permutation-factors upstream.
    for (int64_t spatialDim : dimNums.getOutputSpatialDimensions()) {
      if (inferredResultType.getDimSize(spatialDim) <
          paddedResultShaped.getDimSize(spatialDim)) {
        return rewriter.notifyMatchFailure(
            op,
            "inferred convolution output spatial size is smaller than the "
            "target padded size. Spatial sharding must be resolved by "
            "resolve-permutation-factors to be divisible or replicated.");
      }
    }

    // Recreate the convolution operation.
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(
        {getShardableValue(paddedLhs), getShardableValue(paddedRhs)});
    state.addTypes(inferredResultType);
    state.addAttributes(op->getAttrs());
    Operation* newOp = rewriter.create(state);

    TensorShardingAttr outSharding = getSharding(result);
    setSharding(newOp->getResult(0), outSharding);

    // Trim the output spatial dimensions back to the target padded size.
    Value finalResult =
        trimOutputForDims(newOp->getResult(0), paddedResultShaped,
                          dimNums.getOutputSpatialDimensions(), outSharding,
                          rewriter, PaddingValueKind::kZero, cache);

    rewriter.replaceOp(op, finalResult);
    return success();
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
    patterns.add<GenericOpPattern, StablehloSliceOpPattern>(typeConverter,
                                                            &getContext());
    patterns.add<StablehloPadOpPattern, StablehloReshapeOpPattern>(
        typeConverter, &getContext());
    patterns.add<FuncOpPattern>(typeConverter, &getContext());
    // Sharing the padding cache reference across pattern instances is safe from
    // data races because pattern application within a function is sequential.
    patterns.add<AllSliceOpPattern, StablehloDotGeneralOpPattern,
                 StablehloConvolutionOpPattern, AllToAllOpPattern,
                 AllGatherOpPattern, ReduceScatterOpPattern>(
        typeConverter, &getContext(), paddingCache);
    ConversionTarget target(getContext());

    auto isLegalType = [&](Type type, TensorShardingAttr sharding) {
      return getDivisiblePaddedType(type, sharding, symbolTable) == type;
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
      if (auto reduceScatterOp = dyn_cast<ReduceScatterOp>(op)) {
        return isLegalType(reduceScatterOp.getOperand().getType(),
                           reduceScatterOp.getOutSharding());
      }
      if (auto allGatherOp = dyn_cast<AllGatherOp>(op)) {
        return llvm::all_of(op->getOperands(), isLegalValue);
      }
      if (auto allToAllOp = dyn_cast<AllToAllOp>(op)) {
        return llvm::all_of(op->getOperands(), isLegalValue) &&
               llvm::all_of(op->getResults(), isLegalValue);
      }
      if (isa<AllReduceOp, ShardedToUnreducedOp, ReplicatedToUnreducedOp>(op)) {
        return llvm::all_of(op->getOperands(), isLegalValue) &&
               llvm::all_of(op->getResults(), isLegalValue);
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
