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
#include "llvm/ADT/SmallVectorExtras.h"
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

enum class PaddingValueKind { kZero, kOne, kUnknown };

inline StringRef toString(PaddingValueKind kind) {
  switch (kind) {
    case PaddingValueKind::kZero:
      return "kZero";
    case PaddingValueKind::kOne:
      return "kOne";
    case PaddingValueKind::kUnknown:
      return "kUnknown";
  }
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
    if (it != cache.end()) return it->second;
    return std::nullopt;
  }

  const DenseMap<Value, PaddingValueKind>& getCache() const { return cache; }

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
    case PaddingValueKind::kUnknown:
      llvm_unreachable("Cannot create constant for kUnknown padding");
  }
  llvm_unreachable("invalid PaddingValueKind");
}

// Creates a padded value for 'value' with 'paddedType' and 'desiredKind'.
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
  cache.setPadding(padOp, paddingKind);
  return padOp;
}

// Converts op to its local version by replacing its operands with the already
// converted operands.
LogicalResult padGenericOp(Operation* op, ValueRange operands,
                           ConversionPatternRewriter& rewriter,
                           const PaddedTypeConverter* typeConverter,
                           PaddingCache& cache) {
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

  for (Value result : newOp->getResults()) {
    cache.setPadding(result, PaddingValueKind::kUnknown);
  }

  rewriter.replaceOp(op, newOp->getResults());
  return success();
}

// Pattern for ops that just need operands updates and result type updates to
// match padded shape.
class GenericOpPattern : public ConversionPattern {
 public:
  GenericOpPattern(TypeConverter& converter, MLIRContext* ctx,
                   PaddingCache& cache)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx),
        cache(cache) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    Dialect* dialect = op->getDialect();
    if ((dialect && dialect->getNamespace() != "stablehlo" &&
         !isa<sdy::ReturnOp>(op)) ||
        isa<stablehlo::SliceOp>(op)) {
      return failure();
    }
    return padGenericOp(op, operands, rewriter,
                        static_cast<const PaddedTypeConverter*>(typeConverter),
                        cache);
  }

 private:
  PaddingCache& cache;
};

class FuncOpPattern : public OpConversionPattern<func::FuncOp> {
 public:
  FuncOpPattern(TypeConverter& converter, MLIRContext* ctx)
      : OpConversionPattern(converter, ctx) {}

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
          if (auto kind = cache.getPadding(adaptor.getTensor())) {
            cache.setPadding(sliceOp.getResult(), *kind);
          }
          replacements.push_back(sliceOp);
          continue;
        }
      }
      if (auto kind = cache.getPadding(adaptor.getTensor())) {
        cache.setPadding(res, *kind);
      }
      replacements.push_back(res);
    }

    rewriter.replaceOp(op, replacements);
    return success();
  }

 private:
  PaddingCache& cache;
};

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
    if (paddedInputType == rankedInputType) {
      return padGenericOp(op, adaptor.getOperands(), rewriter, converter,
                          cache);
    }

    Value padOp = createPaddedValue(cast<RankedTensorType>(paddedInputType),
                                    input, PaddingValueKind::kZero, symbolTable,
                                    rewriter, cache);
    OperationState state(op->getLoc(), op->getName());
    state.addOperands({padOp});
    state.addTypes(
        {getPaddedType(op.getResult().getType(), outSharding, symbolTable)});
    state.addAttributes(op->getAttrs());
    Operation* newOp = rewriter.create(state);

    if (auto kind = cache.getPadding(padOp)) {
      cache.setPadding(newOp->getResult(0), *kind);
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

 private:
  PaddingCache& cache;
};

class StablehloSliceOpPattern : public OpConversionPattern<stablehlo::SliceOp> {
 public:
  StablehloSliceOpPattern(TypeConverter& converter, MLIRContext* ctx,
                          PaddingCache& cache)
      : OpConversionPattern(converter, ctx), cache(cache) {}

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
      return padGenericOp(op, adaptor.getOperands(), rewriter, converter,
                          cache);
    }

    auto paddedRankedType = cast<RankedTensorType>(paddedType);
    ArrayRef<int64_t> paddedShape = paddedRankedType.getShape();

    // Update limit_indices to expand the slice to match padded shape.
    ArrayRef<int64_t> limitIndices = op.getLimitIndices();
    SmallVector<int64_t> newLimits = llvm::to_vector(limitIndices);

    for (int i = 0; i < paddedShape.size(); ++i) {
      newLimits[i] = op.getStartIndices()[i] + paddedShape[i];
    }

    auto newOp = stablehlo::SliceOp::create(
        rewriter, op.getLoc(), paddedType, adaptor.getOperand(),
        rewriter.getDenseI64ArrayAttr(op.getStartIndices()),
        rewriter.getDenseI64ArrayAttr(newLimits),
        rewriter.getDenseI64ArrayAttr(op.getStrides()));

    // Copy sharding attribute to the new result.
    setSharding(newOp.getResult(), sharding);

    if (auto kind = cache.getPadding(adaptor.getOperand())) {
      cache.setPadding(newOp.getResult(), *kind);
    }

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }

 private:
  PaddingCache& cache;
};

// Dumps the padding cache content as `sdy.padding_kinds` array attributes on
// the operations producing the padded values. This is used only for testing and
// debugging.
void dumpPaddingCacheAttributes(const PaddingCache& paddingCache) {
  // Map from an operation to the padding kinds of its results.
  DenseMap<Operation*, SmallVector<StringRef>> opToPaddingKinds;
  for (auto [value, kind] : paddingCache.getCache()) {
    // We only attach attributes to operations, so skip block/function
    // arguments.
    auto opResult = dyn_cast<OpResult>(value);
    if (!opResult) continue;

    Operation* op = opResult.getOwner();
    auto& kinds = opToPaddingKinds[op];
    if (kinds.empty()) {
      kinds.resize(op->getNumResults(), "none");
    }
    kinds[opResult.getResultNumber()] = toString(kind);
  }

  // Attach the grouped result padding kinds as an array attribute on each
  // operation.
  for (const auto& [op, kinds] : opToPaddingKinds) {
    Builder b(op->getContext());
    auto attrElts =
        llvm::map_to_vector(kinds, [&](StringRef kindStr) -> Attribute {
          return b.getStringAttr(kindStr);
        });
    op->setAttr("sdy.padding_kinds", b.getArrayAttr(attrElts));
  }
}

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
    patterns.add<FuncOpPattern>(typeConverter, &getContext());
    // Sharing the padding cache reference across pattern instances is safe from
    // data races because pattern application within a function is sequential.
    patterns.add<AllSliceOpPattern, AllGatherOpPattern, GenericOpPattern,
                 StablehloSliceOpPattern>(typeConverter, &getContext(),
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

    if (this->dumpPaddingCache) {
      dumpPaddingCacheAttributes(paddingCache);
    }
  }
};

}  // namespace
}  // namespace sdy
}  // namespace mlir
