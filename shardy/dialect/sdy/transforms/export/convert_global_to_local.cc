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

#include <iterator>
#include <optional>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/sdy/ir/dialect.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_CONVERTGLOBALTOLOCALPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

// Computes the local type of a given type and sharding.
//
// If the sharding is not defined or the mesh is not defined, the original
// type is returned. If the sharding is defined and the mesh is defined,
// the local type is computed using the sharding. If the local type
// computation fails, an error is reported.
Type getLocalType(Type type, TensorShardingAttr sharding,
                  const SymbolTable& symbolTable) {
  if (!sharding) {
    return type;
  }
  MeshAttr mesh = sharding.getMesh(symbolTable);
  if (!mesh) {
    return type;
  }

  Type localType =
      sharding.getLocalType(type, mesh, /*allowNonDivisible=*/false);
  SDY_CHECK(localType)
      << "Failed to compute local type due to non-divisible sharding";
  return localType;
}

struct ConversionState {
  llvm::DenseSet<Operation*> toConvertOps;

  void addToConvertOp(Operation* op) { toConvertOps.insert(op); }
  void removeToConvertOp(Operation* op) { toConvertOps.erase(op); }
  bool needConversion(Operation* op) { return toConvertOps.contains(op); }
};

class GlobalToLocalTypeConverter : public TypeConverter {
 public:
  GlobalToLocalTypeConverter(SymbolTable& symbolTable)
      : symbolTable(symbolTable) {
    addConversion([](Type type) { return type; });

    // Converts global RankedTensorType to local type.
    addConversion([&](Value value) -> std::optional<Type> {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type) {
        // Pass through to other converters if it is not a RankedTensorType.
        return std::nullopt;
      }
      TensorShardingAttr sharding;
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        // For block arguments, we look up the pre-populated sharding. This is
        // because the block is unlinked from its original function by the
        // conversion framework when FuncOp signature is converted.
        auto it = argShardings.find(blockArg);
        if (it != argShardings.end()) {
          sharding = it->second;
        }
      } else {
        // For other values, it's safe to call getSharding as we keep the
        // sharding attribute on the converted op.
        sharding = getSharding(value);
      }
      return getLocalType(type, sharding, symbolTable);
    });

    // Materializations to resolve intermediate casts.
    auto materialize = [](OpBuilder& b, Type t, ValueRange inputs,
                          Location loc) -> Value {
      return UnrealizedConversionCastOp::create(b, loc, t, inputs).getResult(0);
    };
    addSourceMaterialization(materialize);
    addTargetMaterialization(materialize);
  }

  // Constructs a map from block argument to sharding attribute, as the block
  // will be unlinked during the FuncOp signature conversion which prevents us
  // from using the getSharding() utility function.
  void populateArgShardings(func::FuncOp funcOp) {
    for (BlockArgument arg : funcOp.getArguments()) {
      if (auto sharding = funcOp.getArgAttrOfType<TensorShardingAttr>(
              arg.getArgNumber(), "sdy.sharding")) {
        argShardings[arg] = sharding;
      }
    }
  }

  const SymbolTable& getSymbolTable() const { return symbolTable; }

 private:
  const SymbolTable& symbolTable;
  llvm::DenseMap<Value, TensorShardingAttr> argShardings;
};

// Pattern for generic ops that do not require special handling beyond type
// conversion, such as StableHLO ops, sdy.all_reduce, etc.
class GenericOpPattern : public ConversionPattern {
 public:
  GenericOpPattern(TypeConverter& converter, MLIRContext* ctx,
                   ConversionState& state)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, ctx),
        conversionState(state) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    // Skip non-StableHLO non-sdy ops and ops with specific patterns.
    // TODO(bixia): add ops with specific conversion patterns.
    if ((op->getDialect()->getNamespace() != "stablehlo" &&
         op->getDialect()->getNamespace() != "sdy")) {
      return failure();
    }

    // Compute local shapes for results.
    SmallVector<Type> newResultTypes;
    llvm::transform(
        op->getResults(), std::back_inserter(newResultTypes),
        [&](Value result) { return typeConverter->convertType(result); });
    Operation* newOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(), operands,
                        newResultTypes, op->getAttrs(), op->getSuccessors());
    rewriter.replaceOp(op, newOp->getResults());
    conversionState.removeToConvertOp(op);
    return success();
  }

 private:
  ConversionState& conversionState;
};

class ReturnOpPattern : public OpConversionPattern<func::ReturnOp> {
 public:
  ReturnOpPattern(TypeConverter& converter, MLIRContext* ctx,
                  ConversionState& state)
      : OpConversionPattern<func::ReturnOp>(converter, ctx),
        conversionState(state) {}

  LogicalResult matchAndRewrite(
      func::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    conversionState.removeToConvertOp(op);
    return success();
  }

 private:
  ConversionState& conversionState;
};

class FuncOpSignaturePattern : public OpConversionPattern<func::FuncOp> {
 public:
  FuncOpSignaturePattern(TypeConverter& converter, MLIRContext* ctx,
                         ConversionState& state)
      : OpConversionPattern<func::FuncOp>(converter, ctx),
        conversionState(state) {}

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());
    const SymbolTable& symbolTable = converter->getSymbolTable();

    TypeConverter::SignatureConversion signature(op.getNumArguments());
    for (const BlockArgument& arg : op.getArguments()) {
      signature.addInputs(arg.getArgNumber(), converter->convertType(arg));
    }
    SmallVector<Type> newResultTypes;
    for (int i = 0; i < op.getNumResults(); ++i) {
      Type globalType = op.getResultTypes()[i];
      if (auto rankedType = dyn_cast<RankedTensorType>(globalType)) {
        auto sharding = getFuncResultSharding(op, i);
        newResultTypes.push_back(
            getLocalType(rankedType, sharding, symbolTable));
      } else {
        newResultTypes.push_back(globalType);
      }
    }
    conversionState.removeToConvertOp(op);
    auto newFuncType =
        rewriter.getFunctionType(signature.getConvertedTypes(), newResultTypes);
    // Update the function type.
    rewriter.modifyOpInPlace(op, [&] { op.setType(newFuncType); });

    if (failed(rewriter.convertRegionTypes(&op.getBody(), *getTypeConverter(),
                                           &signature))) {
      return failure();
    }

    return success();
  }

 private:
  ConversionState& conversionState;
};

// This pass converts a Shardy module with consistent sharding notations and
// global tensor types to a module with local tensor types.
//
// The conversion is based on TensorShardingAttr which is not part of a type
// representation. For example, in order to convert the type of a value into a
// local type, we use getSharding(value) to retreat the TensorShardingAttr from
// its defining op or its owning op (for block arguments). The problem of not
// having TensorShardingAttr as part of the type is that we can't just look at
// the type and its associated TensorShardingAttr to tell whether the type for
// a given value has already been converted and become "legal". To resolve this,
// we keep track of the converted ops in the state of the converter and consider
// only ops in the set as legal.
//
// When a FuncOp is converted, the function body is unlinked by the conversion
// framework. After that, when an op of the function body is converted, we can
// no longer use getSharding() to retrieve the sharding attribute for a function
// argument. To resolve this, we collect a map from function arguments to
// sharding attributes before we start to convert any ops.
//
struct ConvertGlobalToLocalPass
    : public impl::ConvertGlobalToLocalPassBase<ConvertGlobalToLocalPass> {
 protected:
  void runOnOperation() final {
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);
    GlobalToLocalTypeConverter typeConverter(symbolTable);

    module.walk([&](func::FuncOp funcOp) {
      typeConverter.populateArgShardings(funcOp);
    });

    ConversionState conversionState;
    // Walk the module and collect the set of ops that need to be converted.
    // We use the set to determine whether a given op is legal or not during
    // conversion.
    module.walk([&](Operation* op) {
      if (isa<MeshOp>(op)) {
        return;
      }
      conversionState.addToConvertOp(op);
    });

    RewritePatternSet patterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    patterns.add<FuncOpSignaturePattern, ReturnOpPattern, GenericOpPattern>(
        typeConverter, &getContext(), conversionState);

    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp op) { return !conversionState.needConversion(op); });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return !conversionState.needConversion(op); });

    target.addDynamicallyLegalDialect<stablehlo::StablehloDialect>(
        [&](Operation* op) { return !conversionState.needConversion(op); });

    target.addDynamicallyLegalDialect<SdyDialect>(
        [&](Operation* op) { return !conversionState.needConversion(op); });
    target.addLegalOp<MeshOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
