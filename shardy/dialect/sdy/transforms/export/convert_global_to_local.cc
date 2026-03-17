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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/sdy/ir/dialect.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/transforms/propagation/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_CONVERTGLOBALTOLOCALPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

constexpr int64_t kChannelHandleType = 1;

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
  int64_t nextChannelId = 0;

  void addToConvertOp(Operation* op) { toConvertOps.insert(op); }
  void removeToConvertOp(Operation* op) { toConvertOps.erase(op); }
  bool needConversion(Operation* op) { return toConvertOps.contains(op); }
  int64_t getNextChannelId() { return nextChannelId++; }
};

class GlobalToLocalTypeConverter : public TypeConverter {
 public:
  GlobalToLocalTypeConverter(SymbolTable& symbolTable)
      : symbolTable(symbolTable) {
    addConversion([](Type type) { return type; });

    // Converts global RankedTensorType to local type.
    addConversion([&](Value value) -> std::optional<Type> {
      if (auto type = dyn_cast<RankedTensorType>(value.getType())) {
        return getLocalType(type, getSharding(value), symbolTable);
      }
      // Pass through to other converters if it is not a RankedTensorType.
      return std::nullopt;
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

  // Collects block arguments for ShardableDataFlowOp, such as
  // manual_computation and named_computation.
  void populateArgShardings(ShardableDataFlowOpInterface op) {
    for (auto [arg, sharding] :
         llvm::zip(op.getBlockArgumentEdgeOwners(),
                   op.getBlockArgumentEdgeOwnerShardings())) {
      argShardings[arg] = sharding;
    }
  }

  // Retrieves the sharding of a value, checking the cached argShardings for
  // unlinked block args.
  TensorShardingAttr getSharding(Value value) const {
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      // For block arguments, we look up the pre-populated sharding. This is
      // because the block is unlinked from its original function by the
      // conversion framework when FuncOp signature is converted.
      auto it = argShardings.find(blockArg);
      if (it != argShardings.end()) {
        return it->second;
      }
    }
    // For other values, it's safe to call getSharding as we keep the sharding
    // attribute on the converted op.
    return mlir::sdy::getSharding(value);
  };

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
    if ((op->getDialect()->getNamespace() != "stablehlo" &&
        !isa<sdy::AllReduceOp, sdy::ReturnOp>(op)) ||
        isa<stablehlo::IotaOp>(op)) {
      return failure();
    }

    // Compute local shapes for results.
    SmallVector<Type> newResultTypes;
    llvm::transform(
        op->getResults(), std::back_inserter(newResultTypes),
        [&](Value result) { return typeConverter->convertType(result); });

    // Use OperationState to copy all properties including nested regions.
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(newResultTypes);
    state.addAttributes(op->getAttrs());
    state.addSuccessors(op->getSuccessors());
    for (int i = 0; i < op->getNumRegions(); ++i) {
      state.addRegion();
    }
    Operation* newOp = rewriter.create(state);

    // Move the regions from the old operation to the new one, assuming the
    // region signatures remain unchanged, such as the regions for
    // stablehlo.all_reduce or stablehlo.reduce. For regions that require
    // special handling, they should be handled by specific patterns.
    for (auto [oldRegion, newRegion] :
         llvm::zip(op->getRegions(), newOp->getRegions())) {
      rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.end());
    }

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
    for (auto [index, arg] : llvm::enumerate(op.getArguments())) {
      signature.addInputs(index, converter->convertType(arg));
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

// Returns the channel ID after the maximum channel ID in the given `moduleOp`.
// TODO(b/419222666): remove dependency on `channel_handle` attribute name.
int64_t getNextChannelId(ModuleOp moduleOp) {
  int64_t maxChannelId = 0;
  moduleOp->walk([&](mlir::Operation* op) {
    if (auto channelHandle =
            op->getAttrOfType<stablehlo::ChannelHandleAttr>("channel_handle")) {
      maxChannelId = std::max(maxChannelId, channelHandle.getHandle());
    }
  });
  return maxChannelId + 1;
}

class AllGatherOpPattern : public OpConversionPattern<AllGatherOp> {
 public:
  AllGatherOpPattern(TypeConverter& converter, MLIRContext* ctx,
                     ConversionState& state, bool perDimAllGather)
      : OpConversionPattern<AllGatherOp>(converter, ctx),
        conversionState(state),
        perDimAllGather(perDimAllGather) {}

  LogicalResult rewriteAllGatherCombiningDims(
      AllGatherOp op, MeshAttr mesh, Value input,
      ConversionPatternRewriter& rewriter) const {
    MLIRContext* ctx = op->getContext();
    auto localType = cast<RankedTensorType>(input.getType());
    ArrayRef<int64_t> localShape = localType.getShape();
    int64_t rank = localShape.size();

    SmallVector<int64_t> gatheringDims;
    SmallVector<int64_t> gatheringFactors;
    SmallVector<AxisRefAttr> allGatheringAxes;

    // Collect all gathering axes, their corresponding dimensions and their
    // gathering factors of the dimensions.
    for (auto [dim, axesList] : llvm::enumerate(op.getGatheringAxes())) {
      ::llvm::ArrayRef<AxisRefAttr> axes = axesList.getValue();
      if (axes.empty()) {
        continue;
      }
      gatheringDims.push_back(dim);
      int64_t factor = 1;
      for (AxisRefAttr axis : axes) {
        factor *= axis.getSize(mesh);
        allGatheringAxes.push_back(axis);
      }
      gatheringFactors.push_back(factor);
    }

    SDY_CHECK(gatheringDims.size() > 1);

    // Add a leading dimension of size 1 to gather all partitions at once.
    SmallVector<int64_t> tmpShape = {1};
    llvm::append_range(tmpShape, localShape);
    Value curInput = stablehlo::ReshapeOp::create(
        rewriter, op.getLoc(),
        RankedTensorType::get(tmpShape, localType.getElementType()), input);

    // Perform All-Gather on dim 0.
    DenseIntElementsAttr replicaGroups = getReplicaGroups(
        AxisRefListAttr::get(ctx, allGatheringAxes), mesh, rewriter);
    auto channelHandle = stablehlo::ChannelHandleAttr::get(
        ctx, /*handle=*/conversionState.getNextChannelId(), kChannelHandleType);
    // Change tmpShape to represent the result of the All-Gather.
    tmpShape[0] = replicaGroups.getShapedType().getShape().back();
    auto allGather = stablehlo::AllGatherOp::create(
        rewriter, op.getLoc(),
        TypeRange{RankedTensorType::get(tmpShape, localType.getElementType())},
        curInput, /*all_gather_dim=*/0, replicaGroups, channelHandle,
        /*use_global_device_ids=*/true);
    curInput = allGather.getResult(0);

    // Split the leading dimension into individual gathering factors.
    SmallVector<int64_t> splitShape = gatheringFactors;
    llvm::append_range(splitShape, localShape);
    curInput = stablehlo::ReshapeOp::create(
        rewriter, op.getLoc(),
        RankedTensorType::get(splitShape, localType.getElementType()),
        curInput);

    // Transpose factors next to their corresponding dimensions.
    SmallVector<int64_t> xposePerm;
    int64_t nextFactorIdx = 0;
    for (int64_t i = 0; i < rank; ++i) {
      if (llvm::is_contained(gatheringDims, i)) {
        xposePerm.push_back(nextFactorIdx++);
        xposePerm.push_back(i + gatheringDims.size());
      } else {
        xposePerm.push_back(i + gatheringDims.size());
      }
    }
    curInput = stablehlo::TransposeOp::create(
        rewriter, op.getLoc(), curInput,
        rewriter.getDenseI64ArrayAttr(xposePerm));

    // Merge the factors into the local dimensions.
    SmallVector<int64_t> finalShape = llvm::to_vector(localShape);
    for (size_t j = 0; j < gatheringDims.size(); ++j) {
      finalShape[gatheringDims[j]] *= gatheringFactors[j];
    }
    curInput = stablehlo::ReshapeOp::create(
        rewriter, op.getLoc(),
        RankedTensorType::get(finalShape, localType.getElementType()),
        curInput);

    conversionState.removeToConvertOp(op);
    rewriter.replaceOp(op, curInput);
    return success();
  }

  LogicalResult rewriteAllGatherPerDim(
      AllGatherOp op, MeshAttr mesh, Value input,
      ConversionPatternRewriter& rewriter) const {
    Value curInput = input;
    ::llvm::ArrayRef<AxisRefListAttr> gatheringAxes = op.getGatheringAxes();
    for (int64_t dim = gatheringAxes.size() - 1; dim >= 0; --dim) {
      auto axisList = cast<AxisRefListAttr>(gatheringAxes[dim]);
      if (axisList.empty()) {
        continue;
      }

      auto inputType = cast<RankedTensorType>(curInput.getType());
      SmallVector<int64_t> curShape = llvm::to_vector(inputType.getShape());
      DenseIntElementsAttr replicaGroups =
          getReplicaGroups(axisList, mesh, rewriter);

      int64_t groupSize = replicaGroups.getShapedType().getShape().back();
      if (curShape[dim] != ShapedType::kDynamic) {
        curShape[dim] *= groupSize;
      }

      auto channelHandle = stablehlo::ChannelHandleAttr::get(
          op->getContext(), /*handle=*/conversionState.getNextChannelId(),
          kChannelHandleType);

      auto allGather = stablehlo::AllGatherOp::create(
          rewriter, op.getLoc(),
          TypeRange{
              RankedTensorType::get(curShape, inputType.getElementType())},
          curInput, dim, replicaGroups, channelHandle,
          /*use_global_device_ids=*/true);
      curInput = allGather.getResult(0);
    }

    conversionState.removeToConvertOp(op);
    rewriter.replaceOp(op, curInput);
    return success();
  }

  LogicalResult matchAndRewrite(
      AllGatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    int64_t numGatheringDims = llvm::count_if(
        op.getGatheringAxes(),
        [](AxisRefListAttr axesList) { return !axesList.empty(); });

    SDY_CHECK(numGatheringDims > 0)
        << "No-op AllGatherOp should have been removed by "
           "canonicalization.";

    MeshAttr mesh = op.getOutSharding().getMesh(op);
    if (!mesh) {
      return op.emitOpError("failed to resolve mesh");
    }

    if (perDimAllGather || numGatheringDims == 1) {
      return rewriteAllGatherPerDim(op, mesh, adaptor.getTensor(), rewriter);
    }

    return rewriteAllGatherCombiningDims(op, mesh, adaptor.getTensor(),
                                         rewriter);
  }

 private:
  ConversionState& conversionState;
  bool perDimAllGather;
};

// Returns the logical index of the shard that the given device (`deviceId`)
// resides in, along a dimension sharded by the provided `axes`.
//
// This "shard index" ranges is [0, (TotalShardCount - 1)] and identifies
// the device's position in the logical grid formed by the sharding axes.
int64_t getShardIndex(int64_t deviceId, MeshAttr mesh,
                      ArrayRef<AxisRefAttr> axes) {
  int64_t shardIndex = 0;
  for (AxisRefAttr axis : axes) {
    int64_t axisSize = axis.getSize(mesh);
    int64_t suffixSize = 1;
    bool foundAxis = false;
    // Calculate the product of the sizes of all mesh axes that follow the
    // current axis. This product is the stride needed to extract the axis
    // coordinate from the linear device ID.
    for (MeshAxisAttr meshAxis : mesh.getAxes()) {
      if (foundAxis) {
        suffixSize *= meshAxis.getSize();
      }
      if (meshAxis.getName() == axis.getName()) {
        foundAxis = true;
      }
    }

    // Extract the coordinate component for the current (possibly sub-) axis.
    int64_t axisCoord =
        (deviceId / (suffixSize * axis.getSubAxisPreSize())) % axisSize;

    // Linearize the coordinates of all axes sharding this dimension.
    shardIndex = shardIndex * axisSize + axisCoord;
  }
  return shardIndex;
}

// Returns a 0-rank i64 tensor containing the global offset for the given shard
// axes and local shard size.
Value getDimensionOffset(Location loc, MeshAttr mesh,
                         ArrayRef<AxisRefAttr> axes, int64_t shardSize,
                         ConversionPatternRewriter& rewriter) {
  int64_t numDevices = mesh.getTotalSize();
  Type i64Ty = rewriter.getI64Type();
  auto indexTy = RankedTensorType::get({}, i64Ty);

  // partitionId = (i64)stablehlo.partition_id
  Value partitionId = stablehlo::ConvertOp::create(
      rewriter, loc, indexTy, stablehlo::PartitionIdOp::create(rewriter, loc));

  // Calculate a compile-time offset table for this dimension.
  SmallVector<int64_t> offsetsTable = llvm::to_vector(
      llvm::map_range(llvm::seq<int64_t>(0, numDevices), [&](int64_t devId) {
        return getShardIndex(devId, mesh, axes) * shardSize;
      }));

  // Create the offset table and look up the value for the current device.
  auto tableConst = stablehlo::ConstantOp::create(
      rewriter, loc,
      DenseIntElementsAttr::get(RankedTensorType::get({numDevices}, i64Ty),
                                offsetsTable));
  auto offsetSlice = stablehlo::DynamicSliceOp::create(
      rewriter, loc, RankedTensorType::get({1}, i64Ty), tableConst,
      ValueRange{partitionId}, rewriter.getDenseI64ArrayAttr({1}));

  return stablehlo::ReshapeOp::create(rewriter, loc, indexTy, offsetSlice);
}

Value emitDynamicSliceForAxes(Location loc, Value globalTensor, MeshAttr mesh,
                              ArrayRef<AxisRefListAttr> slicingAxesPerDim,
                              RankedTensorType localResultType,
                              ConversionPatternRewriter& rewriter) {
  // Generate start indices for slicing.
  auto indexTy = RankedTensorType::get({}, rewriter.getI64Type());
  SmallVector<Value> startIndices;
  startIndices.reserve(localResultType.getRank());
  for (int64_t i = 0; i < localResultType.getRank(); ++i) {
    ArrayRef<AxisRefAttr> axes = slicingAxesPerDim[i].getValue();
    if (axes.empty()) {
      startIndices.push_back(stablehlo::ConstantOp::create(
          rewriter, loc,
          DenseIntElementsAttr::get(indexTy, static_cast<int64_t>(0))));
      continue;
    }

    startIndices.push_back(getDimensionOffset(
        loc, mesh, axes, localResultType.getDimSize(i), rewriter));
  }

  return stablehlo::DynamicSliceOp::create(rewriter, loc, localResultType,
                                           globalTensor, startIndices,
                                           localResultType.getShape())
      .getResult();
}

class AllSliceOpPattern : public OpConversionPattern<AllSliceOp> {
 public:
  AllSliceOpPattern(TypeConverter& converter, MLIRContext* ctx,
                    ConversionState& state)
      : OpConversionPattern<AllSliceOp>(converter, ctx),
        conversionState(state) {}

  LogicalResult matchAndRewrite(
      AllSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());
    const SymbolTable& symbolTable = converter->getSymbolTable();
    MeshAttr mesh = op.getOutSharding().getMesh(symbolTable);

    if (!mesh) {
      return op.emitOpError("failed to resolve mesh");
    }

    RankedTensorType localResultType =
        cast<RankedTensorType>(converter->convertType(op));

    rewriter.replaceOp(
        op, emitDynamicSliceForAxes(loc, adaptor.getTensor(), mesh,
                                    op.getSlicingAxesAttr(), localResultType,
                                    rewriter));
    conversionState.removeToConvertOp(op);

    return success();
  }

 private:
  ConversionState& conversionState;
};

class AllToAllOpPattern : public OpConversionPattern<AllToAllOp> {
 public:
  AllToAllOpPattern(TypeConverter& converter, MLIRContext* ctx,
                    ConversionState& state)
      : OpConversionPattern<AllToAllOp>(converter, ctx),
        conversionState(state) {}

  LogicalResult rewriteAllToAllOneParam(
      AllToAllOp op, MeshAttr mesh, Value input,
      ConversionPatternRewriter& rewriter) const {
    AllToAllParamAttr param = op.getParams()[0];
    auto axisList = AxisRefListAttr::get(op->getContext(), param.getAxes());
    if (axisList.empty()) {
      return op.emitOpError("failed to resolve axes");
    }

    DenseIntElementsAttr replicaGroups =
        getReplicaGroups(axisList, mesh, rewriter);
    int64_t numDevicesPerGroup =
        replicaGroups.getShapedType().getShape().back();
    auto inputType = cast<RankedTensorType>(input.getType());
    SmallVector<int64_t> resultShape = llvm::to_vector(inputType.getShape());
    int64_t srcDim = param.getSrcDim();
    int64_t tgtDim = param.getTgtDim();

    if (resultShape[srcDim] == ShapedType::kDynamic ||
        resultShape[tgtDim] == ShapedType::kDynamic) {
      return op.emitOpError("dynamic shape not supported");
    }
    if (resultShape[tgtDim] % numDevicesPerGroup != 0) {
      return op.emitOpError("dimension not divisible by num devices per group");
    }
    resultShape[srcDim] *= numDevicesPerGroup;
    resultShape[tgtDim] /= numDevicesPerGroup;

    auto channelHandle = stablehlo::ChannelHandleAttr::get(
        op->getContext(), /*handle=*/conversionState.getNextChannelId(),
        kChannelHandleType);
    auto allToAll = stablehlo::AllToAllOp::create(
        rewriter, op.getLoc(),
        TypeRange{
            RankedTensorType::get(resultShape, inputType.getElementType())},
        ValueRange{input},
        /*split_dimension=*/tgtDim,
        /*concat_dimension=*/srcDim,
        /*split_count=*/numDevicesPerGroup, replicaGroups, channelHandle);

    conversionState.removeToConvertOp(op);
    rewriter.replaceOp(op, allToAll->getResults());
    return success();
  }

  LogicalResult rewriteAllToAllMultipleParams(
      AllToAllOp op, MeshAttr mesh, Value input,
      ConversionPatternRewriter& rewriter) const {
    Location loc = op.getLoc();
    auto inputType = cast<RankedTensorType>(input.getType());
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t rank = inputShape.size();
    ArrayRef<AllToAllParamAttr> params = op.getParams();
    int64_t numParams = params.size();

    // Sort parameters by target_dims.
    SmallVector<int64_t> sortedParamIndices(numParams);
    std::iota(sortedParamIndices.begin(), sortedParamIndices.end(), 0);
    llvm::sort(sortedParamIndices, [&](int64_t a, int64_t b) {
      return params[a].getTgtDim() < params[b].getTgtDim();
    });

    // For each param with index sortedparamIndices[i], compute the number of
    // devices in each param and collect all axes.
    SmallVector<int64_t> numDevicesPerParams;
    SmallVector<AxisRefAttr> allAxes;
    for (int64_t idx : sortedParamIndices) {
      numDevicesPerParams.push_back(
          getTotalAxesSize(params[idx].getAxes(), mesh));
      for (auto axis : params[idx].getAxes()) {
        allAxes.push_back(axis);
      }
    }

    // Split the input on target_dims using the number of devices per param as
    // splitted factors, transpose the splitted factors to the front, then
    // reshape the splitted factors to a single dim in dim 0.

    // The shape after we split the input.
    SmallVector<int64_t> shape0;
    // The permutation on the splitted input to move the factors to the front.
    SmallVector<int64_t> permutation0;
    for (int64_t i = 0; i < rank; ++i) {
      auto* it = llvm::find_if(sortedParamIndices, [&](int64_t idx) {
        return params[idx].getTgtDim() == i;
      });
      if (it == sortedParamIndices.end()) {
        shape0.push_back(inputShape[i]);
        continue;
      }
      int64_t numDevicesPerParam =
          numDevicesPerParams[std::distance(sortedParamIndices.begin(), it)];
      permutation0.push_back(shape0.size());
      shape0.push_back(numDevicesPerParam);
      if (inputShape[i] == ShapedType::kDynamic ||
          inputShape[i] % numDevicesPerParam != 0) {
        return op.emitOpError(
            "dynamic shape or dimension not divisible by num devices per "
            "param");
      }
      shape0.push_back(inputShape[i] / numDevicesPerParam);
    }
    for (int64_t i = 0; i < shape0.size(); ++i) {
      if (!llvm::is_contained(permutation0, i)) {
        permutation0.push_back(i);
      }
    }

    Value reshape0 = stablehlo::ReshapeOp::create(
        rewriter, loc,
        RankedTensorType::get(shape0, inputType.getElementType()), input);
    Value transpose0 =
        stablehlo::TransposeOp::create(rewriter, loc, reshape0, permutation0);
    SmallVector<int64_t> transpose0Shape = llvm::to_vector(
        cast<RankedTensorType>(transpose0.getType()).getShape());
    DenseIntElementsAttr replicaGroups = getReplicaGroups(
        AxisRefListAttr::get(rewriter.getContext(), allAxes), mesh, rewriter);
    int64_t numDevicesPerGroup =
        replicaGroups.getShapedType().getShape().back();
    // The size of dim 0 that contains all the splitted factors, which is the
    // the same as the number of devices in each replica group.
    SmallVector<int64_t> shape1 = {numDevicesPerGroup};
    llvm::append_range(
        shape1, ArrayRef<int64_t>(transpose0Shape).drop_front(numParams));
    Value reshape1 = stablehlo::ReshapeOp::create(
        rewriter, loc,
        RankedTensorType::get(shape1, inputType.getElementType()), transpose0);

    // Perform all-to-all on the combined dim 0.
    auto channelHandle = stablehlo::ChannelHandleAttr::get(
        rewriter.getContext(), conversionState.getNextChannelId(), /*type=*/1);
    auto allToAll = stablehlo::AllToAllOp::create(
        rewriter, loc, reshape1.getType(), reshape1,
        /*split_dimension=*/0, /*concat_dimension=*/0,
        /*split_count=*/numDevicesPerGroup, replicaGroups, channelHandle);

    // Distribute gathered data to src_dims.

    // Split dim 0 back into factors.
    Value reshape2 = stablehlo::ReshapeOp::create(
        rewriter, loc, transpose0.getType(), allToAll.getResult(0));
    // Align factors back to their corresponding source dimensions.
    SmallVector<int64_t> permutation1;
    for (int64_t i = 0; i < rank; ++i) {
      permutation1.push_back(i + numParams);
    }
    for (int64_t i = 0; i < numParams; ++i) {
      int64_t srcDim = params[sortedParamIndices[i]].getSrcDim();
      auto* it = llvm::find(permutation1, srcDim + numParams);
      permutation1.insert(it, i);
    }
    Value transpose1 =
        stablehlo::TransposeOp::create(rewriter, loc, reshape2, permutation1);
    SmallVector<int64_t> transpose1Shape = llvm::to_vector(
        cast<RankedTensorType>(transpose1.getType()).getShape());
    SmallVector<int64_t> finalShape;
    for (int64_t i = 0; i < transpose1Shape.size(); ++i) {
      if (permutation1[i] < numParams) {
        // Restore the full local dimension by merging factor and quotient.
        finalShape.push_back(transpose1Shape[i] * transpose1Shape[i + 1]);
        i++;
      } else {
        finalShape.push_back(transpose1Shape[i]);
      }
    }

    Value reshape3 = stablehlo::ReshapeOp::create(
        rewriter, loc,
        RankedTensorType::get(finalShape, inputType.getElementType()),
        transpose1);

    rewriter.replaceOp(op, reshape3);
    conversionState.removeToConvertOp(op);
    return success();
  }

  LogicalResult matchAndRewrite(
      AllToAllOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    MeshAttr mesh = op.getOutSharding().getMesh(op);
    if (!mesh) {
      return op.emitOpError("failed to resolve mesh");
    }

    SDY_CHECK(!op.getParams().empty());
    if (op.getParams().size() == 1) {
      return rewriteAllToAllOneParam(op, mesh, adaptor.getTensor(), rewriter);
    }
    return rewriteAllToAllMultipleParams(op, mesh, adaptor.getTensor(),
                                         rewriter);
  }

 private:
  ConversionState& conversionState;
};

class CollectivePermuteOpPattern
    : public OpConversionPattern<CollectivePermuteOp> {
 public:
  CollectivePermuteOpPattern(TypeConverter& converter, MLIRContext* ctx,
                             ConversionState& state)
      : OpConversionPattern<CollectivePermuteOp>(converter, ctx),
        conversionState(state) {}

  LogicalResult matchAndRewrite(
      CollectivePermuteOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());
    TensorShardingAttr outSharding = op.getOutSharding();
    TensorShardingAttr inSharding = converter->getSharding(op.getTensor());

    SDY_CHECK(outSharding != inSharding)
        << "Shardy canonicalizer should have removed no-op collective-permute.";

    MeshAttr mesh = outSharding.getMesh(converter->getSymbolTable());
    if (!mesh) {
      return op.emitOpError("failed to resolve mesh");
    }

    // Collective all AxisRef to produce an ordered list of all sharding axes.
    SmallVector<AxisRefAttr> allAxisRefs;
    for (TensorShardingAttr sharding : {inSharding, outSharding}) {
      if (sharding) {
        sharding.forEachAxisRef(
            [&](AxisRefAttr axis) { allAxisRefs.push_back(axis); });
      }
    }
    SmallVector<AxisRefAttr> allOrderedAxes = getOrderedAxisRefs(
        AxisRefListAttr::get(getContext(), allAxisRefs), mesh);

    // Return the ordered list of axes for a given sharding that is a
    // permutation of allOrderedAxes. This ensures every device ID is mapped to
    // a unique logical ID in the space defined by these sharding axes.
    auto getShardingAxes = [&](TensorShardingAttr sharding) {
      SmallVector<AxisRefAttr> axes;
      for (DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
        for (AxisRefAttr axis : dimSharding.getAxes()) {
          for (AxisRefAttr atomic : allOrderedAxes) {
            if (axis.getName() == atomic.getName() && axis.contains(atomic)) {
              if (!llvm::is_contained(axes, atomic)) {
                axes.push_back(atomic);
              }
            }
          }
        }
      }
      // Add axes not involved in slicing.
      for (AxisRefAttr atomic : allOrderedAxes) {
        if (!llvm::is_contained(axes, atomic)) {
          axes.push_back(atomic);
        }
      }
      return axes;
    };

    SmallVector<AxisRefAttr> outShardingAxes = getShardingAxes(outSharding);
    int64_t numDevices = mesh.getTotalSize();
    llvm::DenseMap<int64_t, int64_t> logicalIdToOutDevId;
    for (int64_t j = 0; j < numDevices; ++j) {
      logicalIdToOutDevId[getShardIndex(j, mesh, outShardingAxes)] = j;
    }

    SmallVector<AxisRefAttr> inShardingAxes = getShardingAxes(inSharding);
    SmallVector<int64_t> pairs;
    for (int64_t i = 0; i < numDevices; ++i) {
      pairs.push_back(i);
      pairs.push_back(
          logicalIdToOutDevId[getShardIndex(i, mesh, inShardingAxes)]);
    }

    auto pairsAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({numDevices, 2}, rewriter.getI64Type()), pairs);
    auto channel = stablehlo::ChannelHandleAttr::get(
        getContext(), conversionState.getNextChannelId(), 1);
    rewriter.replaceOpWithNewOp<stablehlo::CollectivePermuteOp>(
        op, adaptor.getTensor().getType(), adaptor.getTensor(), pairsAttr,
        channel);

    conversionState.removeToConvertOp(op);
    return success();
  }

 private:
  ConversionState& conversionState;
};

class ConstantOpPattern : public OpConversionPattern<sdy::ConstantOp> {
 public:
  ConstantOpPattern(TypeConverter& converter, MLIRContext* ctx,
                    ConversionState& state)
      : OpConversionPattern<sdy::ConstantOp>(converter, ctx),
        conversionState(state) {}

  LogicalResult matchAndRewrite(
      sdy::ConstantOp op, OpAdaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type globalType = op.getType();
    auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());
    auto localType = cast<RankedTensorType>(converter->convertType(op));
    ElementsAttr elementsAttr = op.getValue();
    Location loc = op.getLoc();

    // Unsharded or splat constants.
    if (localType == globalType || elementsAttr.isSplat()) {
      Attribute newAttr =
          localType == globalType
              ? elementsAttr
              : SplatElementsAttr::get(localType,
                                       elementsAttr.getSplatValue<Attribute>());

      rewriter.replaceOp(
          op, stablehlo::ConstantOp::create(rewriter, loc, localType,
                                            cast<ElementsAttr>(newAttr)));
      conversionState.removeToConvertOp(op);
      return success();
    }

    // Sharded dense constants.
    const SymbolTable& symbolTable = converter->getSymbolTable();
    TensorShardingAttr sharding = getSharding(op.getResult());
    MeshAttr mesh = sharding.getMesh(symbolTable);
    if (!mesh) {
      return op.emitOpError("failed to resolve mesh for constant");
    }
    auto globalConst =
        stablehlo::ConstantOp::create(rewriter, loc, globalType, elementsAttr);

    SmallVector<AxisRefListAttr> slicingAxesPerDim;
    slicingAxesPerDim.reserve(localType.getRank());
    for (DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
      slicingAxesPerDim.push_back(
          AxisRefListAttr::get(op.getContext(), dimSharding.getAxes()));
    }

    rewriter.replaceOp(
        op, emitDynamicSliceForAxes(loc, globalConst, mesh, slicingAxesPerDim,
                                    localType, rewriter));
    conversionState.removeToConvertOp(op);

    return success();
  }

 private:
  ConversionState& conversionState;
};

class ManualComputationOpPattern
    : public OpConversionPattern<ManualComputationOp> {
 public:
  ManualComputationOpPattern(TypeConverter& converter, MLIRContext* ctx,
                             ConversionState& state)
      : OpConversionPattern<ManualComputationOp>(converter, ctx),
        conversionState(state) {}

  LogicalResult matchAndRewrite(
      ManualComputationOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());

    // Prepare signature conversion and convert the region.
    TypeConverter::SignatureConversion signature(
        op.getBody().getNumArguments());
    for (auto [index, arg] : llvm::enumerate(op.getBody().getArguments())) {
      signature.addInputs(index, converter->convertType(arg));
    }
    if (failed(rewriter.convertRegionTypes(&op.getBody(), *converter,
                                           &signature))) {
      return failure();
    }

    // Inline the region.
    Block& bodyBlock = op.getBody().front();
    auto sdyReturnOp = cast<sdy::ReturnOp>(bodyBlock.getTerminator());
    SmallVector<Value> results = sdyReturnOp.getResults();
    rewriter.inlineBlockBefore(&bodyBlock, op, adaptor.getTensors());

    // Replace the uses of sdy.manual_computation with the inlined return
    // values.
    rewriter.replaceOp(op, results);
    rewriter.eraseOp(sdyReturnOp);

    conversionState.removeToConvertOp(op);
    return success();
  }

 private:
  ConversionState& conversionState;
};

class ReduceScatterOpPattern : public OpConversionPattern<ReduceScatterOp> {
 public:
  ReduceScatterOpPattern(TypeConverter& converter, MLIRContext* ctx,
                         ConversionState& state,
                         bool combineMultiDimensionReduceScatter)
      : OpConversionPattern<ReduceScatterOp>(converter, ctx),
        conversionState(state),
        combineMultiDimensionReduceScatter(combineMultiDimensionReduceScatter) {
  }

  LogicalResult rewriteReduceScatterOneDim(
      ReduceScatterOp op, Value input, int64_t scatterDim, AxisRefListAttr axes,
      MeshAttr mesh, ConversionPatternRewriter& rewriter) const {
    Location loc = op.getLoc();
    auto inputType = cast<RankedTensorType>(input.getType());
    const auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());
    auto localResultType = cast<RankedTensorType>(converter->convertType(op));

    auto channelHandle = stablehlo::ChannelHandleAttr::get(
        op->getContext(), conversionState.getNextChannelId(), /*type=*/1);
    DenseIntElementsAttr replicaGroups = getReplicaGroups(axes, mesh, rewriter);

    auto reduceScatter = stablehlo::ReduceScatterOp::create(
        rewriter, loc, localResultType, input, scatterDim, replicaGroups,
        channelHandle, /*use_global_device_ids=*/true);
    stablehlo::buildReduceBody<stablehlo::AddOp>(
        inputType.getElementType(), reduceScatter.getComputation(), rewriter);

    rewriter.replaceOp(op, reduceScatter.getResult());
    conversionState.removeToConvertOp(op);
    return success();
  }

  LogicalResult rewriteReduceScatterCombiningMultipleDims(
      ReduceScatterOp op, Value input, MeshAttr mesh,
      ArrayRef<AxisRefListAttr> slicingAxesPerDim,
      ArrayRef<AxisRefAttr> allReduceAxes,
      ConversionPatternRewriter& rewriter) const {
    Location loc = op.getLoc();
    auto inputType = cast<RankedTensorType>(input.getType());
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t rank = inputShape.size();
    const auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());
    auto localResultType = cast<RankedTensorType>(converter->convertType(op));

    // Identify the scattering dimensions and calculate their factors.
    SmallVector<int64_t> scatteredDims;
    SmallVector<int64_t> scatteredFactors;
    for (int64_t i = 0; i < rank; ++i) {
      if (!slicingAxesPerDim[i].empty()) {
        scatteredDims.push_back(i);
        scatteredFactors.push_back(
            getTotalAxesSize(slicingAxesPerDim[i], mesh));
      }
    }

    // Reshape to split each scattered dimension into (factor, quotient).
    SmallVector<int64_t> splitShape;
    // Maps the original dimension index to the index of its 'factor' sub-dim in
    // splitShape.
    SmallVector<int64_t> factorDimIndices;
    for (int64_t i = 0; i < rank; ++i) {
      auto* it = llvm::find(scatteredDims, i);
      if (it == scatteredDims.end()) {
        splitShape.push_back(inputShape[i]);
      }
      int64_t factor =
          scatteredFactors[std::distance(scatteredDims.begin(), it)];
      factorDimIndices.push_back(splitShape.size());
      splitShape.push_back(factor);
      if (inputShape[i] == ShapedType::kDynamic ||
          inputShape[i] % factor != 0) {
        return op.emitOpError("dimension not divisible by scattering factor");
      }
      splitShape.push_back(inputShape[i] / factor);
    }
    Value curInput = stablehlo::ReshapeOp::create(
        rewriter, loc,
        RankedTensorType::get(splitShape, inputType.getElementType()), input);

    // Transpose to move all scattering factor dimensions to the front.
    SmallVector<int64_t> permutation = factorDimIndices;
    for (int64_t i = 0; i < static_cast<int64_t>(splitShape.size()); ++i) {
      if (!llvm::is_contained(factorDimIndices, i)) {
        permutation.push_back(i);
      }
    }
    curInput = stablehlo::TransposeOp::create(
        rewriter, loc, curInput, rewriter.getDenseI64ArrayAttr(permutation));

    // Reshape to combine all factor dimensions into a single leading dimension
    // 0.
    int64_t totalFactor =
        llvm::accumulate(scatteredFactors, 1, std::multiplies<int64_t>());
    SmallVector<int64_t> combinedShape = {totalFactor};
    llvm::append_range(combinedShape, localResultType.getShape());
    curInput = stablehlo::ReshapeOp::create(
        rewriter, loc,
        RankedTensorType::get(combinedShape, inputType.getElementType()),
        curInput);

    // Perform one stablehlo.reduce_scatter on dimension 0.
    DenseIntElementsAttr replicaGroups = getReplicaGroups(
        AxisRefListAttr::get(rewriter.getContext(), allReduceAxes), mesh,
        rewriter);
    auto channelHandle = stablehlo::ChannelHandleAttr::get(
        op->getContext(), conversionState.getNextChannelId(), /*type=*/1);

    // The result of the reduce-scatter will have dimension 0 reduced to size 1.
    combinedShape[0] = 1;
    auto rsType =
        RankedTensorType::get(combinedShape, inputType.getElementType());
    auto reduceScatter = stablehlo::ReduceScatterOp::create(
        rewriter, loc, rsType, curInput, /*scatter_dimension=*/0, replicaGroups,
        channelHandle, /*use_global_device_ids=*/true);
    stablehlo::buildReduceBody<stablehlo::AddOp>(
        inputType.getElementType(), reduceScatter.getComputation(), rewriter);

    // Reshape to remove the leading dimension of size 1, matching the final
    // local shape.
    rewriter.replaceOp(
        op, stablehlo::ReshapeOp::create(rewriter, loc, localResultType,
                                         reduceScatter.getResult()));
    conversionState.removeToConvertOp(op);

    return success();
  }

  LogicalResult rewriteReduceScatterWithAllReduceAndDynamicSlice(
      ReduceScatterOp op, Value input, MeshAttr mesh,
      ArrayRef<AxisRefListAttr> slicingAxesPerDim,
      ArrayRef<AxisRefAttr> allReduceAxes,
      ConversionPatternRewriter& rewriter) const {
    Location loc = op.getLoc();
    auto inputType = cast<RankedTensorType>(input.getType());
    const auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());
    auto localResultType = cast<RankedTensorType>(converter->convertType(op));

    // Perform one All-Reduce across all involved axes.
    auto channelHandle = stablehlo::ChannelHandleAttr::get(
        op->getContext(), conversionState.getNextChannelId(), /*type=*/1);
    DenseIntElementsAttr replicaGroups = getReplicaGroups(
        AxisRefListAttr::get(rewriter.getContext(), allReduceAxes), mesh,
        rewriter);

    auto allReduce = stablehlo::AllReduceOp::create(
        rewriter, loc, inputType, input, replicaGroups, channelHandle,
        /*use_global_device_ids=*/true);
    stablehlo::buildReduceBody<stablehlo::AddOp>(
        inputType.getElementType(), allReduce.getComputation(), rewriter);

    // Perform the "Scatter" part using a multi-dimensional DynamicSlice.
    Value localPiece =
        emitDynamicSliceForAxes(loc, allReduce.getResult(0), mesh,
                                slicingAxesPerDim, localResultType, rewriter);

    rewriter.replaceOp(op, localPiece);
    conversionState.removeToConvertOp(op);
    return success();
  }

  LogicalResult matchAndRewrite(
      ReduceScatterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());
    const SymbolTable& symbolTable = converter->getSymbolTable();
    MeshAttr mesh = converter->getSharding(op.getTensor()).getMesh(symbolTable);
    if (!mesh) {
      return op.emitOpError("failed to resolve mesh");
    }

    // Collect axes and count dimensions involved in the scatter.
    SmallVector<AxisRefAttr> allReduceAxes;
    int64_t lastSlicingDim;
    int64_t numSlicingDims = 0;
    for (auto [dim, axesList] : llvm::enumerate(op.getReduceScatterAxes())) {
      if (!axesList.empty()) {
        lastSlicingDim = dim;
        numSlicingDims++;
        allReduceAxes.append(axesList.getValue().begin(),
                             axesList.getValue().end());
      }
    }

    SDY_CHECK(numSlicingDims > 0)
        << "Shardy should have canonicalized no-op reduce-scatter.";

    if (numSlicingDims == 1) {
      return rewriteReduceScatterOneDim(
          op, adaptor.getTensor(), lastSlicingDim,
          cast<AxisRefListAttr>(op.getReduceScatterAxes()[lastSlicingDim]),
          mesh, rewriter);
    }
    if (combineMultiDimensionReduceScatter) {
      return rewriteReduceScatterCombiningMultipleDims(
          op, adaptor.getTensor(), mesh, op.getReduceScatterAxesAttr(),
          allReduceAxes, rewriter);
    }
    return rewriteReduceScatterWithAllReduceAndDynamicSlice(
        op, adaptor.getTensor(), mesh, op.getReduceScatterAxesAttr(),
        allReduceAxes, rewriter);
  }

 private:
  ConversionState& conversionState;
  bool combineMultiDimensionReduceScatter;
};

class NamedComputationOpPattern
    : public OpConversionPattern<NamedComputationOp> {
 public:
  NamedComputationOpPattern(TypeConverter& converter, MLIRContext* ctx,
                            ConversionState& state)
      : OpConversionPattern<NamedComputationOp>(converter, ctx),
        conversionState(state) {}

  LogicalResult matchAndRewrite(
      NamedComputationOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());

    SmallVector<Type> newResultTypes;
    llvm::transform(
        op.getResults(), std::back_inserter(newResultTypes),
        [&](Value result) { return converter->convertType(result); });

    // Prepare signature conversion for the internal region.
    TypeConverter::SignatureConversion signature(
        op.getBody().getNumArguments());
    for (auto [index, arg] : llvm::enumerate(op.getBody().getArguments())) {
      signature.addInputs(index, converter->convertType(arg));
    }

    auto newOp = NamedComputationOp::create(
        rewriter, op.getLoc(), newResultTypes, op.getName(),
        adaptor.getOperands(), op.getInShardings().value_or(nullptr),
        op.getOutShardings().value_or(nullptr));

    // Move and convert the region.
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), *converter,
                                           &signature))) {
      return failure();
    }

    rewriter.replaceOp(op, newOp.getResults());
    conversionState.removeToConvertOp(op);
    return success();
  }

 private:
  ConversionState& conversionState;
};

class StablehloIotaOpPattern : public OpConversionPattern<stablehlo::IotaOp> {
 public:
  StablehloIotaOpPattern(TypeConverter& converter, MLIRContext* ctx,
                         ConversionState& state)
      : OpConversionPattern<stablehlo::IotaOp>(converter, ctx),
        conversionState(state) {}

  LogicalResult matchAndRewrite(
      stablehlo::IotaOp op, OpAdaptor,
      ConversionPatternRewriter& rewriter) const override {
    TensorShardingAttr sharding = getSharding(op->getResult(0));
    if (isFullyReplicated(sharding)) {
      conversionState.removeToConvertOp(op);
      return success();
    }

    auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());
    MeshAttr mesh = sharding.getMesh(converter->getSymbolTable());
    if (!mesh) {
      return op.emitOpError("failed to resolve mesh");
    }
    auto localType = cast<RankedTensorType>(converter->convertType(op));
    Location loc = op.getLoc();
    int64_t iotaDim = op.getIotaDimension();
    // Create the local iota.
    Value localIota =
        stablehlo::IotaOp::create(rewriter, loc, localType, iotaDim);
    if (sharding.getDimShardings()[iotaDim].getAxes().empty()) {
      rewriter.replaceOp(op, localIota);
      conversionState.removeToConvertOp(op);
      return success();
    }

    // Calculate and apply the global offset for this shard.
    int64_t shardSize = localType.getDimSize(iotaDim);
    Type convertedOffsetType =
        RankedTensorType::get({}, localType.getElementType());
    Value offset = getDimensionOffset(
        loc, mesh, sharding.getDimShardings()[iotaDim].getAxes(), shardSize,
        rewriter);
    Value offsetConverted = stablehlo::ConvertOp::create(
        rewriter, loc, convertedOffsetType, offset);
    Value broadcastOffset = stablehlo::BroadcastOp::create(
        rewriter, loc, localType, offsetConverted,
        rewriter.getDenseI64ArrayAttr(localType.getShape()));
    rewriter.replaceOpWithNewOp<stablehlo::AddOp>(op, localIota,
                                                  broadcastOffset);

    conversionState.removeToConvertOp(op);
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
  using ConvertGlobalToLocalPassBase::ConvertGlobalToLocalPassBase;

 protected:
  void runOnOperation() final {
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);
    GlobalToLocalTypeConverter typeConverter(symbolTable);

    module.walk([&](Operation* op) {
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        typeConverter.populateArgShardings(funcOp);
      } else if (auto dataFlowOp = dyn_cast<ShardableDataFlowOpInterface>(op)) {
        typeConverter.populateArgShardings(dataFlowOp);
      }
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
    conversionState.nextChannelId = getNextChannelId(module);

    RewritePatternSet patterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    patterns
        .add<AllSliceOpPattern, AllToAllOpPattern, CollectivePermuteOpPattern,
             ConstantOpPattern, FuncOpSignaturePattern, GenericOpPattern,
             ManualComputationOpPattern, NamedComputationOpPattern,
             ReturnOpPattern, StablehloIotaOpPattern>(
            typeConverter, &getContext(), conversionState);
    patterns.add<AllGatherOpPattern>(typeConverter, &getContext(),
                                     conversionState, perDimAllGather);
    patterns.add<ReduceScatterOpPattern>(typeConverter, &getContext(),
                                         conversionState,
                                         combineMultiDimensionReduceScatter);

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
