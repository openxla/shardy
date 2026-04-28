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
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
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
      // The block argument is not sharded.
      return nullptr;
    }
    // For other values, it's safe to call getSharding as we keep the sharding
    // attribute on the converted op.
    return mlir::sdy::getSharding(value);
  };

  SmallVector<Type> convertResultTypes(ValueRange results) const {
    SmallVector<Type> localResultTypes;
    localResultTypes.reserve(results.size());
    llvm::transform(results, std::back_inserter(localResultTypes),
                    [&](Value result) { return convertType(result); });
    return localResultTypes;
  }

  const SymbolTable& getSymbolTable() const { return symbolTable; }

 private:
  const SymbolTable& symbolTable;
  llvm::DenseMap<Value, TensorShardingAttr> argShardings;
};

// Copies all attributes from 'op' to 'newOp', except those specified in
// 'attributesToExclude'.
void copyAttributes(Operation* op, Operation* newOp,
                    ArrayRef<StringRef> attributesToExclude) {
  for (auto attr : op->getAttrs()) {
    if (!llvm::is_contained(attributesToExclude, attr.getName().getValue())) {
      newOp->setAttr(attr.getName(), attr.getValue());
    }
  }
}

// Converts op to its local version by replacing its operands with the already
// converted operands and removing the op from the toConvertOps list.
LogicalResult localizeGenericOp(Operation* op, ValueRange operands,
                                ConversionPatternRewriter& rewriter,
                                const GlobalToLocalTypeConverter* typeConverter,
                                ConversionState& conversionState) {
  // Use OperationState to copy all properties including nested regions.
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(operands);
  state.addTypes(typeConverter->convertResultTypes(op->getResults()));
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
    Dialect* dialect = op->getDialect();
    if ((dialect && dialect->getNamespace() != "stablehlo" &&
         !isa<sdy::ReturnOp>(op)) ||
        isa<stablehlo::ConvolutionOp, stablehlo::DotGeneralOp, stablehlo::DotOp,
            stablehlo::GatherOp, stablehlo::IotaOp, stablehlo::PadOp,
            stablehlo::ScatterOp, stablehlo::SliceOp>(op)) {
      return failure();
    }
    return localizeGenericOp(
        op, operands, rewriter,
        static_cast<const GlobalToLocalTypeConverter*>(typeConverter),
        conversionState);
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

class AllReduceOpPattern : public OpConversionPattern<sdy::AllReduceOp> {
 public:
  AllReduceOpPattern(TypeConverter& converter, MLIRContext* ctx,
                     ConversionState& state)
      : OpConversionPattern<sdy::AllReduceOp>(converter, ctx),
        conversionState(state) {}

  LogicalResult matchAndRewrite(
      sdy::AllReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    TensorShardingAttr outSharding = op.getOutSharding();
    SDY_CHECK(outSharding);

    auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());
    MeshAttr mesh = outSharding.getMesh(converter->getSymbolTable());
    if (!mesh) {
      return op.emitOpError("failed to resolve mesh");
    }

    DenseIntElementsAttr replicaGroups =
        getReplicaGroups(op.getReductionAxesAttr(), mesh, rewriter);
    auto channelHandle = stablehlo::ChannelHandleAttr::get(
        op->getContext(), conversionState.getNextChannelId(),
        kChannelHandleType);

    auto allReduce = stablehlo::AllReduceOp::create(
        rewriter, op.getLoc(), converter->convertResultTypes(op->getResults()),
        adaptor.getOperands(), replicaGroups, channelHandle,
        /*use_global_device_ids=*/true);
    Type elementType = cast<RankedTensorType>(allReduce->getResult(0).getType())
                           .getElementType();
    stablehlo::buildReduceBody<stablehlo::AddOp>(
        elementType, allReduce.getComputation(), rewriter);

    rewriter.replaceOp(op, allReduce.getResults());
    conversionState.removeToConvertOp(op);
    return success();
  }

 private:
  ConversionState& conversionState;
};

// Returns the logical index of the shard that the given device (`deviceId`)
// resides in, along a dimension sharded by the provided `axes`.
//
// This "shard index" ranges is [0, (TotalShardCount - 1)] and identifies
// the device's position in the logical grid formed by the sharding axes.
int64_t getShardIndex(int64_t deviceId, MeshAttr mesh,
                      ArrayRef<AxisRefAttr> axes) {
  // Resolve the physical-to-logical mapping. The 'logicalDeviceId' is the index
  // of the device ID in the mesh's device list. If the list is empty, it
  // follows the iota order.
  int64_t logicalDeviceId = deviceId;
  ArrayRef<int64_t> deviceIds = mesh.getDeviceIds();
  if (!deviceIds.empty()) {
    const auto* it = llvm::find(deviceIds, deviceId);
    SDY_CHECK(it != deviceIds.end()) << "Device ID not found in mesh";
    logicalDeviceId = std::distance(deviceIds.begin(), it);
  }

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
        (logicalDeviceId / (suffixSize * axis.getSubAxisPreSize())) % axisSize;

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
  SmallVector<int64_t> offsetsTable = llvm::map_to_vector(
      llvm::seq<int64_t>(0, numDevices), [&](int64_t devId) {
        return getShardIndex(devId, mesh, axes) * shardSize;
      });

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
        rewriter.getContext(), conversionState.getNextChannelId(),
        kChannelHandleType);
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

    // Previous passes guarantee that only sdy.collective_permute may have
    // different meshes in input and output sharding.
    MeshAttr outMesh = outSharding.getMesh(converter->getSymbolTable());
    MeshAttr inMesh = inSharding.getMesh(converter->getSymbolTable());
    if (!outMesh || !inMesh) {
      return op.emitOpError("failed to resolve mesh");
    }
    SDY_CHECK(inMesh.getTotalSize() == outMesh.getTotalSize())
        << "collective-permute between meshes of different sizes is not "
           "supported.";

    // Collective all AxisRef to produce an ordered list of all sharding axes.
    SmallVector<AxisRefAttr> allAxisRefs;
    for (TensorShardingAttr sharding : {inSharding, outSharding}) {
      if (sharding) {
        sharding.forEachAxisRef(
            [&](AxisRefAttr axis) { allAxisRefs.push_back(axis); });
      }
    }
    // Use outMesh as the reference for axis ordering.
    SmallVector<AxisRefAttr> allOrderedAxes = getOrderedAxisRefs(
        AxisRefListAttr::get(getContext(), allAxisRefs), outMesh);

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
    int64_t numDevices = outMesh.getTotalSize();
    llvm::DenseMap<int64_t, int64_t> logicalIdToOutDevId;
    for (int64_t j = 0; j < numDevices; ++j) {
      logicalIdToOutDevId[getShardIndex(j, outMesh, outShardingAxes)] = j;
    }

    SmallVector<AxisRefAttr> inShardingAxes = getShardingAxes(inSharding);
    SmallVector<int64_t> pairs;
    for (int64_t i = 0; i < numDevices; ++i) {
      pairs.push_back(i);
      pairs.push_back(
          logicalIdToOutDevId[getShardIndex(i, inMesh, inShardingAxes)]);
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
        op->getContext(), conversionState.getNextChannelId(),
        kChannelHandleType);
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
        op->getContext(), conversionState.getNextChannelId(),
        kChannelHandleType);

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
        op->getContext(), conversionState.getNextChannelId(),
        kChannelHandleType);
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

    // Prepare signature conversion for the internal region.
    TypeConverter::SignatureConversion signature(
        op.getBody().getNumArguments());
    for (auto [index, arg] : llvm::enumerate(op.getBody().getArguments())) {
      signature.addInputs(index, converter->convertType(arg));
    }

    auto newOp = NamedComputationOp::create(
        rewriter, op.getLoc(), converter->convertResultTypes(op.getResults()),
        op.getName(), adaptor.getOperands(),
        op.getInShardings().value_or(nullptr),
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

// Returns the group count through dividing the global group count by the size
// of the axes sharding the corresponding factor.
int64_t getLocalGroupCount(int64_t globalCount, ArrayRef<AxisRefAttr> axes,
                           MeshAttr mesh) {
  if (globalCount <= 1 || axes.empty()) {
    return globalCount;
  }
  int64_t totalAxesSize = getTotalAxesSize(axes, mesh);
  // The 'divisor' is the sharding size applied to globalCount.
  int64_t divisor = std::min(globalCount, totalAxesSize);
  SDY_CHECK(globalCount % divisor == 0)
      << "global group count is not divisible by total axes size";
  return globalCount / divisor;
}

class StablehloConvolutionOpPattern
    : public OpConversionPattern<stablehlo::ConvolutionOp> {
 public:
  StablehloConvolutionOpPattern(TypeConverter& converter, MLIRContext* ctx,
                                ConversionState& state)
      : OpConversionPattern<stablehlo::ConvolutionOp>(converter, ctx),
        conversionState(state) {}

  LogicalResult matchAndRewrite(
      stablehlo::ConvolutionOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());
    const SymbolTable& symbolTable = converter->getSymbolTable();

    TensorShardingAttr lhsSharding = converter->getSharding(op.getLhs());
    TensorShardingAttr rhsSharding = converter->getSharding(op.getRhs());
    TensorShardingAttr resultSharding = converter->getSharding(op.getResult());

    if (isFullyReplicated(lhsSharding) && isFullyReplicated(rhsSharding) &&
        isFullyReplicated(resultSharding)) {
      return localizeGenericOp(op, adaptor.getOperands(), rewriter, converter,
                               conversionState);
    }

    MeshAttr mesh;
    for (auto sharding : {lhsSharding, rhsSharding, resultSharding}) {
      if (sharding && (mesh = sharding.getMesh(symbolTable))) break;
    }
    if (!mesh) {
      return op.emitOpError("failed to resolve mesh");
    }

    // Verify that sharding of reduction axes are handled by an sdy.all_reduce.
    SmallVector<AxisRefAttr> reductionAxes;
    auto addRhsReductionAxes = [&](int64_t dim) {
      if (rhsSharding) {
        for (auto axis : rhsSharding.getDimShardings()[dim].getAxes()) {
          if (!llvm::is_contained(reductionAxes, axis)) {
            reductionAxes.push_back(axis);
          }
        }
      }
    };

    stablehlo::ConvDimensionNumbersAttr dimNums = op.getDimensionNumbers();
    // Check Kernel Input Feature dimension (Contracting).
    addRhsReductionAxes(dimNums.getKernelInputFeatureDimension());
    // Check Kernel Spatial dimensions (Contracting).
    for (int64_t spatialDim : dimNums.getKernelSpatialDimensions()) {
      addRhsReductionAxes(spatialDim);
    }

    if (!reductionAxes.empty()) {
      SDY_CHECK(op->hasOneUse() && isa<sdy::AllReduceOp>(*op->user_begin()))
          << "Expected sharded contracting convolution to have one user and "
             "the user is an sdy.all_reduce.";

      auto allReduceOp = cast<sdy::AllReduceOp>(*op->user_begin());
      sortAndMergeAxes(reductionAxes, mesh);
      SDY_CHECK(AxisRefListAttr::get(op->getContext(), reductionAxes) ==
                allReduceOp.getReductionAxesAttr())
          << "The axes of the sdy.all_reduce should match the sharded "
             "reduction dimensions of the convolution.";
    }

    auto getAxes = [&](TensorShardingAttr sharding, int64_t dim) {
      return (sharding && dim < sharding.getRank())
                 ? sharding.getDimShardings()[dim].getAxes()
                 : ArrayRef<AxisRefAttr>();
    };

    // feature_group_count constrains the input feature dimension (LHS) and the
    // kernel output feature dimension (RHS). We only look at the kernel
    // sharding here, as sdy-insert-explicit-reshards guarantees they are
    // sharded the same way.
    int64_t localFeatureGroupCount = getLocalGroupCount(
        op.getFeatureGroupCount(),
        getAxes(rhsSharding, dimNums.getKernelOutputFeatureDimension()), mesh);
    // batch_group_count constrains the input batch dimension (LHS) and the
    // kernel output feature dimension (RHS). We only inspect the LHS batch
    // dimension for the same reason.
    int64_t localBatchGroupCount = getLocalGroupCount(
        op.getBatchGroupCount(),
        getAxes(lhsSharding, dimNums.getInputBatchDimension()), mesh);

    rewriter.replaceOpWithNewOp<stablehlo::ConvolutionOp>(
        op, converter->convertType(op.getResult()), adaptor.getLhs(),
        adaptor.getRhs(), op.getWindowStridesAttr(), op.getPaddingAttr(),
        op.getLhsDilationAttr(), op.getRhsDilationAttr(),
        op.getWindowReversalAttr(), dimNums, localFeatureGroupCount,
        localBatchGroupCount, op.getPrecisionConfigAttr());

    conversionState.removeToConvertOp(op);
    return success();
  }

 private:
  ConversionState& conversionState;
};

bool NonPadOrPadWithZero(Value operand) {
  auto padOp = operand.getDefiningOp<stablehlo::PadOp>();
  if (!padOp) {
    return true;
  }
  DenseElementsAttr attr;
  bool isZero = false;
  if (matchPattern(padOp.getPaddingValue(), m_Constant(&attr)) &&
      attr.isSplat()) {
    Attribute splatValue = attr.getSplatValue<Attribute>();
    if (auto floatAttr = dyn_cast<FloatAttr>(splatValue)) {
      isZero = floatAttr.getValue().isZero();
    } else if (auto intAttr = dyn_cast<IntegerAttr>(splatValue)) {
      isZero = intAttr.getValue().isZero();
    }
  }
  return isZero;
}

class StablehloDotOpPattern : public OpConversionPattern<stablehlo::DotOp> {
 public:
  StablehloDotOpPattern(TypeConverter& converter, MLIRContext* ctx,
                        ConversionState& state)
      : OpConversionPattern<stablehlo::DotOp>(converter, ctx),
        conversionState(state) {}

  LogicalResult matchAndRewrite(
      stablehlo::DotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());
    Value lhs = op.getLhs();
    TensorShardingAttr lhsSharding = converter->getSharding(lhs);

    if (!isFullyReplicated(lhsSharding)) {
      const SymbolTable& symbolTable = converter->getSymbolTable();
      MeshAttr mesh = lhsSharding.getMesh(symbolTable);
      if (!mesh) {
        return op.emitOpError("failed to resolve mesh.");
      }

      // Identify the sharded contracting dimension axes in the LHS, which is
      // the last dimension.
      auto lhsType = cast<RankedTensorType>(lhs.getType());
      int64_t lhsContractingDim = lhsType.getRank() - 1;
      ArrayRef<AxisRefAttr> contractingAxes =
          lhsSharding.getDimShardings()[lhsContractingDim].getAxes();

      if (!contractingAxes.empty()) {
        int64_t totalAxesSize = getTotalAxesSize(contractingAxes, mesh);

        // Check divisibility of sharded contracting dimensions.
        SDY_CHECK(lhsType.getDimSize(lhsContractingDim) % totalAxesSize == 0)
            << "Sharded contracting dimension must be divisible by the total "
               "size "
               "of the axes sharding it.";
        auto checkPadding = [](Value operand) {
          SDY_CHECK(NonPadOrPadWithZero(operand))
              << "Padding value must be zero for sharded contracting dot.";
        };
        checkPadding(lhs);
        checkPadding(op.getRhs());

        // Verify that an AllReduce has been inserted by a previous pass.
        // The local partial sums must be combined.
        SDY_CHECK(op->hasOneUse() && isa<sdy::AllReduceOp>(*op->user_begin()))
            << "Expected sharded contracting dot to have one user and the user "
               "is "
               "an sdy.all_reduce.";

        auto allReduceOp = cast<sdy::AllReduceOp>(*op->user_begin());
        SDY_CHECK(allReduceOp.getReductionAxesAttr() ==
                  AxisRefListAttr::get(op->getContext(), contractingAxes))
            << "The axes of the sdy.all_reduce should match the sharded "
               "contracting dimension of the dot.";
      }
    }

    return localizeGenericOp(op, adaptor.getOperands(), rewriter, converter,
                             conversionState);
  }

 private:
  ConversionState& conversionState;
};

class StablehloDotGeneralOpPattern
    : public OpConversionPattern<stablehlo::DotGeneralOp> {
 public:
  StablehloDotGeneralOpPattern(TypeConverter& converter, MLIRContext* ctx,
                               ConversionState& state)
      : OpConversionPattern<stablehlo::DotGeneralOp>(converter, ctx),
        conversionState(state) {}

  LogicalResult matchAndRewrite(
      stablehlo::DotGeneralOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());

    Value lhs = op.getLhs();
    TensorShardingAttr lhsSharding = converter->getSharding(lhs);

    if (!isFullyReplicated(lhsSharding)) {
      const SymbolTable& symbolTable = converter->getSymbolTable();
      MeshAttr mesh = lhsSharding.getMesh(symbolTable);
      if (!mesh) {
        return op.emitError("failed to resolve mesh.");
      }

      // Identify sharded contracting dimension axes in the LHS.
      stablehlo::DotDimensionNumbersAttr dimNums = op.getDotDimensionNumbers();
      auto lhsType = cast<RankedTensorType>(lhs.getType());
      SmallVector<AxisRefAttr> reductionAxes;
      for (int64_t lhsDim : dimNums.getLhsContractingDimensions()) {
        ArrayRef<AxisRefAttr> axes =
            lhsSharding.getDimShardings()[lhsDim].getAxes();
        if (axes.empty()) {
          continue;
        }
        int64_t totalAxesSize = getTotalAxesSize(axes, mesh);
        SDY_CHECK(lhsType.getDimSize(lhsDim) % totalAxesSize == 0)
            << "Sharded contracting dimension must be divisible by axes size.";
        reductionAxes.append(axes.begin(), axes.end());
      }

      if (!reductionAxes.empty()) {
        auto checkPadding = [](Value operand) {
          SDY_CHECK(NonPadOrPadWithZero(operand))
              << "Padding value must be zero for sharded contracting "
                 "dot_general.";
        };
        checkPadding(lhs);
        checkPadding(op.getRhs());

        // Verify that an AllReduce has been inserted by a previous pass.
        SDY_CHECK(op->hasOneUse() && isa<sdy::AllReduceOp>(*op->user_begin()))
            << "Expected sharded contracting dot_general to have one user and "
               "the user is an sdy.all_reduce.";

        auto allReduceOp = cast<sdy::AllReduceOp>(*op->user_begin());
        SDY_CHECK(allReduceOp.getReductionAxesAttr() ==
                  AxisRefListAttr::get(op->getContext(), reductionAxes))
            << "The axes of the sdy.all_reduce should match the sharded "
               "contracting dimensions of the dot_general.";
      }
    }

    return localizeGenericOp(op, adaptor.getOperands(), rewriter, converter,
                             conversionState);
  }

 private:
  ConversionState& conversionState;
};

SmallVector<int64_t> computeLocalSliceSizes(stablehlo::GatherOp op,
                                            TensorShardingAttr operandSharding,
                                            MeshAttr mesh) {
  SmallVector<int64_t> localSliceSizes = llvm::to_vector(op.getSliceSizes());
  if (isFullyReplicated(operandSharding)) {
    return localSliceSizes;
  }
  auto globalOperandType = cast<RankedTensorType>(op.getOperand().getType());
  auto operandBatchingDims = op.getDimensionNumbers().getOperandBatchingDims();

  for (int64_t i = 0; i < localSliceSizes.size(); ++i) {
    // Skip operand batching dimensions and trivial slice dimensions.
    if (localSliceSizes[i] == 1 || llvm::is_contained(operandBatchingDims, i)) {
      continue;
    }

    ArrayRef<AxisRefAttr> axes = operandSharding.getDimShardings()[i].getAxes();
    if (axes.empty()) {
      continue;
    }

    // It should be a full slice dimension with size divisible by the total axes
    // size.
    int64_t totalAxesSize = getTotalAxesSize(axes, mesh);
    SDY_CHECK(localSliceSizes[i] == globalOperandType.getDimSize(i) &&
              (localSliceSizes[i] % totalAxesSize) == 0);
    localSliceSizes[i] /= totalAxesSize;
  }
  return localSliceSizes;
}

// Returns the min and max for the indices in a scatter/gather:
//
// Is a indexed dim is not a trivial slice dimension, the bounds are (0,
// global_dim_size - 1).
// Otherwise,  the bounds are (offset, offset + shard_size - 1).
std::pair<SmallVector<Value>, SmallVector<Value>> getTrivialSliceDimBounds(
    Location loc, MeshAttr mesh, TensorShardingAttr operandSharding,
    RankedTensorType globalOperandType, RankedTensorType localOperandType,
    ArrayRef<int64_t> startIndexMap, ArrayRef<int64_t> trivialSliceDims,
    Type indexEltTy, ConversionPatternRewriter& rewriter, bool computeMask) {
  SmallVector<Value> minBounds, maxBounds;
  minBounds.reserve(startIndexMap.size());
  if (computeMask) {
    maxBounds.reserve(startIndexMap.size());
  }

  for (int64_t indexedDim : startIndexMap) {
    if (!llvm::is_contained(trivialSliceDims, indexedDim)) {
      // For non-trivial dims, the bounds cover the entire global range.
      minBounds.push_back(stablehlo::ConstantOp::create(
          rewriter, loc,
          DenseIntElementsAttr::get(RankedTensorType::get({}, indexEltTy),
                                    (int64_t)0)));
      if (computeMask) {
        maxBounds.push_back(stablehlo::ConstantOp::create(
            rewriter, loc,
            DenseIntElementsAttr::get(
                RankedTensorType::get({}, indexEltTy),
                globalOperandType.getDimSize(indexedDim) - 1)));
      }
      continue;
    }
    ArrayRef<AxisRefAttr> axes =
        operandSharding.getDimShardings()[indexedDim].getAxes();
    int64_t shardSize = localOperandType.getDimSize(indexedDim);
    Value offset = getDimensionOffset(loc, mesh, axes, shardSize, rewriter);
    // Ensure offset matches the index element type (e.g., i32 or i64).
    offset = stablehlo::ConvertOp::create(
        rewriter, loc, RankedTensorType::get({}, indexEltTy), offset);

    minBounds.push_back(offset);
    if (computeMask) {
      Value shardSizeMinus1 = stablehlo::ConstantOp::create(
          rewriter, loc,
          DenseIntElementsAttr::get(RankedTensorType::get({}, indexEltTy),
                                    shardSize - 1));
      maxBounds.push_back(stablehlo::AddOp::create(
          rewriter, loc, RankedTensorType::get({}, indexEltTy), offset,
          shardSizeMinus1));
    }
  }

  return {minBounds, maxBounds};
}

// Creates a global clamp to [0, global_dim_size - 1] for all indexed dims.
Value clampGatherIndices(Location loc, Value indices,
                         RankedTensorType globalOperandType,
                         ArrayRef<int64_t> startIndexMap, int64_t ivd,
                         ConversionPatternRewriter& rewriter) {
  auto indicesType = cast<RankedTensorType>(indices.getType());
  Type indexEltTy = indicesType.getElementType();

  auto buildIndicesLikeTensor = [&](ArrayRef<int64_t> values) -> Value {
    if (ivd < indicesType.getRank() && values.size() > 1) {
      auto attr = DenseIntElementsAttr::get(
          RankedTensorType::get(values.size(), indexEltTy), values);
      Value valuesConst = stablehlo::ConstantOp::create(rewriter, loc, attr);
      return stablehlo::BroadcastInDimOp::create(
          rewriter, loc, indicesType, valuesConst,
          rewriter.getDenseI64ArrayAttr({ivd}));
    }
    return stablehlo::ConstantOp::create(
        rewriter, loc, DenseIntElementsAttr::get(indicesType, values[0]));
  };

  SmallVector<int64_t> globalMaxValues =
      llvm::map_to_vector(startIndexMap, [&](int64_t indexedDim) {
        return globalOperandType.getDimSize(indexedDim) - 1;
      });

  return stablehlo::ClampOp::create(rewriter, loc, indicesType,
                                    buildIndicesLikeTensor({0}), indices,
                                    buildIndicesLikeTensor(globalMaxValues));
}

// Computes the local indices and mask for a scatter or gather op with trivial
// slice dimensions.
//
// If there is any trivial slice dimension are in start_index_map, we need to
// adjust the start_indices to be local and compute a mask based on whether the
// original indices are in the local shard.
//
// If there is any trivial slice dimension is not in start_index_map, we need to
// compute a mask based on whether the global offset of the shard is 0.
template <typename DimNumbersOp>
std::pair<Value, Value> computeLocalIndicesAndMask(
    Location loc, Value indices, MeshAttr mesh,
    TensorShardingAttr operandSharding, RankedTensorType globalOperandType,
    RankedTensorType localOperandType, DimNumbersOp dimNumbers,
    ArrayRef<int64_t> trivialSliceDims, ConversionPatternRewriter& rewriter,
    bool computeMask = true) {
  int64_t ivd = dimNumbers.getIndexVectorDim();
  ArrayRef<int64_t> startIndexMap;
  if constexpr (std::is_same_v<DimNumbersOp,
                               stablehlo::ScatterDimensionNumbersAttr>) {
    startIndexMap = dimNumbers.getScatterDimsToOperandDims();
  } else {
    startIndexMap = dimNumbers.getStartIndexMap();
  }
  auto indicesType = cast<RankedTensorType>(indices.getType());
  Type indexEltTy = indicesType.getElementType();
  Value adjustedIndices = indices;
  Value mask = nullptr;

  bool needIndicesBasedPred = llvm::any_of(trivialSliceDims, [&](int64_t d) {
    return llvm::is_contained(startIndexMap, d);
  });
  if (needIndicesBasedPred) {
    Value baseIndices;
    if constexpr (std::is_same_v<DimNumbersOp,
                                 stablehlo::ScatterDimensionNumbersAttr>) {
      // Use original unclamped indices so OOB indices stay OOB locally.
      baseIndices = indices;
    } else {
      // Use globally clamped indices for gather semantics.
      baseIndices = clampGatherIndices(loc, indices, globalOperandType,
                                       startIndexMap, ivd, rewriter);
    }
    auto [minBounds, maxBounds] = getTrivialSliceDimBounds(
        loc, mesh, operandSharding, globalOperandType, localOperandType,
        startIndexMap, trivialSliceDims, indexEltTy, rewriter, computeMask);

    // Builds bounds tensors matching the indices tensor type.
    auto buildBoundsTensor = [&](ArrayRef<Value> bounds) -> Value {
      if (ivd >= indicesType.getRank() || indicesType.getDimSize(ivd) == 1) {
        // Scalar or single-element vector: uses the direct broadcast.
        return stablehlo::BroadcastInDimOp::create(
            rewriter, loc, indicesType, bounds[0],
            rewriter.getDenseI64ArrayAttr({}));
      }

      // Multi-element bound values, reshape, concatenate, then broadcast.
      SmallVector<Value> reshaped =
          llvm::map_to_vector(bounds, [&](Value b) -> Value {
            return stablehlo::ReshapeOp::create(
                       rewriter, loc, RankedTensorType::get({1}, indexEltTy), b)
                .getResult();
          });
      Value concat = stablehlo::ConcatenateOp::create(
          rewriter, loc,
          RankedTensorType::get({(int64_t)bounds.size()}, indexEltTy), reshaped,
          0);
      return stablehlo::BroadcastInDimOp::create(
          rewriter, loc, indicesType, concat,
          rewriter.getDenseI64ArrayAttr({ivd}));
    };

    Value indicesMin = buildBoundsTensor(minBounds);
    Value indicesMax = computeMask ? buildBoundsTensor(maxBounds) : nullptr;

    adjustedIndices =
        stablehlo::SubtractOp::create(rewriter, loc, baseIndices, indicesMin);

    if (computeMask) {
      Value ge =
          stablehlo::CompareOp::create(rewriter, loc, baseIndices, indicesMin,
                                       stablehlo::ComparisonDirection::GE);
      Value le =
          stablehlo::CompareOp::create(rewriter, loc, baseIndices, indicesMax,
                                       stablehlo::ComparisonDirection::LE);
      mask = stablehlo::AndOp::create(rewriter, loc, ge, le);
    }
  }

  if (!computeMask) {
    return {adjustedIndices, nullptr};
  }

  bool needPartitionBasedPred = llvm::any_of(trivialSliceDims, [&](int64_t d) {
    return !llvm::is_contained(startIndexMap, d);
  });
  if (needPartitionBasedPred) {
    // For unindexed sharded dims, the shard is only valid if global offset 0 is
    // in-shard. This is equivalent to checking if the shard's global dimension
    // offset is 0.
    Value partitionMask = nullptr;

    for (int64_t dim : trivialSliceDims) {
      if (!llvm::is_contained(startIndexMap, dim)) {
        ArrayRef<AxisRefAttr> axes =
            operandSharding.getDimShardings()[dim].getAxes();
        int64_t shardSize = localOperandType.getDimSize(dim);
        Value offset = getDimensionOffset(loc, mesh, axes, shardSize, rewriter);
        Value zero = stablehlo::ConstantOp::create(
            rewriter, loc,
            DenseIntElementsAttr::get(cast<RankedTensorType>(offset.getType()),
                                      (int64_t)0));
        Value eqZero = stablehlo::CompareOp::create(
            rewriter, loc, offset, zero, stablehlo::ComparisonDirection::EQ);
        partitionMask =
            partitionMask
                ? stablehlo::AndOp::create(rewriter, loc, partitionMask, eqZero)
                : eqZero;
      }
    }
    if (mask) {
      // Broadcast scalar partitionMask to match the rank of indices-based mask.
      partitionMask = stablehlo::BroadcastInDimOp::create(
          rewriter, loc, mask.getType(), partitionMask,
          rewriter.getDenseI64ArrayAttr({}));
      mask = stablehlo::AndOp::create(rewriter, loc, mask, partitionMask);
    } else {
      mask = partitionMask;
    }
  }

  return {adjustedIndices, mask};
}

class StablehloGatherOpPattern
    : public OpConversionPattern<stablehlo::GatherOp> {
 public:
  StablehloGatherOpPattern(TypeConverter& converter, MLIRContext* ctx,
                           ConversionState& state)
      : OpConversionPattern<stablehlo::GatherOp>(converter, ctx),
        conversionState(state) {}

  LogicalResult matchAndRewrite(
      stablehlo::GatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());
    TensorShardingAttr operandSharding =
        converter->getSharding(op.getOperand());

    if (isFullyReplicated(operandSharding)) {
      return localizeGenericOp(op, adaptor.getOperands(), rewriter, converter,
                               conversionState);
    }

    MeshAttr mesh = operandSharding.getMesh(converter->getSymbolTable());
    if (!mesh) {
      return op.emitOpError("failed to resolve mesh");
    }

    // Identify trivial slice dimensions (sharded, non-batching, slice size 1).
    auto globalOperandType = cast<RankedTensorType>(op.getOperand().getType());
    stablehlo::GatherDimensionNumbersAttr dimNumbers = op.getDimensionNumbers();
    ArrayRef<int64_t> sliceSizes = op.getSliceSizes();
    SmallVector<int64_t> trivialSliceDims;
    SmallVector<AxisRefAttr> reductionAxes;
    for (int64_t i = 0; i < globalOperandType.getRank(); ++i) {
      auto axes = operandSharding.getDimShardings()[i].getAxes();
      if (!axes.empty() && getTotalAxesSize(axes, mesh) > 1 &&
          sliceSizes[i] == 1 &&
          !llvm::is_contained(dimNumbers.getOperandBatchingDims(), i)) {
        trivialSliceDims.push_back(i);
        reductionAxes.append(axes.begin(), axes.end());
      }
    }

    SmallVector<int64_t> localSliceSizes =
        computeLocalSliceSizes(op, operandSharding, mesh);
    SmallVector<StringRef> attrsToExclude = {"dimension_numbers", "slice_sizes",
                                             "indices_are_sorted"};
    Location loc = op.getLoc();
    if (trivialSliceDims.empty()) {
      // No trivial slice dimensions. We only need to adjust the slice sizes
      // for sharded dimensions and construct the local gather.
      Value result = stablehlo::GatherOp::create(
          rewriter, loc, converter->convertType(op.getResult()),
          adaptor.getOperand(), adaptor.getStartIndices(), dimNumbers,
          rewriter.getDenseI64ArrayAttr(localSliceSizes),
          op.getIndicesAreSorted());
      copyAttributes(op, result.getDefiningOp(), attrsToExclude);
      rewriter.replaceOp(op, result);
      conversionState.removeToConvertOp(op);
      return success();
    }

    // There are trivial slice dimensions. We need to adjust the start_indices,
    // and compute a mask.
    auto [adjustedIndices, mask] = computeLocalIndicesAndMask(
        loc, adaptor.getStartIndices(), mesh, operandSharding,
        globalOperandType,
        cast<RankedTensorType>(adaptor.getOperand().getType()), dimNumbers,
        trivialSliceDims, rewriter);
    Value result = stablehlo::GatherOp::create(
        rewriter, loc, converter->convertType(op.getResult()),
        adaptor.getOperand(), adjustedIndices, dimNumbers,
        rewriter.getDenseI64ArrayAttr(localSliceSizes),
        op.getIndicesAreSorted());
    copyAttributes(op, result.getDefiningOp(), attrsToExclude);

    // Reduce the mask along the index_vector_dim.
    auto maskType = cast<RankedTensorType>(mask.getType());
    int64_t ivd = dimNumbers.getIndexVectorDim();
    if (maskType.getRank() > 0 && ivd < maskType.getRank()) {
      SmallVector<int64_t> reducedShape = llvm::to_vector(maskType.getShape());
      reducedShape.erase(reducedShape.begin() + ivd);

      auto reducedMaskType =
          RankedTensorType::get(reducedShape, rewriter.getI1Type());
      Value initValue = stablehlo::ConstantOp::create(
          rewriter, loc,
          DenseIntElementsAttr::get(
              RankedTensorType::get({}, rewriter.getI1Type()), true));
      auto reduceOp = stablehlo::ReduceOp::create(
          rewriter, loc, reducedMaskType, mask, initValue,
          rewriter.getDenseI64ArrayAttr({ivd}));
      stablehlo::buildReduceBody<stablehlo::AndOp>(
          rewriter.getI1Type(), reduceOp.getOperation()->getRegion(0),
          rewriter);

      mask = reduceOp.getResult(0);
      maskType = cast<RankedTensorType>(mask.getType());
    }

    // Broadcast the mask to the result shape. We need to determine the
    // broadcast dimensions based on the mask rank.
    RankedTensorType resultType = cast<RankedTensorType>(result.getType());
    DenseI64ArrayAttr bcastDims;
    if (maskType.getRank() == 0) {
      // The mask is a scalar.
      bcastDims = rewriter.getDenseI64ArrayAttr({});
    } else {
      // The mask has a shape containing all the batching dims.
      SmallVector<int64_t> gatherBatchDims;
      for (int64_t i = 0; i < resultType.getRank(); ++i) {
        if (!llvm::is_contained(dimNumbers.getOffsetDims(), i)) {
          gatherBatchDims.push_back(i);
        }
      }
      bcastDims = rewriter.getDenseI64ArrayAttr(gatherBatchDims);
    }
    Value bcastMask = stablehlo::BroadcastInDimOp::create(
        rewriter, loc,
        RankedTensorType::get(resultType.getShape(), rewriter.getI1Type()),
        mask, bcastDims);

    // Zero out the result where the mask is false (index was out of bounds
    // for the shard).
    Value zero = stablehlo::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(
            resultType, rewriter.getZeroAttr(resultType.getElementType())));
    result =
        stablehlo::SelectOp::create(rewriter, loc, bcastMask, result, zero);

    // Verify that an AllReduce has been inserted, such as by
    // insert-explicit-reshards pass.
    auto allReduceOp = cast<sdy::AllReduceOp>(*op->user_begin());
    SDY_CHECK(op->hasOneUse() && allReduceOp)
        << "Expected the gather to have one user and the user is an "
           "sdy.all_reduce.";
    SDY_CHECK(allReduceOp.getReductionAxesAttr() ==
              AxisRefListAttr::get(op->getContext(), reductionAxes))
        << "The axes of the sdy.all_reduce should match the sharded collapsed "
           "operand dimensions of the gather.";

    rewriter.replaceOp(op, result);
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
    Value broadcastOffset = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, localType, offsetConverted,
        rewriter.getDenseI64ArrayAttr({}));
    rewriter.replaceOpWithNewOp<stablehlo::AddOp>(op, localIota,
                                                  broadcastOffset);

    conversionState.removeToConvertOp(op);
    return success();
  }

 private:
  ConversionState& conversionState;
};

class StablehloPadOpPattern : public OpConversionPattern<stablehlo::PadOp> {
 public:
  StablehloPadOpPattern(TypeConverter& converter, MLIRContext* ctx,
                        ConversionState& state)
      : OpConversionPattern<stablehlo::PadOp>(converter, ctx),
        conversionState(state) {}

  LogicalResult matchAndRewrite(
      stablehlo::PadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());
    TensorShardingAttr inSharding = converter->getSharding(op.getOperand());
    TensorShardingAttr outSharding = converter->getSharding(op.getResult());
    if (isFullyReplicated(inSharding)) {
      // If input is fully replicated, output must also be fully replicated or
      // else the pad op should have been converted to a pad followed by a
      // collective to reshard its result.
      SDY_CHECK(isFullyReplicated(outSharding));
      return localizeGenericOp(op, adaptor.getOperands(), rewriter, converter,
                               conversionState);
    }
    const SymbolTable& symbolTable = converter->getSymbolTable();
    MeshAttr mesh = inSharding.getMesh(symbolTable);
    if (!mesh) {
      return op.emitOpError("failed to resolve mesh");
    }

    auto globalInputType = cast<RankedTensorType>(op.getOperand().getType());
    auto globalResultType = cast<RankedTensorType>(op.getType());
    int64_t rank = globalInputType.getRank();
    int64_t numDevices = mesh.getTotalSize();
    ArrayRef<int64_t> edgePaddingLow = op.getEdgePaddingLow();
    ArrayRef<int64_t> interiorPaddingAttr = op.getInteriorPadding();
    // Calculate device-specific local offsets and check for uniformity.
    SmallVector<SmallVector<int64_t>> allOffsets(
        rank, SmallVector<int64_t>(numDevices));
    // Whether all devices have the same offset for a given dimension.
    BitVector isDimUniform(rank, true);
    for (int64_t i = 0; i < rank; ++i) {
      ArrayRef<AxisRefAttr> inAxes = inSharding.getDimShardings()[i].getAxes();
      ArrayRef<AxisRefAttr> outAxes =
          outSharding.getDimShardings()[i].getAxes();
      SDY_CHECK(inAxes == outAxes)
          << "dimension " << i << " has mismatched sharding axes ("
          << strippedAttrsString(inAxes) << " vs "
          << strippedAttrsString(outAxes)
          << "). This requires a reshard collective which should have been "
             "handled "
          << "by previous passes.";

      int64_t pLow = edgePaddingLow[i];
      int64_t pInt = interiorPaddingAttr[i];
      int64_t totalAxesSize = getTotalAxesSize(inAxes, mesh);
      // The input slice size per device.
      int64_t sIn =
          globalInputType.getDimSize(i) / (inAxes.empty() ? 1 : totalAxesSize);
      // The output slice size per device.
      int64_t sOut = globalResultType.getDimSize(i) /
                     (outAxes.empty() ? 1 : totalAxesSize);

      for (int64_t devId = 0; devId < numDevices; ++devId) {
        int64_t kIn = getShardIndex(devId, mesh, inAxes);
        int64_t kOut = getShardIndex(devId, mesh, outAxes);
        // offset = (GlobalStartOfInputData)−(GlobalStartOfOutputShard)
        allOffsets[i][devId] = (kIn * sIn) * (pInt + 1) + pLow - (kOut * sOut);
        if (allOffsets[i][devId] != allOffsets[i][0]) {
          isDimUniform[i] = false;
        }
      }
    }

    Location loc = op.getLoc();
    auto localResultType =
        cast<RankedTensorType>(typeConverter->convertType(op.getResult()));
    auto localInput = adaptor.getOperand();
    auto localInputType = cast<RankedTensorType>(localInput.getType());
    // If ALL dimensions are uniform, use a stablehlo.pad.
    if (isDimUniform.all()) {
      SmallVector<int64_t> localLow, localHigh;
      localLow.reserve(rank);
      localHigh.reserve(rank);
      for (int64_t i = 0; i < rank; ++i) {
        int64_t lLow = allOffsets[i][0];
        localLow.push_back(lLow);
        int64_t sInLocal = localInputType.getDimSize(i);
        int64_t expandedInputSize =
            (sInLocal - 1) * (interiorPaddingAttr[i] + 1) + 1;
        localHigh.push_back(localResultType.getDimSize(i) -
                            (expandedInputSize + lLow));
      }
      rewriter.replaceOp(
          op, stablehlo::PadOp::create(rewriter, loc, localResultType,
                                       localInput, adaptor.getPaddingValue(),
                                       localLow, localHigh, interiorPaddingAttr)
                  .getResult());
      conversionState.removeToConvertOp(op);
      return success();
    }

    // For non-uniform cases, we pad the input enough to cover all possible
    // shifts to produce a safePad, then slice the localResultType piece from
    // the safePad.
    SmallVector<int64_t> kLow, kHigh, safePadShape;
    kLow.reserve(rank);
    kHigh.reserve(rank);
    safePadShape.reserve(rank);
    for (int64_t i = 0; i < rank; ++i) {
      int64_t maxO = *llvm::max_element(allOffsets[i]);
      int64_t minO = *llvm::min_element(allOffsets[i]);
      int64_t sInLocal = localInputType.getDimSize(i);
      int64_t sInExpanded = (sInLocal - 1) * (interiorPaddingAttr[i] + 1) + 1;

      // We pad the input on the left by maxO so that the slice start (maxO -
      // offset) is always >= 0.
      int64_t lowPad = std::max<int64_t>(0, maxO);
      kLow.push_back(lowPad);

      // We pad on the right so that the slice (size localResultSize) is always
      // in bounds.
      int64_t highPad = std::max<int64_t>(
          0, localResultType.getDimSize(i) - minO - sInExpanded);
      kHigh.push_back(highPad);
      safePadShape.push_back(lowPad + sInExpanded + highPad);
    }

    auto safePadType =
        RankedTensorType::get(safePadShape, localInputType.getElementType());
    Value safePad = stablehlo::PadOp::create(
        rewriter, loc, safePadType, localInput, adaptor.getPaddingValue(), kLow,
        kHigh, interiorPaddingAttr);

    Value partitionId = stablehlo::ConvertOp::create(
        rewriter, loc, RankedTensorType::get({}, rewriter.getI64Type()),
        stablehlo::PartitionIdOp::create(rewriter, loc));

    SmallVector<Value> sliceOffsets;
    sliceOffsets.reserve(rank);
    for (int64_t i = 0; i < rank; ++i) {
      // The relative offset in the safePad is (kLow[i] - O_local(k)).
      SmallVector<int64_t> sliceTable = llvm::map_to_vector(
          allOffsets[i], [&](int64_t o) { return kLow[i] - o; });

      auto getTableOffset = [&](ArrayRef<int64_t> tableData,
                                bool uniform) -> Value {
        if (uniform) {
          return stablehlo::ConstantOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(tableData[0]));
        }
        auto tableType =
            RankedTensorType::get({numDevices}, rewriter.getI64Type());
        auto table = stablehlo::ConstantOp::create(
            rewriter, loc, tableType,
            DenseIntElementsAttr::get(tableType, tableData));
        auto slice = stablehlo::DynamicSliceOp::create(
            rewriter, loc, RankedTensorType::get({1}, rewriter.getI64Type()),
            table, {partitionId}, rewriter.getDenseI64ArrayAttr({1}));
        return stablehlo::ReshapeOp::create(
                   rewriter, loc,
                   RankedTensorType::get({}, rewriter.getI64Type()), slice)
            .getResult();
      };

      sliceOffsets.push_back(getTableOffset(sliceTable, isDimUniform[i]));
    }

    // Extract the final local result from safePad.
    rewriter.replaceOp(op, stablehlo::DynamicSliceOp::create(
                               rewriter, loc, localResultType, safePad,
                               sliceOffsets, localResultType.getShape())
                               .getResult());
    conversionState.removeToConvertOp(op);
    return success();
  }

 private:
  ConversionState& conversionState;
};

// Collects axes sharding implicit batch dims in updates.
SmallVector<AxisRefAttr> getScatterReductionAxes(
    stablehlo::ScatterOp op, TensorShardingAttr updatesSharding,
    TensorShardingAttr operandSharding, MeshAttr mesh) {
  SmallVector<AxisRefAttr> reductionAxes;
  if (isFullyReplicated(updatesSharding)) {
    return reductionAxes;
  }
  stablehlo::ScatterDimensionNumbersAttr dnums =
      op.getScatterDimensionNumbers();
  ArrayRef<int64_t> updateWinDims = dnums.getUpdateWindowDims();

  // Implicit/explicit batch dimensions in updates are those not in
  // update_window_dims.
  auto updateType = cast<RankedTensorType>(op.getUpdates().front().getType());
  for (int64_t i = 0; i < updateType.getRank(); ++i) {
    if (llvm::is_contained(updateWinDims, i)) {
      continue;
    }

    // Collective sharding axes on implicit batch dimensions.
    for (AxisRefAttr axis : updatesSharding.getDimShardings()[i].getAxes()) {
      // If this axis is also used to shard the operand, it is an explicit not
      // implicit batching dimension.
      bool shardedInOperand =
          !isFullyReplicated(operandSharding) &&
          llvm::any_of(operandSharding.getDimShardings(),
                       [&](DimensionShardingAttr dimSharding) {
                         return llvm::is_contained(dimSharding.getAxes(), axis);
                       });

      if (!shardedInOperand) {
        reductionAxes.push_back(axis);
      }
    }
  }

  return reductionAxes;
}

// Returns the identity constant (as a scalar) for the scatter reduction.
Value getScatterReductionIdentity(stablehlo::ScatterOp scatter, OpBuilder& b) {
  Operation* reductionOp = getCommonSupportedReductionOp(scatter);
  if (!reductionOp) {
    return nullptr;
  }

  Location loc = scatter.getLoc();
  Type type = scatter.getInputs().front().getType();
  Type elementType = cast<RankedTensorType>(type).getElementType();
  auto scalarType = RankedTensorType::get({}, elementType);

  return llvm::TypeSwitch<Operation*, Value>(reductionOp)
      .Case([&](stablehlo::AddOp) {
        return stablehlo::ConstantOp::create(b, loc, b.getZeroAttr(scalarType));
      })
      .Case([&](stablehlo::AndOp) {
        return stablehlo::ConstantOp::create(
            b, loc,
            DenseElementsAttr::get(scalarType,
                                   b.getIntegerAttr(elementType, 1)));
      })
      .Case([&](stablehlo::OrOp) {
        return stablehlo::ConstantOp::create(b, loc, b.getZeroAttr(scalarType));
      })
      .Case([&](stablehlo::MulOp) {
        if (isa<FloatType>(elementType)) {
          return stablehlo::ConstantOp::create(
              b, loc,
              DenseElementsAttr::get(scalarType,
                                     b.getFloatAttr(elementType, 1.0)));
        }
        return stablehlo::ConstantOp::create(
            b, loc,
            DenseElementsAttr::get(scalarType,
                                   b.getIntegerAttr(elementType, 1)));
      })
      .Case([&](stablehlo::MaxOp) {
        Attribute minAttr;
        if (auto floatType = dyn_cast<FloatType>(elementType)) {
          auto minFloat = APFloat::getLargest(floatType.getFloatSemantics(),
                                              /*Negative=*/true);
          minAttr = DenseElementsAttr::get(scalarType, minFloat);
        } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
          auto minInt = APInt::getSignedMinValue(intType.getWidth());
          minAttr = DenseElementsAttr::get(scalarType, minInt);
        }
        return minAttr ? stablehlo::ConstantOp::create(b, loc, minAttr)
                       : nullptr;
      })
      .Case([&](stablehlo::MinOp) {
        Attribute maxAttr;
        if (auto floatType = dyn_cast<FloatType>(elementType)) {
          auto maxFloat = APFloat::getLargest(floatType.getFloatSemantics(),
                                              /*Negative=*/false);
          maxAttr = DenseElementsAttr::get(scalarType, maxFloat);
        } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
          auto maxInt = APInt::getSignedMaxValue(intType.getWidth());
          maxAttr = DenseElementsAttr::get(scalarType, maxInt);
        }
        return maxAttr ? stablehlo::ConstantOp::create(b, loc, maxAttr)
                       : nullptr;
      })
      .Default([](auto) { return nullptr; });
}

class StablehloScatterOpPattern
    : public OpConversionPattern<stablehlo::ScatterOp> {
 public:
  StablehloScatterOpPattern(TypeConverter& converter, MLIRContext* ctx,
                            ConversionState& state)
      : OpConversionPattern<stablehlo::ScatterOp>(converter, ctx),
        conversionState(state) {}

  LogicalResult matchAndRewrite(
      stablehlo::ScatterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());
    Value input = op.getInputs().front();
    Value update = op.getUpdates().front();
    TensorShardingAttr inputSharding = converter->getSharding(input);
    TensorShardingAttr updateSharding = converter->getSharding(update);
    if (isFullyReplicated(inputSharding) && isFullyReplicated(updateSharding)) {
      return localizeGenericOp(op, adaptor.getOperands(), rewriter, converter,
                               conversionState);
    }

    const SymbolTable& symbolTable = converter->getSymbolTable();
    MeshAttr mesh = inputSharding ? inputSharding.getMesh(symbolTable)
                                  : updateSharding.getMesh(symbolTable);
    if (!mesh) {
      return op.emitError("failed to resolve mesh.");
    }

    SmallVector<AxisRefAttr> reductionAxes =
        getScatterReductionAxes(op, updateSharding, inputSharding, mesh);
    if (reductionAxes.empty() && isFullyReplicated(inputSharding)) {
      return localizeGenericOp(op, adaptor.getOperands(), rewriter, converter,
                               conversionState);
    }

    Location loc = op.getLoc();
    SmallVector<Value> localInputs = adaptor.getInputs();
    auto localInputType = cast<RankedTensorType>(localInputs.front().getType());
    if (!reductionAxes.empty()) {
      Value identity = getScatterReductionIdentity(op, rewriter);
      if (!identity) {
        return op.emitError("failed to get scatter reduction identity.");
      }
      // Generate leader mask: (shardIndex(reductionAxes) == 0).
      int64_t numDevices = mesh.getTotalSize();
      SmallVector<bool> leaderTable = llvm::map_to_vector(
          llvm::seq<int64_t>(0, numDevices), [&](int64_t devId) {
            return getShardIndex(devId, mesh, reductionAxes) == 0;
          });
      Value partitionId = stablehlo::ConvertOp::create(
          rewriter, loc, RankedTensorType::get({}, rewriter.getI64Type()),
          stablehlo::PartitionIdOp::create(rewriter, loc));
      auto tableConst = stablehlo::ConstantOp::create(
          rewriter, loc,
          DenseElementsAttr::get(
              RankedTensorType::get({numDevices}, rewriter.getI1Type()),
              leaderTable));
      Value isLeader =
          stablehlo::DynamicSliceOp::create(
              rewriter, loc, RankedTensorType::get({1}, rewriter.getI1Type()),
              tableConst, {partitionId}, rewriter.getDenseI64ArrayAttr({1}))
              .getResult();

      isLeader = stablehlo::BroadcastInDimOp::create(
          rewriter, loc,
          RankedTensorType::get(localInputType.getShape(),
                                rewriter.getI1Type()),
          isLeader, rewriter.getDenseI64ArrayAttr({0}));
      // Broadcast the scalar identity to the shape of the current local input.
      Value broadcastIdentity = stablehlo::BroadcastInDimOp::create(
          rewriter, loc, localInputs.front().getType(), identity,
          rewriter.getDenseI64ArrayAttr({}));
      llvm::transform(localInputs, localInputs.begin(), [&](Value input) {
        return stablehlo::SelectOp::create(rewriter, loc, isLeader, input,
                                           broadcastIdentity);
      });
    }

    auto globalInputType = cast<RankedTensorType>(input.getType());
    auto dnums = op.getScatterDimensionNumbers();
    // Identify sharded indexed dimensions in the input.
    SmallVector<int64_t> indexedInputDims;
    if (!isFullyReplicated(inputSharding)) {
      for (int64_t i = 0; i < globalInputType.getRank(); ++i) {
        if (!inputSharding.getDimShardings()[i].getAxes().empty()) {
          SDY_CHECK(getTotalAxesSize(
                        inputSharding.getDimShardings()[i].getAxes(), mesh) > 1)
              << "We should have removed trivial sharding in a previous pass.";
          if (llvm::is_contained(dnums.getScatterDimsToOperandDims(), i)) {
            indexedInputDims.push_back(i);
          } else if (llvm::is_contained(dnums.getInsertedWindowDims(), i)) {
            // TODO(b/496605332): Support this case.
            return op.emitOpError() << "sharding operand dimension " << i
                                    << " that is not indexed but in inserted "
                                       "window dimension is not supported.";
          }
        }
      }
    }

    // If no indexed dims are sharded and there is no reduction axes, just
    // localize the types.
    if (indexedInputDims.empty() && reductionAxes.empty()) {
      return localizeGenericOp(op, adaptor.getOperands(), rewriter, converter,
                               conversionState);
    }

    // Adjust indices for sharded dimensions (Disjoint writes are parallel).
    Value adjustedIndices = adaptor.getScatterIndices();
    if (!isFullyReplicated(inputSharding)) {
      adjustedIndices =
          computeLocalIndicesAndMask(loc, adjustedIndices, mesh, inputSharding,
                                     globalInputType, localInputType, dnums,
                                     indexedInputDims, rewriter,
                                     /*computeMask=*/false)
              .first;
    }

    // Create the local scatter using adjusted indices and the original updates.
    auto localScatter = stablehlo::ScatterOp::create(
        rewriter, loc, converter->convertResultTypes(op.getResults()),
        localInputs, adjustedIndices, adaptor.getUpdates(), dnums,
        op.getIndicesAreSorted(), op.getUniqueIndices());
    copyAttributes(
        op, localScatter, /*attributesToExclude=*/
        {"scatter_dimension_numbers", "indices_are_sorted", "unique_indices"});
    // Inline the original reduction region.
    rewriter.inlineRegionBefore(op.getUpdateComputation(),
                                localScatter.getUpdateComputation(),
                                localScatter.getUpdateComputation().end());

    rewriter.replaceOp(op, localScatter.getResults());
    conversionState.removeToConvertOp(op);
    return success();
  }

 private:
  ConversionState& conversionState;
};

class StablehloSliceOpPattern : public OpConversionPattern<stablehlo::SliceOp> {
 public:
  StablehloSliceOpPattern(TypeConverter& converter, MLIRContext* ctx,
                          ConversionState& state)
      : OpConversionPattern<stablehlo::SliceOp>(converter, ctx),
        conversionState(state) {}

  LogicalResult matchAndRewrite(
      stablehlo::SliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto* converter =
        static_cast<const GlobalToLocalTypeConverter*>(getTypeConverter());
    TensorShardingAttr inSharding = converter->getSharding(op.getOperand());
    TensorShardingAttr outSharding = converter->getSharding(op.getResult());

    if (isFullyReplicated(inSharding) && isFullyReplicated(outSharding)) {
      return localizeGenericOp(op, adaptor.getOperands(), rewriter, converter,
                               conversionState);
    }

    auto globalInputType = cast<RankedTensorType>(op.getOperand().getType());
    SmallVector<int64_t> newStartIndices, newLimitIndices;
    int64_t rank = globalInputType.getRank();
    newStartIndices.reserve(rank);
    newLimitIndices.reserve(rank);

    for (int64_t i = 0; i < rank; ++i) {
      auto inAxes = inSharding.getDimShardings()[i].getAxes();
      auto outAxes = outSharding.getDimShardings()[i].getAxes();

      if (inAxes != outAxes) {
        return op.emitOpError()
               << "dimension " << i << " has mismatched sharding axes ("
               << strippedAttrsString(inAxes) << " vs "
               << strippedAttrsString(outAxes)
               << "). This requires a reshard which should have been handled "
                  "by previous passes.";
      }

      int64_t globalStart = op.getStartIndices()[i];
      int64_t globalLimit = op.getLimitIndices()[i];
      if (inAxes.empty()) {
        // Unsharded -> Unsharded. Use global indices.
        newStartIndices.push_back(globalStart);
        newLimitIndices.push_back(globalLimit);
      } else {
        // Sharded -> Sharded. Expect a full slice.
        if (globalStart != 0 || globalLimit != globalInputType.getDimSize(i)) {
          return op.emitOpError() << "dimension " << i
                                  << " is sharded but the slice is not a full "
                                     "slice and requires device communication.";
        }
        newStartIndices.push_back(0);
        newLimitIndices.push_back(
            cast<RankedTensorType>(adaptor.getOperand().getType())
                .getDimSize(i));
      }
    }

    // Generate a local slice.
    rewriter.replaceOp(
        op, stablehlo::SliceOp::create(
                rewriter, op.getLoc(),
                cast<RankedTensorType>(converter->convertType(op.getResult())),
                adaptor.getOperand(), newStartIndices, newLimitIndices,
                op.getStrides())
                .getResult());

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
      if (isa<MeshOp, ModuleOp>(op)) {
        // Ops that are always legal do not need to be converted.
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
    patterns.add<
        AllReduceOpPattern, AllSliceOpPattern, AllToAllOpPattern,
        CollectivePermuteOpPattern, ConstantOpPattern, FuncOpSignaturePattern,
        GenericOpPattern, ManualComputationOpPattern, NamedComputationOpPattern,
        ReturnOpPattern, StablehloConvolutionOpPattern,
        StablehloDotGeneralOpPattern, StablehloDotOpPattern,
        StablehloGatherOpPattern, StablehloIotaOpPattern, StablehloPadOpPattern,
        StablehloScatterOpPattern, StablehloSliceOpPattern>(
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
    // Enable type conversion for ops from unknown dialects.
    target.markUnknownOpDynamicallyLegal(
        [&](Operation* op) { return !conversionState.needConversion(op); });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
