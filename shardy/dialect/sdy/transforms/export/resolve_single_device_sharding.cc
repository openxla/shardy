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

#include <cstdint>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/transforms/export/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_RESOLVESINGLEDEVICESHARDINGPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

// Returns the single device ID if sharding is placed on a maximal mesh.
std::optional<int64_t> getSingleDeviceId(TensorShardingAttr sharding,
                                         const SymbolTable& symbolTable) {
  if (!isSingleDeviceSharding(sharding, symbolTable)) {
    return std::nullopt;
  }
  ArrayRef<int64_t> deviceIds = sharding.getMesh(symbolTable).getDeviceIds();
  if (deviceIds.empty()) {
    return std::nullopt;
  }
  return deviceIds.front();
}

// Returns the single device ID for op if any of its result shardings specify
// one. Asserts that all single device result shardings agree on the same device
// ID and non-single-device result shardings are fully replicated or unsharded.
std::optional<int64_t> getSingleDeviceIdForOp(Operation* op,
                                              const SymbolTable& symbolTable) {
  std::optional<int64_t> targetDeviceId;
  bool hasNonReplicatedResult = false;
  for (Value result : op->getResults()) {
    if (!isa<RankedTensorType>(result.getType())) {
      continue;
    }
    TensorShardingAttr sharding = getSharding(result);
    if (!sharding) {
      continue;
    }
    std::optional<int64_t> deviceId = getSingleDeviceId(sharding, symbolTable);
    if (deviceId.has_value()) {
      if (targetDeviceId.has_value()) {
        SDY_CHECK_EQ(*targetDeviceId, *deviceId);
      } else {
        targetDeviceId = deviceId;
      }
    } else if (!sharding.isFullyReplicated()) {
      hasNonReplicatedResult = true;
    }
  }

  if (targetDeviceId.has_value()) {
    SDY_CHECK(!hasNonReplicatedResult);
  }

  return targetDeviceId;
}

// Creates predicate: getDeviceId(...) == targetDeviceId
Value createDeviceGuard(Location loc, int64_t targetDeviceId,
                        int64_t replicaCount, int64_t partitionCount,
                        IRRewriter& rewriter) {
  Value currentDeviceId =
      getDeviceId(replicaCount, partitionCount, loc, rewriter);
  auto scalarI64Type = RankedTensorType::get({}, rewriter.getI64Type());
  Value targetDeviceIdConst = stablehlo::ConstantOp::create(
      rewriter, loc, DenseElementsAttr::get(scalarI64Type, targetDeviceId));
  return stablehlo::CompareOp::create(rewriter, loc, currentDeviceId,
                                      targetDeviceIdConst,
                                      stablehlo::ComparisonDirection::EQ);
}

// Lowers a single-device operation into stablehlo.if guarded by target device
// ID, with an else-branch returning zeros or tokens.
void resolveSingleDeviceOp(Operation* op, const SymbolTable& symbolTable,
                           MeshOp globalMeshOp, int64_t replicaCount,
                           int64_t partitionCount, IRRewriter& rewriter) {
  std::optional<int64_t> targetDeviceId =
      getSingleDeviceIdForOp(op, symbolTable);
  SDY_CHECK(targetDeviceId.has_value());

  Location loc = op->getLoc();
  rewriter.setInsertionPoint(op);

  Value isTargetDevice = createDeviceGuard(loc, *targetDeviceId, replicaCount,
                                           partitionCount, rewriter);

  SmallVector<Type> resultTypes(op->getResultTypes());
  auto ifOp =
      stablehlo::IfOp::create(rewriter, loc, resultTypes, isTargetDevice);

  // True branch: executes op
  Block* trueBlock = new Block();
  ifOp.getTrueBranch().push_back(trueBlock);
  rewriter.setInsertionPointToStart(trueBlock);
  Operation* clonedOp = rewriter.clone(*op);
  clonedOp->removeAttr(kShardingAttr);
  SmallVector<ReshardOp> operandReshards;
  for (OpOperand& operand : clonedOp->getOpOperands()) {
    if (auto reshardOp = operand.get().getDefiningOp<ReshardOp>()) {
      if (isSingleDeviceSharding(reshardOp.getSharding(), symbolTable)) {
        operand.set(reshardOp.getInput());
        operandReshards.push_back(reshardOp);
      }
    }
  }
  stablehlo::ReturnOp::create(rewriter, loc, clonedOp->getResults());

  // False branch: returns zero tensors or tokens
  Block* falseBlock = new Block();
  ifOp.getFalseBranch().push_back(falseBlock);
  rewriter.setInsertionPointToStart(falseBlock);
  SmallVector<Value> falseZeros;
  for (Type type : resultTypes) {
    if (auto rankedType = dyn_cast<RankedTensorType>(type)) {
      Value zeroConst = createZeroConstant(rewriter, loc, rankedType);
      falseZeros.push_back(zeroConst);
    } else if (isa<stablehlo::TokenType>(type)) {
      Value tokenConst = stablehlo::CreateTokenOp::create(rewriter, loc);
      falseZeros.push_back(tokenConst);
    }
  }
  stablehlo::ReturnOp::create(rewriter, loc, falseZeros);

  // After conditional: AllReduce to populate values to all devices.
  rewriter.setInsertionPointAfter(ifOp);

  SmallVector<AxisRefAttr> allReduceAxes;
  for (MeshAxisAttr axis : globalMeshOp.getMesh().getAxes()) {
    allReduceAxes.push_back(
        AxisRefAttr::get(rewriter.getContext(), axis.getName()));
  }

  for (auto [result, ifResult] :
       llvm::zip(op->getResults(), ifOp.getResults())) {
    auto rankedType = dyn_cast<RankedTensorType>(ifResult.getType());
    if (!rankedType) {
      result.replaceAllUsesWith(ifResult);
      continue;
    }
    TensorShardingAttr replicatedSharding =
        TensorShardingAttr::getFullyReplicated(
            rewriter.getContext(), rankedType.getRank(), globalMeshOp.getName(),
            /*isClosed=*/true);
    auto allReduceOp =
        AllReduceOp::create(rewriter, loc, ifResult, allReduceAxes,
                            ReductionOp::SUM, replicatedSharding);
    for (OpOperand& use : llvm::make_early_inc_range(result.getUses())) {
      if (auto reshardOp = dyn_cast<ReshardOp>(use.getOwner())) {
        if (reshardOp.getSharding() == replicatedSharding) {
          reshardOp.replaceAllUsesWith(allReduceOp.getResult());
          rewriter.eraseOp(reshardOp);
          continue;
        }
      }
      use.set(allReduceOp.getResult());
    }
  }

  rewriter.eraseOp(op);

  for (ReshardOp reshardOp : operandReshards) {
    if (reshardOp->use_empty()) {
      rewriter.eraseOp(reshardOp);
    }
  }
}

struct ResolveSingleDeviceShardingPass
    : public impl::ResolveSingleDeviceShardingPassBase<
          ResolveSingleDeviceShardingPass> {
  using ResolveSingleDeviceShardingPassBase::
      ResolveSingleDeviceShardingPassBase;

 protected:
  void runOnOperation() final {
    func::FuncOp funcOp = getOperation();
    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
    MeshOp globalMeshOp = getGlobalMeshOp(moduleOp);
    if (!globalMeshOp) {
      return;
    }

    SymbolTable symbolTable(moduleOp);
    SmallVector<Operation*> singleDevOps;
    funcOp.walk([&](Operation* op) {
      if (op->getName().getStringRef().starts_with("sdy.")) {
        return;
      }
      if (getSingleDeviceIdForOp(op, symbolTable).has_value()) {
        singleDevOps.push_back(op);
      }
    });

    IRRewriter rewriter(funcOp);
    for (Operation* op : singleDevOps) {
      resolveSingleDeviceOp(op, symbolTable, globalMeshOp, replicaCount,
                            partitionCount, rewriter);
    }
  }
};

}  // namespace
}  // namespace sdy
}  // namespace mlir
