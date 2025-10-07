/* Copyright 2025 The MPMD Authors.

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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Threading.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/sharding_propagation/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/sharding_propagation/utils.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/explicit_reshards_util.h"
#include "shardy/dialect/sdy/transforms/propagation/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_EXTRACTRESHARDSFROMINTERMESHTRANSFERSPASS
#include "shardy/dialect/mpmd/transforms/sharding_propagation/passes.h.inc"

namespace {

using ::mlir::sdy::TensorShardingAttr;

template <typename... OpTys>
bool hasAnyUserOfTypeExcept(Value value, Operation* except) {
  return llvm::any_of(value.getUsers(), [&except](Operation* user) {
    return isa<OpTys...>(user) && user != except;
  });
}

RankedTensorType GetLocalTensorType(RankedTensorType global_type,
                                    TensorShardingAttr sharding,
                                    sdy::MeshAttr mesh) {
  if (!sharding) {
    return global_type;
  }
  return sharding.getLocalTensorType(global_type, mesh);
}

// Returns true if the resharding should be done at the producer site, i.e., if
// the destination tensor is on host, or if the source tensor is not on host and
// is smaller (after sharding) than the destination tensor.
bool ReshardAtProducerSite(MeshTensorType src_mesh_type,
                           MeshTensorType dst_mesh_type,
                           TensorShardingAttr src_sharding_or_null,
                           TensorShardingAttr dst_sharding_or_null,
                           sdy::MeshAttr mesh) {
  SDY_CHECK(!src_mesh_type.isOnHost() || !dst_mesh_type.isOnHost());

  if (dst_mesh_type.isOnHost()) {
    return true;
  }

  if (src_mesh_type.isOnHost()) {
    return false;
  }

  int64_t src_num_elements =
      GetLocalTensorType(src_mesh_type.getGlobalTensorType(),
                         src_sharding_or_null, mesh)
          .getNumElements();
  int64_t dst_num_elements =
      GetLocalTensorType(dst_mesh_type.getGlobalTensorType(),
                         dst_sharding_or_null, mesh)
          .getNumElements();
  return src_num_elements > dst_num_elements;
}

void HandleTransfer(TransferOp transfer, RewriterBase& rewriter,
                    sdy::MeshAttr mesh) {
  auto src_mesh_type =
      cast<mpmd::MeshTensorType>(transfer.getTensor().getType());
  auto dst_mesh_type = cast<mpmd::MeshTensorType>(transfer.getType());

  // TODO: jupvfranco - We need a better way to handle resharding on host.
  if (src_mesh_type.isOnHost() && dst_mesh_type.isOnHost()) {
    transfer->emitError()
        << "Resharding on host not supported with an mpmd.transfer.";
    return;
  }

  TensorShardingAttr src_sharding_or_null =
      sdy::getSharding(transfer.getTensor());
  TensorShardingAttr dst_sharding_or_null =
      sdy::getSharding(transfer.getResult());

  // No resharding.
  if (sdy::isEquivalent(src_sharding_or_null, dst_sharding_or_null)) {
    return;
  }

  // TODO: jupvfranco - the following two cases should have been supported
  // according to StaticInputProgramPolicy::Config::SetupResharding. However,
  // I still get an error for an N:1 case. Disabling this for now.
  // 1:N resharding, i.e., replicated to sharded case.
  // if (src_mesh_type.isFullyReplicated() &&
  //     !dst_mesh_type.isFullyReplicated()) {
  //   return;
  // }
  // // N:1 resharding, i.e., sharded to replicated case.
  // if (!src_mesh_type.isFullyReplicated() &&
  //     dst_mesh_type.isFullyReplicated()) {
  //   return;
  // }

  static llvm::once_flag onceFlag;
  sdy::emitOpWarningOnce(onceFlag, transfer,
                         "Found a resharding transfer in the module. This may "
                         "cause performance issues.");

  auto reshard_body = [](ArrayRef<Value> args,
                         OpBuilder&) -> llvm::SmallVector<Value> {
    return {args.front()};
  };
  if (ReshardAtProducerSite(src_mesh_type, dst_mesh_type, src_sharding_or_null,
                            dst_sharding_or_null, mesh)) {
    // Reshard at producer-site because destination type is smaller than source
    // type. I.e.,
    //   %x = op : <M, D>
    //   %t = transfer (%x) : (<M, D>) -> <M', D'>
    //    ~>
    //   %x = op : <M, D'>
    //   %t = transfer (%x) : (<M, D'>) -> <M', D'>
    //
    // Or if the transferred value is an argument:
    //   %t = transfer (%arg) : (<M, D>) -> <M', D'>
    //    ~>
    //   %r = fragment (%arg) : (<M, D>) -> <M, D'>
    //   %t = transfer (%r) : (<M, D'>) -> <M', D'>
    OpOperand& operand = transfer->getOpOperand(0);
    Value value = operand.get();
    MeshTensorType new_operand_type = MeshTensorType::get(
        rewriter.getContext(), src_mesh_type.getMeshName(),
        dst_mesh_type.getRankedTensorType());
    if (isa<BlockArgument>(value) || isa<TransferOp>(value.getDefiningOp())) {
      // We do not want to update the type of the block argument, not to
      // interfere with the function signature (and its shardings).
      rewriter.setInsertionPoint(transfer);
      FragmentOp reshard = FragmentOp::createMeshFragmentWithGlobalBody(
          value.getLoc(), /*user_origin=*/{}, src_mesh_type.getMeshName(),
          value, new_operand_type, rewriter, reshard_body);
      reshard.setUserSpecifiedResultSharding(0, dst_sharding_or_null);
      operand.set(reshard.getResult(0));
      return;
    }

    // If the value is used by a terminator, we need to create a fragment to
    // make sure we don't interfere with the function signature (and its
    // shardings). Similarly, if the value is used by *another* transfer, we
    // also create a fragment to reshard it, so that we don't create more
    // resharding transfers.
    if (hasAnyUserOfTypeExcept<TransferOp, func::ReturnOp>(value, transfer)) {
      rewriter.setInsertionPoint(transfer);
      FragmentOp reshard = FragmentOp::createMeshFragmentWithGlobalBody(
          value.getLoc(), /*user_origin=*/{}, new_operand_type.getMeshName(),
          value, value.getType(), rewriter, reshard_body);
      reshard.setUserSpecifiedResultSharding(0, src_sharding_or_null);
      rewriter.replaceUsesWithIf(
          value, reshard.getResult(0), [transfer](OpOperand& use) {
            Operation* owner = use.getOwner();
            // Automatically excludes `reshard`.
            return isa<TransferOp, func::ReturnOp>(owner) && owner != transfer;
          });
    }

    // At this point, `value` is produced by a fragment and used by fragments
    // only. Thus, it is safe to simply update its type.
    SDY_CHECK(isa<FragmentOp>(value.getDefiningOp()));
    value.setType(new_operand_type);
    sdy::setSharding(value, dst_sharding_or_null);
    return;
  }

  // Reshard at consumer-site because source type is smaller than destination
  // type. I.e.,
  //   %t = transfer (_) : (<M, D>) -> <M', D'>
  //        consumer (%t) : (<M', D'>) -> ...
  //    ~>
  //   %t = transfer (_) : (<M, D>) -> <M', D>
  //        consumer (%t) : (<M', D>) -> ...
  // Or:
  //   %t = transfer (_) : (<M, D>) -> <M', D'>
  //        return (%t)
  //    ~>
  //   %t = transfer : (<M, D>) -> <M', D>
  //        fragment(%t) : (<M', D>) -> <M', D'>
  rewriter.setInsertionPointAfter(transfer);
  auto new_transfer = rewriter.replaceOpWithNewOp<TransferOp>(
      transfer,
      MeshTensorType::get(rewriter.getContext(), dst_mesh_type.getMeshName(),
                          src_mesh_type.getRankedTensorType()),
      transfer.getOperand());
  if (src_sharding_or_null) {
    sdy::setSharding(new_transfer.getResult(), src_sharding_or_null);
  }

  // Update the in_shardings attr of fragment users.
  // We need to do this because the later
  // `ConvertSdyShardingsToMpmdTypesPass` that moves the sharding to
  // `MeshTensorType` will read the in-shardings from the fragment.
  UpdateFragmentUserInShardings(new_transfer,
    src_sharding_or_null);

  if (sdy::hasAnyUserOfType<TransferOp, func::ReturnOp>(
          new_transfer.getResult())) {
    FragmentOp reshard = FragmentOp::createMeshFragmentWithGlobalBody(
        new_transfer.getLoc(), /*user_origin=*/{}, dst_mesh_type.getMeshName(),
        new_transfer.getResult(),
        MeshTensorType::get(rewriter.getContext(), dst_mesh_type.getMeshName(),
                            dst_mesh_type.getRankedTensorType()),
        rewriter, reshard_body);
    reshard.setUserSpecifiedResultSharding(0, dst_sharding_or_null);
    rewriter.replaceUsesWithIf(
        new_transfer.getResult(), reshard.getResult(0), [](OpOperand& use) {
          return isa<TransferOp, func::ReturnOp>(use.getOwner());
        });
  }
  SDY_CHECK(
      llvm::all_of(new_transfer.getResult().getUsers(),
                   [](Operation* user) { return isa<FragmentOp>(user); }));
}

class ExtractReshardsFromInterMeshTransfersPass
    : public impl::ExtractReshardsFromInterMeshTransfersPassBase<
          ExtractReshardsFromInterMeshTransfersPass> {
  using ExtractReshardsFromInterMeshTransfersPassBase::
      ExtractReshardsFromInterMeshTransfersPassBase;

 private:
  void runOnFunc(func::FuncOp func_op) final {
    SDY_CHECK(mpmd::IsMpmdFunction(func_op))
        << "Expected pass to be applied on MPMD partitioning code path.";
    if (!mpmd::HasHomogeneousTopology(func_op)) {
      // This is needed to guarantee that when we push a distributed type to
      // the mesh_tensor type of a block argument, we create well-formed types.
      // TODO: jupvfranco - relax this condition. Even if the meshes in the
      // topology are different, pushing distributed types to arguments can
      // still form valid types.
      return;
    }
    IRRewriter rewriter(&getContext());
    // Assumes that the topology is homogeneous so we can just get the first
    // mesh.
    sdy::MeshAttr mesh = mpmd::GetTopologyMeshes(func_op).front().getMesh();
    func_op.walk(
        [&](TransferOp transfer) { HandleTransfer(transfer, rewriter, mesh); });
  }
};

}  // namespace
}  // namespace mlir::mpmd
