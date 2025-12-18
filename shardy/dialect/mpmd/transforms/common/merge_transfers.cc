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
#include <iterator>
#include <utility>
#include <vector>

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_MERGETRANSFERSPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

namespace {

// Sets a threshold for the number of elements in a value that makes it a
// candidate for transfer merging. Any transfer with more elements will not be
// merged.
// NOTE: this is set to 1 as it proved to be beneficial when training the Tier6
// model. Though, increasing this threshold could be helpful when optimizing
// other configurations or actually make it worse since it may increase the
// memory footprint (even if slightly) and requires the introduction of concats
// and slices, which could be costly in very small programs (e.g., serving).
inline constexpr int kNumElementsThreshold = 1;
inline constexpr StringRef kIsConcatTransfer = "concat_transfer";

bool IsConcatTransfer(TransferOp transfer) {
  return transfer->hasAttr(kIsConcatTransfer);
}

void SetConcatTransfer(TransferOp transfer) {
  transfer->setAttr(kIsConcatTransfer, UnitAttr::get(transfer.getContext()));
}

// Returns all TransferOps that are users of `value` and were not introduced by
// this pass.
std::vector<TransferOp> GetNonConcatTransferUsers(Value value) {
  std::vector<TransferOp> users;
  for (Operation* user : value.getUsers()) {
    if (auto transfer = dyn_cast<TransferOp>(user)) {
      if (IsConcatTransfer(transfer)) {
        continue;
      }
      users.push_back(transfer);
    }
  }
  return users;
}

// Checks if a value is eligible for merging, i.e., if it is not pinned to host,
// not sharded, and the number of elements is below the threshold.
bool IsEligibleForMerging(Value value) {
  auto mesh_tensor_type = cast<MeshTensorType>(value.getType());
  // We do not concatenate values that are on the host or that are sharded.
  if (mesh_tensor_type.getMemoryKind() &&
      mesh_tensor_type.getMemoryKind().getValue() == kMemoryKindPinnedHost) {
    return false;
  }
  if (mesh_tensor_type.getSharding() &&
      !mesh_tensor_type.getSharding().isFullyReplicated()) {
    return false;
  }
  RankedTensorType result_type = mesh_tensor_type.getRankedTensorType();
  if (kNumElementsThreshold > -1 &&
      result_type.getNumElements() > kNumElementsThreshold) {
    // Being more aggressive here could impact runtime/memory performance.
    // E.g., more concatenation could mean runtime degradation in very
    // small/fast programs, or memory footprint increase since we duplicate
    // the returned values when they're used by multiple consumers.
    return false;
  }
  return true;
}

SetVector<FragmentOp> FragmentUsers(Operation* op) {
  SetVector<FragmentOp> users;
  for (Operation* user : op->getUsers()) {
    if (auto fragment = dyn_cast<FragmentOp>(user)) {
      users.insert(fragment);
    }
  }
  return users;
}

using TypeAndConsumerFragment = std::pair<RankedTensorType, FragmentOp>;

// Given a `producer` fragment, returns which values should be concatenated for
// the merged transfer. The returned data structure maps:
//   <Type, ConsumerFragment> -> vector{indices of results to concatenate}
// The indices in the vector correspond to all the values of a given type that
// are used by the same consumer fragment, via a transfer. E.g.,
//
//   ..., v1, ..., vn, ... = fragment ... {
//   }, with vi: Ti.
//   t1 = transfer v1 : mesh_tensor<m, T> -> mesh_tensor<m', T>
//    ...
//   tn = transfer vn : mesh_tensor<m, T> -> mesh_tensor<m', T>
//   consumer = fragment ... (..., t1, ..., tn, ...)) { ...}
//
// The returned data structure will contain:
//   <T1, consumer> -> sorted({indexof(v1), ..., indexof(vn)}})
//
// The vector of result indices is sorted so that know that after concatenation,
// the result #i of the vector corresponds to the slice #i of the concatenated
// value.
//
// Values that are sharded, or allocated in pinned_host, or have more elements
// than the threshold are not considered for concatenation.
llvm::MapVector<TypeAndConsumerFragment, std::vector<int>> FindValuesToConcat(
    FragmentOp producer) {
  llvm::MapVector<TypeAndConsumerFragment, std::vector<int>> values_to_concat;
  for (OpResult result : producer.getResults()) {
    if (!IsEligibleForMerging(result)) {
      continue;
    }
    auto result_type =
        cast<MeshTensorType>(result.getType()).getRankedTensorType();
    for (TransferOp transfer : GetNonConcatTransferUsers(result)) {
      // We iterate over the *set* (not vector) of fragment users to avoid
      // adding the same consumer multiple times, in case a transfer is used by
      // the same consumer multiple times.
      for (FragmentOp consumer_fragment : FragmentUsers(transfer)) {
        values_to_concat[{result_type, consumer_fragment}].push_back(
            result.getResultNumber());
      }
    }
  }
  return values_to_concat;
}
// Creates a new fragment that is identical to fragment except that it
// additionally returns result_value (of type result_type). Returns
// the last result of the newly created FragmentOp, i.e. the result
// corresponding to returned_value".
Value AddNewResultToFragment(FragmentOp fragment, Value returned_value,
                             MeshTensorType result_type,
                             RewriterBase& rewriter) {
  rewriter.setInsertionPoint(fragment);

  Operation* terminator = fragment.getBody()->getTerminator();
  terminator->insertOperands(terminator->getNumOperands(), returned_value);
  std::vector<Type> result_types(fragment.getResultTypes().begin(),
                                 fragment.getResultTypes().end());
  result_types.push_back(result_type);
  auto new_fragment =
      FragmentOp::create(rewriter, fragment.getLoc(), result_types,
                         fragment.getOperands(), fragment.getOriginAttr(),
                         fragment.getMeshNameAttr(), fragment.getStageIdAttr());
  // Copy all attributes except `origin` and `mesh_name`, which were copied
  // during the creation of the new fragment.
  CopyAttributes(fragment, new_fragment,
                 /*elided_attrs_set=*/{"origin", "mesh_name", "stage_id"});
  new_fragment.getRegion().takeBody(fragment.getRegion());
  rewriter.replaceOp(fragment, new_fragment.getResults().drop_back());
  return new_fragment.getResult(terminator->getNumOperands() - 1);
}

// Concatenates the results of `producer` in indices `results_to_concat`. The
// result of the concatenation is returned by the producer. Returns the newly
// added result.
//
// results_to_concat[i] corresponds to the slice #i of the concatenated value.
//
// Requires: each result in `results_to_concat` has type `result_type`.
//
// Note: concatenated values are not removed as they may have multiple
// consumers.
//
// Example:
//   %producer:N = fragment ... {
//      ...
//      return ..., %v1, %v2, ... : ..., T, T, ...
//   }  // with v1: T, v2: T and T = tensor<d1 x... x dn>
//     ~~>
//   %producer:N+1 = fragment ... {
//      ...
//      %r1 = reshape %v1 : tensor<1 x d1 x ... x dn>
//      %r2 = reshape %v2 : tensor<1 x d1 x ... x dn>
//      %concat = concat(%r1, %r2) : tensor<2 x d1 x ... x dn>
//      return ..., %v1, %v2, ..., %concat
//            : ..., T, T, ..., tensor<2 x d1 x ... x dn>
//   }
//
Value ConcatResultsOnProducer(FragmentOp producer,
                              const std::vector<int>& results_to_concat,
                              RankedTensorType result_type,
                              IRRewriter& rewriter) {
  Operation* terminator = producer.getBody()->getTerminator();
  rewriter.setInsertionPoint(terminator);

  // The list of values that will be concatenated.
  std::vector<Value> concat_operands;
  concat_operands.reserve(results_to_concat.size());

  // The shape of the value before it's concatenated to other values, i.e., it
  // has a leading dimension of size 1.
  SmallVector<int64_t> reshaped_dimensions;
  reshaped_dimensions.reserve(result_type.getShape().size() + 1);
  reshaped_dimensions.push_back(1);
  llvm::copy(result_type.getShape(), std::back_inserter(reshaped_dimensions));

  // Reshape each result in `results_to_concat` to have shape
  // `reshaped_dimensions`/
  for (int result_index : results_to_concat) {
    OpResult result = producer->getResult(result_index);
    Value operand = terminator->getOperand(result.getResultNumber());
    SDY_CHECK(operand.getType() == result_type);
    auto new_shape = RankedTensorType::get(reshaped_dimensions,
                                           result_type.getElementType());
    mlir::Value reshape = stablehlo::ReshapeOp::create(
        rewriter, producer.getLoc(), new_shape, operand);
    concat_operands.push_back(reshape);
  }

  // Finally, concatenate the reshaped values and add the result to the
  // producer.
  Value concat = stablehlo::ConcatenateOp::create(rewriter, producer.getLoc(),
                                                  concat_operands, 0);
  MeshTensorType new_result_type =
      MeshTensorType::get(result_type.getContext(), producer.getMeshName(),
                          cast<RankedTensorType>(concat.getType()));
  return AddNewResultToFragment(producer, concat, new_result_type, rewriter);
}

// Creates a slice op to extract the `slice_index`th slice of shape `shape` from
// the concatenated `concatenated_arg`.
Value CreateSliceOnArgument(ArrayRef<int64_t> shape,
                            BlockArgument concatenated_arg, int64_t slice_index,
                            IRRewriter& rewriter) {
  SmallVector<int64_t> start_indices{slice_index};
  SmallVector<int64_t> limit_indices{slice_index + 1};
  SmallVector<int64_t> strides{1};
  for (int dim : shape) {
    start_indices.push_back(0);
    limit_indices.push_back(dim);
    strides.push_back(1);
  }
  return stablehlo::SliceOp::create(rewriter, concatenated_arg.getLoc(),
                                    concatenated_arg, start_indices,
                                    limit_indices, strides);
}

// Adds the result of `concat_transfer` to the operands (and block arguments) of
// the `consumer` fragment and replaces all uses of each argument in
// `consumer_args` with the respective slice of the concatenated transfer. E.g.,
//
//   %t1 = transfer %producer#...
//   ...
//   %tn = transfer %producer#...
//   %concat_transfer = transfer %concatenated_result
//                    : (mesh_tensor<m, tensor<n x d1 x ... x dn>>)
//                    -> mesh_tensor<m', tensor<n x d1 x ... x dn>>
//   %consumer = fragment ... (..., %t1, .., %tn, ...)
//                          : (..., %arg1, ..., %argn, ...) {
//      use(%arg1) ... use(%argn)
//   }
//   With %concatenated_result == concat(%t1, ..., %tn).
//
//   ~~>
//
//   %t1 = transfer %producer#...
//   ...
//   %tn = transfer %producer#...
//   %concat_transfer = transfer %concatenated_result
//                    : (mesh_tensor<m, tensor<n x d1 x ... x dn>>)
//                    -> mesh_tensor<m', tensor<n x d1 x ... x dn>>
//   %consumer = fragment ... (..., %t1, .., %tn, ..., %concat_transfer)
//                          : (..., %arg1, ..., %argn, ...,
//                             %arg_concat: tensor<n x d1 x ... x dn>) {
//      %s1 = slice(%arg_concat, 0) : tensor<1 x d1 x ... x dn>
//      %s1_reshape = reshape(%s1) : tensor<d1 x ... x dn>
//      ...
//      %sn = slice(%arg_concat, n) : tensor<1 x d1 x ... x dn>
//      %sn_reshape = reshape(%sn) : tensor<d1 x ... x dn>
//      ...
//      use(%s1_reshape) ... use(%sn_reshape)
//      // %arg1, ..., %argn are dead.
//   }
// Requires: every arg in `consumer_args` to be a block argument of `consumer`.
void SliceConcatOnConsumer(
    FragmentOp consumer,
    const llvm::MapVector<BlockArgument, int>& consumer_args,
    TransferOp concat_transfer, IRRewriter& rewriter) {
  rewriter.setInsertionPoint(consumer.getBody(), consumer.getBody()->begin());
  // Append the concatenated result to the operands of the consumer.
  consumer->insertOperands(consumer->getNumOperands(),
                           concat_transfer.getResult());

  // Add a new argument to the consumer for the concatenated result.
  BlockArgument concat_arg = consumer.getBody()->addArgument(
      concat_transfer.getType().getRankedTensorType(),
      concat_transfer.getLoc());
  for (const auto& [arg, slice_index] : consumer_args) {
    RankedTensorType arg_type = cast<RankedTensorType>(arg.getType());

    // Create the respective slice of the concatenated result.
    Value slice = CreateSliceOnArgument(arg_type.getShape(), concat_arg,
                                        slice_index, rewriter);

    // Drop the leading dimension of size 1 that results from the slicing.
    Value reshape = stablehlo::ReshapeOp::create(rewriter, consumer.getLoc(),
                                                 arg_type, slice);

    // Replace the argument with the slice, without erasing it. We do this
    // for the sake of simplicity -- we can use the simplify passes to
    // clean it up afterwards.
    rewriter.replaceAllUsesWith(arg, reshape);
  }
}

// Finds the consumer block arguments that need to be replaced with slices of
// the concatenated transfer.
llvm::MapVector<BlockArgument, int> FindConsumerArgumentsToReplace(
    const std::vector<int>& result_indices, FragmentOp producer,
    FragmentOp consumer) {
  llvm::MapVector<BlockArgument, int> consumer_args;
  for (auto [slice_index, result_index] : llvm::enumerate(result_indices)) {
    for (TransferOp transfer :
         GetNonConcatTransferUsers(producer.getResult(result_index))) {
      for (OpOperand& transfer_use : transfer->getUses()) {
        if (transfer_use.getOwner() == consumer) {
          consumer_args[consumer.getBody()->getArgument(
              transfer_use.getOperandNumber())] = slice_index;
        }
      }
    }
  }
  return consumer_args;
}

// Given a `producer` fragment, concatenates sets of results that are smaller
// than a given threshold and transferred to the same consumer fragment.
void MergeTransfersProducedByFragment(FragmentOp producer,
                                      IRRewriter& rewriter) {
  llvm::MapVector<TypeAndConsumerFragment, std::vector<int>> values_to_concat =
      FindValuesToConcat(producer);
  FragmentOp current_producer = producer;
  for (const auto& [type_and_consumer_fragment, result_indices] :
       values_to_concat) {
    auto [type, consumer_fragment] = type_and_consumer_fragment;
    SDY_CHECK(!result_indices.empty());
    if (result_indices.size() == 1) {
      // Nothing to concatenate.
      continue;
    }

    Value concat_result = ConcatResultsOnProducer(
        current_producer, result_indices, type, rewriter);
    rewriter.setInsertionPointAfterValue(concat_result);

    MeshTensorType concat_result_type =
        cast<MeshTensorType>(concat_result.getType());

    // Transfer the concatenated result to the consumer.
    auto concat_transfer = TransferOp::create(
        rewriter, concat_result.getLoc(),
        MeshTensorType::get(concat_result_type.getContext(),
                            consumer_fragment.getMeshName(),
                            concat_result_type.getRankedTensorType()),
        concat_result);

    // Mark it as the result of a concatenation to guarantee that
    // `concat_result_type` won't flow into a further concatenation.
    SetConcatTransfer(concat_transfer);

    // NOTE: at this point the producer is dead (as a consequence of
    // concatenating results on the producer), so we need to be careful not to
    // use it anymore. We update the current producer to the fragment that
    // replaced `producer`.
    current_producer = cast<FragmentOp>(concat_result.getDefiningOp());

    // Find the arguments of the consumer that are uses of the results that
    // were concatenated.
    llvm::MapVector<BlockArgument, int> consumer_arguments =
        FindConsumerArgumentsToReplace(result_indices, current_producer,
                                       consumer_fragment);

    // Replace the uses of the old results with slices of the new concatenated
    // result.
    SliceConcatOnConsumer(consumer_fragment, consumer_arguments,
                          concat_transfer, rewriter);
  }
}

class MergeTransfersPass
    : public impl::MergeTransfersPassBase<MergeTransfersPass> {
  using MergeTransfersPassBase::MergeTransfersPassBase;

 private:
  void runOnFunc(func::FuncOp func) override {
    if (!IsMpmdFunction(func)) {
      return;
    }

    IRRewriter rewriter(func.getContext());
    Block& block = func.getBody().front();

    // Copy all fragments to a vector so that replacing them with new fragments
    // (in order to introduce new results) doesn't affect the iteration order.
    SmallVector<FragmentOp> fragments(block.getOps<FragmentOp>().begin(),
                                      block.getOps<FragmentOp>().end());
    for (FragmentOp producer : fragments) {
      MergeTransfersProducedByFragment(producer, rewriter);
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
