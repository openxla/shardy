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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_COMMON_UTILS_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_COMMON_UTILS_H_

#include <functional>
#include <string>

#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"

namespace mlir::mpmd {

// The name of the attribute that keeps track of how many times a loop has been
// unrolled.
constexpr StringRef kUnrollCounterAttrName = "unroll_counter";

// TODO(joelwee): rename these attributes to be more generic and less tied to
// splitting for near-zero-bubble pipelining.
// Attribute to mark a fragment after some splitting has been applied to it. The
// distinguishing feature is that the fragment has kept its transferred results.
constexpr StringRef kSplitKeepTransferredAttrName = "split_keep_transferred";
// Attribute to mark a fragment that has been split out of another one. The
// distinguishing feature is that this fragment has dropped any transferred
// results of the original, i.e. its results are never transferred.
constexpr StringRef kSplitDropTransferredAttrName = "split_drop_transferred";

// Attribute to mark the start index of control operands of an operation. It can
// be assumed that all operands including and after this index are control
// operands.
constexpr StringRef kControlOperandStartIdxAttrName =
    "mpmd.control_operand_start_idx";

// Returns true if `value` is used by any op that isn't `user` or nested in a
// region of `user`.
bool HasOtherUsersExcept(Value value, Operation* user);

// Returns the closest surrounding operation for which `pred` returns true
// including `op` itself, or fails if there is no such operation.
//
// If `strict` is false, returns nullptr rather than failing.
Operation* GetAncestorIf(Operation* op, std::function<bool(Operation*)> pred,
                         bool strict = true);

// Returns the closest surrounding operation of `op` that is in the given
// `block`, or fails if there is no such operation.
//
// If `strict` is false, returns nullptr rather than failing.
Operation* GetAncestorInBlock(Block* block, Operation* op, bool strict = true);

// Returns true if `op` has a surrounding operation that is in the given `block`
bool HasAncestorInBlock(Block* block, Operation* op);

// Updates the FunctionType of the given `func_op` to match the block arguments
// and return operands in its region.
void UpdateFunctionType(func::FuncOp func_op);

// Clone an operation and replace its operands.
Operation* Clone(OpBuilder& builder, Operation& operation,
                 ArrayRef<Value> new_operands);

// Copies any attribute not contained in `elided_attrs_set` from `source` to
// `destination`.
// Note: it may overwrite an attribute in the `destination` op.
void CopyAttributes(Operation* source, Operation* destination,
                    llvm::SmallDenseSet<StringRef> elided_attrs_set = {});

// Serializes the MLIR operation as a string.
std::string OperationToString(Operation* op, const OpPrintingFlags& flags);

// Print `op` to string with large constants elided.
std::string PrintOperationForLog(
    Operation* op,
    OpPrintingFlags flags = OpPrintingFlags().elideLargeElementsAttrs());

// Prints a stack trace from the given location.
std::string PrintStackTraceFromLoc(Location loc);

// Prints loc formatted as:
// - If has stack trace, then print it.
// - If NameLoc without child (e.g. arg or result), print just the name.
// - Else defaults to default printing.
std::string PrintLocation(Location loc);

// Returns true if this is one of the two results of the split -- the one that
// drops the transferred results.
bool IsSplitDropTransferred(FragmentOp fragment);

// Returns true if this is one of the two results of the split -- the one that
// keeps the transferred results.
bool IsSplitKeepTransferred(FragmentOp fragment);

// Returns all items in `range` for which `erase` is set to false.
template <class T, class RangeT>
SmallVector<T> FilterRange(RangeT range, const BitVector& erase) {
  SmallVector<T> result;
  result.reserve(range.size());
  for (auto it : llvm::enumerate(range)) {
    if (!erase.test(it.index())) {
      result.push_back(it.value());
    }
  }

  return result;
}

// Searches backward from `fragment` for the nearest FragmentOp on the same
// mesh. Does not consider `fragment` itself. Returns nullptr if none found.
FragmentOp FindLastFragmentOnMesh(FragmentOp fragment);

// Returns the latest (block-order) op that defines any operand of `op`.
// Returns nullptr if all operands are block arguments.
Operation* FindLatestOperandProducer(Operation* op);

// Returns the latest (block-order) FragmentOp among the operand producers of
// `fragment` that is on the same mesh. Returns nullptr if no operand is
// produced by a same-mesh FragmentOp.
FragmentOp FindLatestProducerFragmentOnMesh(FragmentOp fragment);

// Moves `op` after `target` if needed and safe. Returns false if the move
// would break use-def chains. No-op (returns true) if `target` is nullptr,
// `op == target`, or `op` is already after `target`.
bool EnsureAfter(Operation* op, Operation* target);

// Returns true if the stage_ids of the producer and consumer fragments are
// consistent (i.e., they are either identical, or one of them is undefined).
bool AreStageIdsConsistent(FragmentOp producer_op, FragmentOp consumer_op);

// Merges two FragmentOps into a single FragmentOp.
//
// Prerequisites:
//   The producer must be immediately before the consumer in the block.
//
// Behavior:
//   - The producer's body is inlined before the consumer's body in the merged
//     fragment.
//   - Operands are deduplicated: if both fragments reference the same
//     SSA value, it appears only once in the merged fragment's operand
//     list.
//   - Consumer operands that are results of the producer are replaced with the
//     corresponding return values from the producer's body.
//   - Producer results that are only used by the consumer are dropped from the
//     merged fragment's results. Producer results that have other users are
//     preserved (in order) before the consumer's results.
//   - Control dependencies from both fragments are preserved, except those
//     that reference the other fragment being merged (as those become invalid).
//   - The origin of the merged fragment is computed automatically as the union
//     of the origins of the producer and consumer fragments.
//
// The `stage_id` is forwarded as an attribute on the merged FragmentOp.
FragmentOp MergeFragments(FragmentOp producer, FragmentOp consumer,
                          RewriterBase& rewriter);

// Computes discardable attributes for the merged fragment that MergeFragments
// does not handle. Must be called before MergeFragments (which erases both
// ops) and applied to the merged result afterward.
//
// Currently handles:
//   - call_counter: merged if both have the same count or exactly one is
//     user-defined; dropped if conflicting.
//   - inferred_by: union of both when both fragments are inferred;
//     dropped if either is user-defined.
SmallVector<std::pair<StringRef, Attribute>> MergeAttributes(
    FragmentOp producer_op, FragmentOp consumer_op);

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_COMMON_UTILS_H_
