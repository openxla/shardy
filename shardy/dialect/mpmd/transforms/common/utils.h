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
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"

namespace mlir::mpmd {

// The attribute to avoid CSE.
inline constexpr StringRef kMhloNoCseAttr = "mhlo.no_cse";
// The attribute to indicate that an op has side effects.
inline constexpr StringRef kHasSideEffectAttr = "has_side_effect";

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

namespace detail {

// A non-templated version of MergeRegionOps<OpTy> that takes a callback for
// creating the merged op. Note that the `consumer_op` may have some (or even
// none) uses of the `producer_op` results -- i.e. it is not guaranteed to be
// a true consumer. This is implied in the templated version but not here.
Operation* MergeRegionOps(
    Operation* producer_op, Operation* consumer_op,
    RewriterBase& rewriter, int num_static_args,
    std::function<void(OpOperand&, Value)>
        replace_producer_use_in_consumer_block,
    std::function<Operation*(Location, TypeRange,
                                   ValueRange)>
        create_merged_op);

}  // namespace detail

// Merges `producer_op` and `consumer_op` into a single OpTy that returns all
// the results of `producer_op` that are not just used by `consumer_op`, as
// well as all the results of `consumer_op`. All uses of `producer_op` that
// were by `consumer_op` directly (if any) will be replaced with the
// corresponding producer return operands in the merged block, and any uses
// that were nested within the region of `consumer_op` (if any) will be
// replaced using the provided `replace_producer_use_in_consumer_block`.
//
// The merged op won't have duplicate operands.
//
// `num_static_args` specifies the number of static block arguments that both
// `producer_op` and `consumer_op` have as the first arguments, whose types
// are assumed to be identical for both ops, before any dynamic block
// arguments, i.e. block arguments that correspond to operands.
//
// `builder_args` should include any builder argument that should be forwarded
// to `rewriter.create<OpTy>()` in addition to result types and operands.
//
// NOTE: we assume OpTy is an op with a single region, that has
// `num_static_args` static block arguments and an additional block argument
// for each operand, and that all uses of `producer_op` are at or after
// `consumer_op`.
template <class OpTy, class... BuilderArgs>
OpTy MergeRegionOps(OpTy producer_op, OpTy consumer_op,
                    RewriterBase& rewriter, int num_static_args,
                    std::function<void(OpOperand&, Value)>
                        replace_producer_use_in_consumer_block,
                    BuilderArgs&&... builder_args) {
  return cast<OpTy>(detail::MergeRegionOps(
      producer_op, consumer_op, rewriter, num_static_args,
      replace_producer_use_in_consumer_block,
      [&](Location loc, TypeRange result_types,
          ValueRange operands) {
        return rewriter.create<OpTy>(
            loc, result_types, operands,
            std::forward<BuilderArgs>(builder_args)...);
      }));
}

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_COMMON_UTILS_H_
