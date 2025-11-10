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

#include "shardy/dialect/mpmd/transforms/common/utils.h"

#include <functional>
#include <iterator>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/RegionUtils.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"

namespace mlir::mpmd {

bool HasOtherUsersExcept(Value value, Operation* user) {
  return llvm::any_of(value.getUsers(), [&user](Operation* other_user) {
    return !user->isAncestor(other_user);
  });
}

Operation* GetAncestorIf(Operation* op, std::function<bool(Operation*)> pred,
                         bool strict) {
  while (op && !pred(op)) {
    op = op->getParentOp();
    if (strict) {
      SDY_CHECK(op)
          << "`op` does not have an ancestor that satisfies predicate.";
    }
  }
  return op;
}

Operation* GetAncestorInBlock(Block* block, Operation* op, bool strict) {
  return GetAncestorIf(
      op, [block](Operation* op) { return op->getBlock() == block; }, strict);
}

bool HasAncestorInBlock(Block* block, Operation* op) {
  return GetAncestorInBlock(block, op, /*strict=*/false);
}

void UpdateFunctionType(func::FuncOp func_op) {
  func_op.setType(FunctionType::get(
      func_op.getContext(), func_op.getBody().getArgumentTypes(),
      func_op.getBody().front().getTerminator()->getOperandTypes()));
}

Operation* Clone(OpBuilder& builder, Operation& operation,
                 ArrayRef<Value> new_operands) {
  Operation* new_operation = builder.clone(operation);
  new_operation->setOperands(new_operands);
  return new_operation;
}

void CopyAttributes(
    Operation* source, Operation* destination,
    const llvm::SmallDenseSet<mlir::StringRef> elided_attrs_set) {
  for (NamedAttribute attr : source->getAttrs()) {
    if (!elided_attrs_set.contains(attr.getName().strref())) {
      destination->setAttr(attr.getName(), attr.getValue());
    }
  }
}

std::string OperationToString(Operation* op, const OpPrintingFlags& flags) {
  std::string out;
  {
    llvm::raw_string_ostream os(out);
    op->print(os, flags);
  }
  return out;
}

std::string PrintOperationForLog(Operation* op, OpPrintingFlags flags) {
  return OperationToString(op, flags);
}

namespace {

void PrintFileLoc(FileLineColLoc file_loc,
                  llvm::raw_string_ostream& loc_stream) {
  loc_stream << "\n"
             << file_loc.getFilename() << ":" << file_loc.getLine() << ":"
             << file_loc.getStartColumn() << " to " << file_loc.getEndColumn();
}

// Recurses into the child locations of some of location types to find a nested
// file location and prints info if it is found. Returns true if a file location
// is found.
bool RecursivelyPrintLoc(Location loc, llvm::raw_string_ostream& loc_stream) {
  return llvm::TypeSwitch<LocationAttr, bool>(loc)
      .Case([&](CallSiteLoc call_loc) -> bool {
        // We recurse into the callee of a call site, as the caller will be
        // emitted in a different note on the main diagnostic.
        return RecursivelyPrintLoc(call_loc.getCallee(), loc_stream);
      })
      .Case([&](FileLineColLoc file_loc) -> bool {
        PrintFileLoc(file_loc, loc_stream);
        return true;
      })
      .Case([&](FusedLoc fused_loc) -> bool {
        // Fused location is unique in that we try to find a sub-location to
        // show, rather than the top-level location itself.
        for (Location childLoc : fused_loc.getLocations()) {
          if (RecursivelyPrintLoc(childLoc, loc_stream)) {
            return true;
          }
        }
        return false;
      })
      .Case([&](NameLoc name_loc) -> bool {
        if (RecursivelyPrintLoc(name_loc.getChildLoc(), loc_stream)) {
          loc_stream << "\n\t ^ " << name_loc.getName();
          return true;
        };
        return false;
      })
      .Case([&](OpaqueLoc opaque_loc) -> bool {
        // OpaqueLoc always falls back to a different source location.
        return RecursivelyPrintLoc(opaque_loc.getFallbackLocation(),
                                   loc_stream);
      })
      .Case([](UnknownLoc) -> bool {
        // Prefer not to show unknown locations.
        return false;
      });
}

// Finds a nested call site location in the given location.
std::optional<CallSiteLoc> GetCallSiteLoc(Location loc) {
  if (dyn_cast<NameLoc>(loc))
    return GetCallSiteLoc(cast<NameLoc>(loc).getChildLoc());
  if (auto callLoc = dyn_cast<CallSiteLoc>(loc)) {
    return callLoc;
  }
  if (dyn_cast<FusedLoc>(loc)) {
    for (auto subLoc : cast<FusedLoc>(loc).getLocations()) {
      // If fused, just get the first call site location.
      if (auto callLoc = GetCallSiteLoc(subLoc)) {
        return callLoc;
      }
    }
    return std::nullopt;
  }
  return std::nullopt;
}

void PrintStackTraceFromLoc(Location loc,
                            llvm::raw_string_ostream& loc_stream) {
  if (auto call_loc = GetCallSiteLoc(loc)) {
    // Print the info from the current loc.
    RecursivelyPrintLoc(*call_loc, loc_stream);
    // Print the file locations of the callers.
    PrintStackTraceFromLoc(call_loc->getCaller(), loc_stream);
  }
}

}  // namespace

std::string PrintStackTraceFromLoc(Location loc) {
  std::string loc_str;
  llvm::raw_string_ostream loc_stream(loc_str);
  PrintStackTraceFromLoc(loc, loc_stream);
  return loc_str;
}

std::string PrintLocation(Location loc) {
  std::string loc_str;
  llvm::raw_string_ostream loc_stream(loc_str);

  PrintStackTraceFromLoc(loc, loc_stream);
  if (loc_str.empty()) {
    if (auto name_loc = dyn_cast<NameLoc>(loc);
        name_loc && isa<UnknownLoc>(name_loc.getChildLoc())) {
      loc_stream << name_loc.getName();
    } else {
      loc.print(loc_stream);
    }
  }
  return loc_str;
}

bool IsSplitDropTransferred(FragmentOp fragment) {
  return fragment->hasAttr(kSplitDropTransferredAttrName);
}

bool IsSplitKeepTransferred(FragmentOp fragment) {
  return fragment->hasAttr(kSplitKeepTransferredAttrName);
}

namespace detail {
namespace {

// Verifies that the kControlOperandStartIdxAttrName attribute, if present,
// correctly identifies where control operands begin in the operand list.
//
// Context: Dynamic block arguments correspond to operands, while static block
// arguments don't correspond to operands (e.g., used for config/metadata).
// In practice, num_static_args is typically 0. The number of dynamic args is
// obtained by taking all block arguments and dropping the first
// num_static_args.
//
// The operand list layout is: [operands for dynamic args][control operands]
// This function checks that control_start_idx == num_dynamic_args, ensuring
// control operands come directly after operands corresponding to dynamic
// arguments, preventing accidental processing of control operands during merge.
void VerifyControlOperandIndex(Operation* op,
                               ArrayRef<BlockArgument> dynamic_args,
                               int num_static_args,
                               const std::string& op_type) {
  auto control_start_attr =
      op->getAttrOfType<IntegerAttr>(kControlOperandStartIdxAttrName);
  if (control_start_attr) {
    int control_start_idx = control_start_attr.getInt();
    int num_dynamic_args = dynamic_args.size();
    SDY_CHECK_EQ(control_start_idx, num_dynamic_args)
        << "Control operands should come directly after dynamic arguments "
           "operands. Found control_operand_start_idx = "
        << control_start_idx << ", num static arguments = " << num_static_args
        << ", num dynamic arguments = " << dynamic_args.size() << "for "
        << op_type << " " << llvm::to_string(op->getName())
        << ". Contact MPMD team for support.";
  }
}

// Add all unique operands of the producer op to `new_operands` and map them
// to the corresponding block argument. Erase all arguments whose
// corresponding operand is already mapped to another argument, and replace
// their uses with the mapped argument.
void ProcessProducerOperands(Operation* producer_op, Block& producer_block,
                             RewriterBase& rewriter, int num_static_args,
                             SmallVector<Value>& new_operands,
                             IRMapping& mapping) {
  ArrayRef<BlockArgument> dynamic_producer_args =
      producer_block.getArguments().drop_front(num_static_args);

  VerifyControlOperandIndex(producer_op, dynamic_producer_args, num_static_args,
                            "producer_op");

  llvm::BitVector erase_args(producer_block.getNumArguments());
  for (auto [operand, arg] :
       llvm::zip(producer_op->getOperands(), dynamic_producer_args)) {
    if (Value mapped_arg = mapping.lookupOrNull(operand)) {
      rewriter.replaceAllUsesWith(arg, mapped_arg);
      erase_args.set(arg.getArgNumber());
    } else {
      new_operands.push_back(operand);
      mapping.map(operand, arg);
    }
  }
  producer_block.eraseArguments(erase_args);
}

// Add all unique operands of the consumer op, that aren't the result of the
// producer op, to `new_operands` and returns a replacement value for each
// dynamic argument as follows:
// - if the operand is the result of the producer op, add the corresponding
//   return operand of the producer op to the result.
// - otherwise if the operand is already mapped to a block argument, add that
//   argument to the result.
// - otherwise, add that operand to `new_operands`, create a corresponding block
//   argument in `producer_block` and add it to the result, and map the operand
//   to the new block argument.
SmallVector<Value> ProcessConsumerOperands(
    Operation* consumer_op, Block& consumer_block, Operation* producer_op,
    Block& producer_block, int num_static_args,
    SmallVector<Value>& new_operands, IRMapping& mapping) {
  SmallVector<Value> new_consumer_args;
  new_consumer_args.reserve(consumer_block.getNumArguments());

  // Add all static arguments from the producer op to `new_consumer_args`.
  llvm::copy(producer_block.getArguments().take_front(num_static_args),
             std::back_inserter(new_consumer_args));

  Operation* return_op = producer_block.getTerminator();
  ArrayRef<BlockArgument> dynamic_consumer_args =
      consumer_block.getArguments().drop_front(num_static_args);

  VerifyControlOperandIndex(consumer_op, dynamic_consumer_args, num_static_args,
                            "consumer_op");

  for (auto [operand, arg] :
       llvm::zip(consumer_op->getOperands(), dynamic_consumer_args)) {
    if (auto op_result = dyn_cast<OpResult>(operand);
        op_result && op_result.getOwner() == producer_op) {
      new_consumer_args.push_back(
          return_op->getOperand(op_result.getResultNumber()));
    } else if (Value mapped_arg = mapping.lookupOrNull(operand)) {
      new_consumer_args.push_back(mapped_arg);
    } else {
      new_operands.push_back(operand);
      new_consumer_args.push_back(
          producer_block.addArgument(arg.getType(), operand.getLoc()));
      mapping.map(operand, new_consumer_args.back());
    }
  }

  return new_consumer_args;
}

// Returns all the results of the producer op that are not just used by the
// consumer, adds their type to `new_result_types`, and adds the corresponding
// return operands to `new_return_operands`.
//
// In addition, replace all uses of the producer op that were nested in the
// consumer block (we assume the consumer block was already merged into the
// producer block) using the provided `replace_producer_use_in_consumer_block`.
SmallVector<Value> ProcessProducerResults(
    Operation* producer_op, Operation* consumer_op,
    Operation* producer_return_op, RewriterBase& rewriter,
    SmallVector<Type>& new_result_types,
    SmallVector<Value>& new_return_operands,
    std::function<void(OpOperand&, Value)>
        replace_producer_use_in_consumer_block) {
  SmallVector<Value> producer_results_to_replace;
  producer_results_to_replace.reserve(producer_op->getNumResults());

  for (auto [result, return_operand] : llvm::zip(
           producer_op->getResults(), producer_return_op->getOperands())) {
    // We need to replace any nested uses of the result in the merged block
    // before calling consumer_op.
    for (OpOperand& use : llvm::make_early_inc_range(result.getUses())) {
      if (producer_op->isProperAncestor(use.getOwner())) {
        replace_producer_use_in_consumer_block(use, return_operand);
      }
    }

    if (HasOtherUsersExcept(result, consumer_op)) {
      producer_results_to_replace.push_back(result);
      new_result_types.push_back(result.getType());
      new_return_operands.push_back(return_operand);
    }
  }

  return producer_results_to_replace;
}

// Returns the control operands for the merged result of (lhs_op, rhs_op).
//
// Control operands are identified by the kControlOperandStartIdxAttrName
// attribute. This function filters out control operands that reference the
// other fragment being merged, as those dependencies will no longer be valid
// after the merge.
//
// The collected control operands are preserved on the merged fragment to
// maintain scheduling constraints across the merge operation.
SmallVector<Value> CollectControlOperands(Operation* lhs_op,
                                          Operation* rhs_op) {
  SmallVector<Value> control_operands;

  if (!lhs_op || !rhs_op) {
    return control_operands;
  }

  // Collect control operands from lhs_op.
  auto lhs_attr =
      lhs_op->getAttrOfType<IntegerAttr>(kControlOperandStartIdxAttrName);
  if (lhs_attr) {
    int control_start_idx = lhs_attr.getInt();
    for (int i = control_start_idx; i < lhs_op->getNumOperands(); ++i) {
      Value operand = lhs_op->getOperand(i);
      // Don't add control operands that come from rhs_op since they
      // will no longer be valid after the merge.
      if (operand.getDefiningOp() != rhs_op) {
        control_operands.push_back(operand);
      }
    }
  }

  // Collect control operands from rhs_op
  auto rhs_attr =
      rhs_op->getAttrOfType<IntegerAttr>(kControlOperandStartIdxAttrName);
  if (rhs_attr) {
    int control_start_idx = rhs_attr.getInt();
    for (int i = control_start_idx; i < rhs_op->getNumOperands(); ++i) {
      Value operand = rhs_op->getOperand(i);
      // Don't add control operands that come from lhs_op since they
      // will no longer be valid after the merge.
      if (operand.getDefiningOp() != lhs_op) {
        control_operands.push_back(operand);
      }
    }
  }

  return control_operands;
}

}  // namespace

Operation* MergeRegionOps(
    Operation* producer_op, Operation* consumer_op, RewriterBase& rewriter,
    int num_static_args,
    std::function<void(OpOperand&, Value)>
        replace_producer_use_in_consumer_block,
    std::function<Operation*(Location, TypeRange, ValueRange)>
        create_merged_op) {
  SDY_CHECK_EQ(producer_op->getNumRegions(), 1);
  SDY_CHECK_EQ(consumer_op->getNumRegions(), 1);
  Block& producer_block = producer_op->getRegion(0).front();
  Block& consumer_block = consumer_op->getRegion(0).front();

  // Fast path if the producer is trivial to avoid cloning argument lists.
  // This is a bit of a special case, but is very effective since we typically
  // merge in a bottom-up fashion.
  if (producer_op->hasOneUse() && producer_block.getNumArguments() == 1 &&
      // NB: despite the naming, the consumer_op may not be a true consumer.
      // This is WAI, although naming this as consumer_op is a bit misleading.
      // TODO(dvytin): rename consistently in all merge variants.
      *producer_op->getUsers().begin() == consumer_op &&
      producer_op->getNumResults() == 1 && num_static_args == 0) {
    // Find where is the producer op used.
    int64_t operand_index = producer_op->getUses().begin()->getOperandNumber();

    auto fused_loc =
        rewriter.getFusedLoc({producer_op->getLoc(), consumer_op->getLoc()});
    // It is ugly to have to set the insertion point here, but typically the
    // callback will capture the same rewriter (!) so we have to set the
    // insertion point before we call the callback. This is sad. We should be
    // passing the rewriter ref to the callback.
    rewriter.setInsertionPoint(consumer_op);
    // It is also a little sad we cannot reuse the consumer_op entirely in place
    // and have to allocate operand lists for it. But the callback deals with
    // setting the attributes correctly in the fused op. This deserves some
    // refactoring to decouple the expensive parts and allow us to fully reuse
    // the consumer_op, in-place.
    //
    // TODO(dvytin): refactor the callbacks to MergeRegionOps.
    Operation* fused_op = create_merged_op(
        fused_loc, consumer_op->getResultTypes(), consumer_op->getOperands());
    fused_op->getRegion(0).takeBody(consumer_op->getRegion(0));
    Block& fused_block = fused_op->getRegion(0).front();
    // Replace the operand with the producer op operand.
    fused_op->setOperand(operand_index, producer_op->getOperand(0));
    Operation* terminator = producer_block.getTerminator();
    Value yielded_value = terminator->getOperand(0);
    BlockArgument arg = fused_block.getArgument(operand_index);
    rewriter.replaceAllUsesWith(arg, yielded_value);
    arg.setType(producer_block.getArgument(0).getType());
    rewriter.inlineBlockBefore(&producer_block, &fused_block.front(), arg);
    rewriter.eraseOp(terminator);
    rewriter.replaceOp(consumer_op, fused_op->getResults());
    rewriter.eraseOp(producer_op);
    return fused_op;
  }

  SmallVector<Value> new_operands;
  new_operands.reserve(producer_op->getNumOperands() +
                       consumer_op->getNumOperands());
  IRMapping mapping;

  ProcessProducerOperands(producer_op, producer_block, rewriter,
                          num_static_args, new_operands, mapping);

  SmallVector<Value> new_consumer_args = ProcessConsumerOperands(
      consumer_op, consumer_block, producer_op, producer_block, num_static_args,
      new_operands, mapping);

  // Collect control operands from both fragments to preserve scheduling
  // constraints across the merge. Control operands that reference the other
  // fragment being merged are filtered out since they'll be invalid.
  SmallVector<Value> control_operands =
      CollectControlOperands(producer_op, consumer_op);

  int control_operand_start_index = new_operands.size();
  llvm::append_range(new_operands, control_operands);

  // We take the producer return op before merging the blocks.
  Operation* producer_return_op = producer_block.getTerminator();

  // Inline the consumer block at the end of the producer block, to get a merged
  // block.
  rewriter.mergeBlocks(&consumer_block, &producer_block, new_consumer_args);

  int max_num_results =
      producer_op->getNumResults() + consumer_op->getNumResults();
  SmallVector<Type> new_result_types;
  SmallVector<Value> new_return_operands;
  new_result_types.reserve(max_num_results);
  new_return_operands.reserve(max_num_results);

  SmallVector<Value> producer_results_to_replace = ProcessProducerResults(
      producer_op, consumer_op, producer_return_op, rewriter, new_result_types,
      new_return_operands, replace_producer_use_in_consumer_block);

  // Now we can erase the return op of the producer op as we'll need to create a
  // new return op for the merged block.
  rewriter.eraseOp(producer_return_op);

  // Add all result types of the consumer op to `new_result_types`.
  llvm::copy(consumer_op->getResultTypes(),
             std::back_inserter(new_result_types));

  // Add all return operands of the consumer op to `new_return_operands`. Note
  // that this needs to be done after the call to `mergeBlocks` because the
  // block arguments have been replaced for the consumer block.
  Operation* return_op = producer_block.getTerminator();
  llvm::copy(return_op->getOperands(), std::back_inserter(new_return_operands));

  // Set the operands of the return op to those of the merged block.
  return_op->setOperands(new_return_operands);

  // Finally create the merged op with a fused location right before consumer
  // op, take the merged block from the producer, and replace the results of
  // both the producer and consumer ops with the corresponding results of the
  // merged op.
  rewriter.setInsertionPoint(consumer_op);
  Location fused_loc =
      rewriter.getFusedLoc({producer_op->getLoc(), consumer_op->getLoc()});
  Operation* new_op =
      create_merged_op(fused_loc, new_result_types, new_operands);

  // Place the new control operand start index attribute on the merged operation
  // so that control dependencies persist across merges until they are
  // explicitly removed by RemoveAllControlDependencies().
  if (!control_operands.empty()) {
    new_op->setAttr(kControlOperandStartIdxAttrName,
                    IntegerAttr::get(IntegerType::get(new_op->getContext(), 64),
                                     control_operand_start_index));
  }

  new_op->getRegion(0).takeBody(producer_op->getRegion(0));

  for (auto [old_result, new_result] :
       llvm::zip_first(producer_results_to_replace, new_op->getResults())) {
    rewriter.replaceAllUsesExcept(old_result, new_result, consumer_op);
  }
  rewriter.replaceOp(consumer_op, new_op->getResults().take_back(
                                      consumer_op->getNumResults()));
  rewriter.eraseOp(producer_op);

  return new_op;
}

}  // namespace detail
}  // namespace mlir::mpmd
