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

#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/fragment_arg_res_attrs.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/export/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/export/utils.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_MARKALIASINGANDDONATIONPASS
#include "shardy/dialect/mpmd/transforms/export/passes.h.inc"

namespace {

// If the input can be aliased with an output, then it returns the index of the
// output it can be aliased with. An index cannot be aliased to an output if
// the output index has already been used for aliasing (i.e., it is present in
// the `aliased_outputs` set)
std::optional<unsigned int> FindAliasingOutput(
    FragmentOp op, unsigned int input_index,
    const DenseSet<unsigned int>& aliased_outputs) {
  Type input_type = op->getOperand(input_index).getType();
  for (OpResult output : op->getResults()) {
    unsigned output_index = output.getResultNumber();
    // The input can only be aliased with an output if
    // 1. They are the same type,
    // 2. the output has not been aliased yet, and
    // 3. the output is not on host memory.
    // 4. the input and output have the same layout.
    if (output.getType() != input_type || IsResultOnHost(output) ||
        !IsInputOutputLayoutMatch(op, input_index, output_index)) {
      continue;
    }
    if (!aliased_outputs.contains(output_index)) {
      return output_index;
    }
  }
  return std::nullopt;
}

// Constructs a set of indices of donated inputs and a map between an input
// index to an output index that it can be  aliased within a given region-based
// `op`, given `donatable_input_indices`, a vector of input indices whose last
// usage is the given operation, `aliased_block_args` a set of block
// arguments the user has marked to be aliased, and `donated_block_args` a set
// of block arguments the user has marked to be donated.
std::pair<DenseSet<unsigned int>, DenseMap<unsigned int, unsigned int>>
ConstructDonationSetAndIOAliasingMap(
    FragmentOp op, ArrayRef<unsigned int> donatable_input_indices,
    const DenseSet<BlockArgument>& aliased_block_args,
    const DenseSet<BlockArgument>& donated_block_args) {
  DenseSet<unsigned int> donated_input_indices_set;
  DenseMap<unsigned int, unsigned int> input_output_aliasing_map;

  // Keep track of which outputs have been aliased. Each output can be aliased
  // at most once.
  DenseSet<unsigned int> aliased_outputs;

  for (unsigned int input_index : donatable_input_indices) {
    // Don't donate values which are on host.
    if (IsArgOnHost(op, input_index)) {
      continue;
    }
    Value input_value = op->getOperand(input_index);

    // Do not donate inputs that are used in transfer ops because transfer and
    // fragment ops could overlap which makes the input not suitable for
    // donation.
    if (sdy::hasAnyUserOfType<TransferOp>(input_value)) {
      continue;
    }

    if (auto block_arg = dyn_cast<BlockArgument>(input_value)) {
      // Only donate/alias a block argument if the user has marked it to
      // be donated/aliased.
      if (donated_block_args.contains(block_arg)) {
        donated_input_indices_set.insert(input_index);
      } else if (aliased_block_args.contains(block_arg)) {
        if (auto aliased_output_idx =
                FindAliasingOutput(op, input_index, aliased_outputs)) {
          input_output_aliasing_map[input_index] = *aliased_output_idx;
          SDY_CHECK(aliased_outputs.insert(*aliased_output_idx).second);
        } else {
          // Donate the input if no aliasing output was found.
          donated_input_indices_set.insert(input_index);
        }
      }
    } else {
      // Try to find an alias. If no alias is found, then donate the input.
      if (auto aliased_output_idx =
              FindAliasingOutput(op, input_index, aliased_outputs)) {
        input_output_aliasing_map[input_index] = *aliased_output_idx;
        SDY_CHECK(aliased_outputs.insert(*aliased_output_idx).second);
      } else {
        donated_input_indices_set.insert(input_index);
      }
    }
  }
  return std::make_pair(donated_input_indices_set, input_output_aliasing_map);
}

void MarkOperandsForAliasingAndDonation(func::FuncOp main_func,
                                        MLIRContext* ctx) {
  DenseSet<BlockArgument> aliased_entrypoint_func_args =
      GetAliasedBlockArguments(main_func);
  DenseSet<BlockArgument> donated_entrypoint_func_args =
      GetDonatedBlockArguments(main_func);
  DenseMap<Operation*, SmallVector<unsigned int>> donation_candidates =
      OperandsForDeletionMapping(main_func);
  IRRewriter rewriter(ctx);

  for (auto [op, donatable_input_indices] : donation_candidates) {
    // Only analyze fragment ops because they are the only ops converted to
    // functions and we can set the aliasing or donation attributes.
    auto fragment_op = dyn_cast<FragmentOp>(op);
    if (!fragment_op) {
      continue;
    }
    // Sort the indices to ensure a deterministic order.
    llvm::sort(donatable_input_indices);

    // Step 1: Add the indices of the donated block arguments.
    // Step 2: Construct the aliasing map. The inputs that are not aliasable
    // will be donated.
    std::pair<DenseSet<unsigned int>, DenseMap<unsigned int, unsigned int>>
        donated_idx_set_and_io_aliasing_map =
            ConstructDonationSetAndIOAliasingMap(
                fragment_op, donatable_input_indices,
                aliased_entrypoint_func_args, donated_entrypoint_func_args);

    // Only set the donation and aliasing attributes if there is anything to
    // alias or donate.
    if (donated_idx_set_and_io_aliasing_map.first.empty() &&
        donated_idx_set_and_io_aliasing_map.second.empty()) {
      continue;
    }

    SmallVector<Attribute> aliasing_and_donation_attributes =
        GetArgAttrsOrCreateDefault(fragment_op);
    for (auto input_index : donated_idx_set_and_io_aliasing_map.first) {
      InsertAttr(aliasing_and_donation_attributes[input_index],
                 kBufferDonationAttrName, rewriter.getBoolAttr(true));
    }
    for (auto [input_index, output_index] :
         donated_idx_set_and_io_aliasing_map.second) {
      InsertAttr(aliasing_and_donation_attributes[input_index],
                 kAliasingAttrName, rewriter.getI32IntegerAttr(output_index));
    }
    SetArgAttrs(op, aliasing_and_donation_attributes);
  }
}

class MarkAliasingAndDonationPass
    : public impl::MarkAliasingAndDonationPassBase<
          MarkAliasingAndDonationPass> {
  using MarkAliasingAndDonationPassBase::MarkAliasingAndDonationPassBase;

 private:
  void runOnFunc(func::FuncOp main_func) override {
    if (IsMpmdFunction(main_func)) {
      MarkOperandsForAliasingAndDonation(main_func, main_func.getContext());
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
