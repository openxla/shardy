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

#include <cstddef>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"  // IWYU pragma: keep

namespace mlir::mpmd {

#define GEN_PASS_DEF_FRAGMENTDEDUPPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

namespace {

// For each return operand in index `i`, finds its first occurrence
// setting `first_indices[i]` to the index of the first occurrence. For example,
// if values is [a, b, a, c, d, a] then the returned first_indices will be [0,
// 1, 0, 2, 3, 0].
//
// Two return operands are considered duplicates if they are the same value and
// the respective fragment results have the same type.
//
// Returns true if there are duplicate values in the fragment's return.
bool FindDuplicateReturnOperands(FragmentOp fragment,
                                 SmallVector<unsigned>& first_indices) {
  bool found_duplicate = false;
  ValueRange values = fragment.getBody()->getTerminator()->getOperands();
  TypeRange types = fragment.getResultTypes();
  DenseMap<std::pair<Value, Type>, size_t> index_of_first_occurrence;
  for (auto [i, value_and_type] : llvm::enumerate(llvm::zip(values, types))) {
    auto value_and_type_key = std::make_pair(std::get<0>(value_and_type),
                                             std::get<1>(value_and_type));
    auto it = index_of_first_occurrence.find(value_and_type_key);
    if (it == index_of_first_occurrence.end()) {
      // Found the first occurrence of a value. Insert it into the map and add
      // its index to first_indices.
      index_of_first_occurrence.try_emplace(value_and_type_key, i);
      first_indices.push_back(i);
    } else {
      // Found a duplicate value. Add the index of its first occurrence to
      // first_indices.
      found_duplicate = true;
      first_indices.push_back(it->second);
    }
  }
  return found_duplicate;
}

// Erases any duplicate results.
void RemoveDuplicatedResults(FragmentOp fragment, RewriterBase& rewriter) {
  SmallVector<unsigned> index_of_first_occurrence;
  index_of_first_occurrence.reserve(fragment.getNumResults());
  if (!FindDuplicateReturnOperands(fragment, index_of_first_occurrence)) {
    return;
  }

  for (OpResult result : fragment.getResults()) {
    unsigned index = result.getResultNumber();
    unsigned index_to_replace = index_of_first_occurrence[index];
    if (index != index_to_replace) {
      rewriter.replaceAllUsesWith(result, fragment.getResult(index_to_replace));
    }
  }
}

// For each value `fragment.getOperands()[i]`, finds its first occurrence
// setting `first_indices[i]` to the index of the first occurrence. For example,
// if the operands are [a, b, a, c, d, a] then the returned first_indices will
// be [0, 1, 0, 2, 3, 0].
//
// The main difference from the logic in `FindDuplicateReturnOperands` is that
// this function does not take types into account: if a value is passed as
// operand to the fragment twice, then we are guaranteed that there two operand
// types are the same.
//
// Returns true if there are duplicate values in `values`.
bool FindDuplicateArguments(FragmentOp fragment,
                            SmallVector<unsigned>& first_indices) {
  bool found_duplicate = false;
  DenseMap<Value, size_t> index_of_first_occurrence;
  for (auto [i, value] : llvm::enumerate(fragment.getOperands())) {
    auto it = index_of_first_occurrence.find(value);
    if (it == index_of_first_occurrence.end()) {
      // Found the first occurrence of a value. Insert it into the map and add
      // its index to first_indices.
      index_of_first_occurrence.insert({value, i});
      first_indices.push_back(i);
    } else {
      // Found a duplicate value. Add the index of its first occurrence to
      // first_indices.
      found_duplicate = true;
      first_indices.push_back(it->second);
    }
  }
  return found_duplicate;
}

// Erases any duplicated operands.
void RemoveDuplicatedOperands(FragmentOp fragment, RewriterBase& rewriter) {
  Block& block = fragment.getRegion().front();

  SmallVector<unsigned> index_of_first_occurrence;
  index_of_first_occurrence.reserve(fragment.getNumOperands());
  if (!FindDuplicateArguments(fragment, index_of_first_occurrence)) {
    return;
  }

  for (OpOperand& operand : fragment->getOpOperands()) {
    unsigned operand_index = operand.getOperandNumber();
    unsigned index_to_replace = index_of_first_occurrence[operand_index];
    if (operand_index != index_to_replace) {
      rewriter.replaceAllUsesWith(block.getArgument(operand_index),
                                  block.getArgument(index_to_replace));
    }
  }
}

// This pass deduplicates operands and results of fragments.
class FragmentDedupPass
    : public impl::FragmentDedupPassBase<FragmentDedupPass> {
  using FragmentDedupPassBase::FragmentDedupPassBase;

 private:
  void runOnFunc(func::FuncOp func_op) override {
    IRRewriter rewriter(func_op.getContext());
    func_op.walk<WalkOrder::PreOrder>([&rewriter](FragmentOp fragment) {
      RemoveDuplicatedOperands(fragment, rewriter);
      RemoveDuplicatedResults(fragment, rewriter);
      // We know that fragments cannot be nested in fragments, so there's
      // nothing to visit in the body of `fragment`.
      return WalkResult::skip();
    });
  }
};

}  // namespace
}  // namespace mlir::mpmd
