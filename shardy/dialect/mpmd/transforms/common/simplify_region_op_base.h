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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_COMMON_SIMPLIFY_REGION_OP_BASE_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_COMMON_SIMPLIFY_REGION_OP_BASE_H_

#include <functional>

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::mpmd {

using SimplifiedRegionOpCreateFn = std::function<Operation*(
    TypeRange result_types, ValueRange operands, BitVector erased_results)>;

// Simplifies the given `op`. In particular, it:
//  - deduplicates results, and their corresponding return values;
//  - deduplicates operands, and their corresponding block arguments, if the op
//    has any operands;
//  - removes results whose corresponding return operand is a block argument of
//    the op;
//  - removes operands whose corresponding block argument has no more uses (or
//    didn't have any to begin with); and
//  - removes results that are unused.
//
// NOTE: This method assumes that the op has the same number of results and
// return values, and if the op has any operands we also assume that it has the
// same number of operands and block arguments.
LogicalResult SimplifyRegionOp(Operation* op, PatternRewriter& rewriter,
                               SimplifiedRegionOpCreateFn create_op);

// A base class for patterns that simplify a given op. See SimplifyRegionOp for
// more information.
template <class OpTy>
class SimplifyRegionOpPatternBase : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const final {
    return SimplifyRegionOp(
        op, rewriter,
        [&, this](TypeRange result_types, ValueRange operands,
                  BitVector erased_results) -> Operation* {
          return createNewOp(op, rewriter, result_types, operands,
                             erased_results);
        });
  }

 protected:
  virtual OpTy createNewOp(OpTy op, PatternRewriter& rewriter,
                           TypeRange result_types, ValueRange operands,
                           BitVector erased_results) const = 0;
};

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_COMMON_SIMPLIFY_REGION_OP_BASE_H_
