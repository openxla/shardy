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

#ifndef SHARDY_DIALECT_MPMD_IR_DIALECT_H_
#define SHARDY_DIALECT_MPMD_IR_DIALECT_H_

// IWYU pragma: begin_keep

#include <cstdint>
#include <functional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

// IWYU pragma: end_keep

namespace mlir::mpmd {

// A function that adds operations to the body of a FragmentOp. The operations
// must be non-mpmd operations (e.g. StableHLO operations). The function is
// returning the final values that callers of this method should add as
// terminator operands of the FragmentOp they are building.
using FragmentOpBodyPopulator = std::function<SmallVector<Value>(
    ArrayRef<Value> args, OpBuilder& block_builder)>;

// A function that adds operations to the body of a ForOp using the provided
// block arguments and returns the values that should be returned by the body's
// terminator (which will be added after calling this function).
using ForOpBodyPopulator = std::function<SmallVector<Value>(
    ArrayRef<Value> args, Value index, OpBuilder& block_builder)>;

// Parses an optional transpose count.
mlir::ParseResult parseOptionalTransposeCount(AsmParser& parser,
                                              int64_t& transpose_count);

// Prints an optional transpose count.
void printOptionalTransposeCount(AsmPrinter& printer, int64_t transpose_count);

}  // namespace mlir::mpmd

// IWYU pragma: begin_exports

// Dialect main class is defined in ODS, we include it here.
#include "shardy/dialect/mpmd/ir/dialect.h.inc"

// Include the auto-generated header file containing type declarations. (This
// has to go before including `ops.h.inc` since the MPMD operation definitions
// require types and attrs to be defined.)

// ODS-generated enum classes.
#include "shardy/dialect/mpmd/ir/enums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "shardy/dialect/mpmd/ir/attrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "shardy/dialect/mpmd/ir/types.h.inc"

// Include the auto-generated header file containing op declarations.
#define GET_OP_CLASSES
#include "shardy/dialect/mpmd/ir/ops.h.inc"

// IWYU pragma: end_exports

#endif  // SHARDY_DIALECT_MPMD_IR_DIALECT_H_
