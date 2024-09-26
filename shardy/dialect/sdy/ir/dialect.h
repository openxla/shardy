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

#ifndef SHARDY_DIALECT_SDY_IR_DIALECT_H_
#define SHARDY_DIALECT_SDY_IR_DIALECT_H_

// IWYU pragma: begin_keep

#include <cstdint>
#include <functional>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"

// IWYU pragma: end_keep

// IWYU pragma: begin_exports

// Dialect main class is defined in ODS, we include it here.
#include "shardy/dialect/sdy/ir/dialect.h.inc"
// ODS-generated attribute classes.
#define GET_ATTRDEF_CLASSES
#include "shardy/dialect/sdy/ir/attrs.h.inc"
// ODS-generated enum classes.
#include "shardy/dialect/sdy/ir/enums.h.inc"

// Below are methods that are the bodies of ODS-generated op-interface classes
// which cannot be inlined due to cyclic dependencies on helper functions.
namespace mlir {
namespace sdy {
namespace details {

// Default implementation of the `getOpResultEdgeOwnerShardings` method of
// `ShardableDataFlowOpInterface`.
ArrayRef<TensorShardingAttr> getOpResultEdgeOwnerShardingsImpl(Operation* op);

// Default implementation of the `setOpResultEdgeOwnerSharding` method of
// `ShardableDataFlowOpInterface`.
void setOpResultEdgeOwnerShardingImpl(Operation* op, unsigned index,
                                      TensorShardingAttr sharding);

// Default implementation of the `setOpResultEdgeOwnerShardings` method of
// `ShardableDataFlowOpInterface`.
void setOpResultEdgeOwnerShardingsImpl(Operation* op,
                                       ArrayRef<TensorShardingAttr> shardings);

}  // namespace details
}  // namespace sdy
}  // namespace mlir

// ODS-generated op-interface classes.
#include "shardy/dialect/sdy/ir/op_interface.h.inc"
// ODS-generated op classes.
#define GET_OP_CLASSES
#include "shardy/dialect/sdy/ir/ops.h.inc"

// IWYU pragma: end_exports

#endif  // SHARDY_DIALECT_SDY_IR_DIALECT_H_
