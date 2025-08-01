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

#ifndef SHARDY_DIALECT_SDY_IR_PRINTERS_H_
#define SHARDY_DIALECT_SDY_IR_PRINTERS_H_

#include <cassert>
#include <cstdint>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

void printMeshOrRef(AsmPrinter& printer, Attribute meshOrRef);

// Prints each optional axis list as ", <keyword>={<axes>}", in a predefined
// order if multiple are present.
void printReplicatedAndUnreducedAxes(AsmPrinter& printer,
                                     ArrayRef<AxisRefAttr> replicatedAxes,
                                     ArrayRef<AxisRefAttr> unreducedAxes);

// Prints the factor sizes of an OpShardingRule. The keys in the list are the
// indices in the factor mapping, with i=0, j=1, k=2,... z=17. For any index
// greater than 17 it's printed as z_X where `X>0` and the index being X-17
// (for example `z_1`, `z_123`, etc.). For example a list of `{6, 2, 4}` is
// printed as `{i=6, j=2, k=4}`.
void printFactorSizes(AsmPrinter& printer, ArrayRef<int64_t> factorSizes);

// Prints the `factors` of `type` in an OpShardingRule. Given a vector [0, 2],
// we print `type={i, k}`.
void printFactorsWithType(AsmPrinter& printer, ArrayRef<int64_t> factors,
                          StringRef type);

void printIsCustomRule(AsmPrinter& printer, bool isCustomRule);

// Prints a single block region without the block id, for example:
//
// (%blockArg1, ..., %blockArgM) {
//   // ops in the block
// }
//
// This is needed for using `custom<SingleBlockRegionNoBlockId>` in
// `assemblyFormat`.
void printSingleBlockRegionNoBlockId(OpAsmPrinter& printer, Operation*,
                                     Region& region);

// Prints the TensorShardingPerValueAttr without the outside <>.
//
// The default assemblyFormat of TensorShardingPerValueAttr would have us
// print it as:
//
// <[<@mesh, ...>, ..., <@mesh, ...>]>
//
// In some ops we want to avoid the extra <> so we have a custom parser/printer
// for it. So we get the following instead:
//
// [<@mesh, ...>, ..., <@mesh, ...>]
//
// This is needed for using `custom<StrippedTensorShardingPerValueAttr>` in
// `assemblyFormat`.
void printStrippedTensorShardingPerValueAttr(
    AsmPrinter& printer, Operation* op,
    TensorShardingPerValueAttr shardingPerValue);

// Prints a minus symbol. Pass in any string for `StringRef` as it is not used.
// TODO(bartchr): figure out how we can avoid requiring a StringRef. It makes
// the assembly format a bit ugly having to pass in an empty string. Issue seems
// to be MLIR tblgen requires 2 arguments for a custom parser/printer.
void printMinus(AsmPrinter& printer, StringRef);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_IR_PRINTERS_H_
