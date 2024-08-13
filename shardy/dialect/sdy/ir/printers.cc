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

#include "shardy/dialect/sdy/ir/printers.h"

#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/AssemblyFormat.h"

namespace mlir {
namespace sdy {

void MeshAttr::print(AsmPrinter& printer) const {
  printer << "<";
  if (getDeviceId().has_value()) {
    printer << "device_id=" << *getDeviceId();
  } else {
    llvm::interleaveComma(getAxes(), printer,
                          [&](MeshAxisAttr axis) { axis.print(printer); });
  }
  printer << ">";
}

void DimensionShardingAttr::print(AsmPrinter& printer) const {
  printer << "{";
  printer.printStrippedAttrOrType(getAxes());
  if (!getIsClosed()) {
    // Print {"a", ?}, but never {, ?}
    if (!emptyAxes()) {
      printer << ", ";
    }
    printer << "?";
  }
  printer << "}";
  if (getPriority()) {
    printer << "p" << getPriority().value();
  }
}

void DimMappingAttr::print(AsmPrinter& printer) const {
  for (int64_t factorIndex : getFactorIndices()) {
    printer << factorSymbolString(factorIndex);
  }
}

void printFactorSizes(AsmPrinter& printer, ArrayRef<int64_t> factorSizes) {
  if (factorSizes.empty()) {
    return;
  }
  printer << " {";
  int i = 0;
  llvm::interleaveComma(factorSizes, printer, [&](int64_t factorSize) {
    printer << factorSymbolString(i++) << "=" << factorSize;
  });
  printer << "}";
}

void printIsCustomRule(AsmPrinter& printer, bool isCustomRule) {
  if (isCustomRule) {
    printer << ", custom";
  }
}

void printSingleBlockRegionNoBlockId(OpAsmPrinter& printer, Operation*,
                                     Region& region) {
  // Print body arguments manually to avoid printing the block ID.
  printer << "(";
  llvm::interleaveComma(
      region.getArguments(), printer,
      [&](BlockArgument regionArg) { printer.printRegionArgument(regionArg); });
  printer << ") ";

  printer.printRegion(region, /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

void printStrippedTensorShardingPerValueAttr(
    AsmPrinter& printer, Operation*,
    TensorShardingPerValueAttr shardingPerValue) {
  printer << "[";
  printer.printStrippedAttrOrType(shardingPerValue.getShardings());
  printer << "]";
}

void ConstantOp::print(OpAsmPrinter& p) {
  hlo::printConstantOp(p, getOperation(), getValue());
}

}  // namespace sdy
}  // namespace mlir
