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
#include "mlir/IR/Attributes.h"
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

void printMeshOrRef(AsmPrinter& printer, Attribute meshOrRef) {
  if (auto mesh = dyn_cast<MeshAttr>(meshOrRef)) {
    printer << MeshAttr::getMnemonic();
    printer.printStrippedAttrOrType(mesh);
  } else {
    printer << meshOrRef;
  }
}

namespace {

void printOptionalNamedAxisList(AsmPrinter& printer, StringRef keyword,
                                ArrayRef<AxisRefAttr> axisList) {
  if (!axisList.empty()) {
    printer << ", " << keyword << "={";
    printer.printStrippedAttrOrType(axisList);
    printer << "}";
  }
}

}  // namespace

void printReplicatedAndUnreducedAxes(AsmPrinter& printer,
                                     ArrayRef<AxisRefAttr> replicatedAxes,
                                     ArrayRef<AxisRefAttr> unreducedAxes) {
  printOptionalNamedAxisList(printer, "replicated", replicatedAxes);
  printOptionalNamedAxisList(printer, "unreduced", unreducedAxes);
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

void printFactorsWithType(AsmPrinter& printer, ArrayRef<int64_t> factors,
                          StringRef type) {
  if (factors.empty()) {
    return;
  }
  printer << " " << type << "={";
  llvm::interleaveComma(factors, printer, [&](int64_t factor) {
    printer << factorSymbolString(factor);
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

void printEdgeValueRef(AsmPrinter& printer, EdgeNodeType type, int64_t index) {
  printer << stringifyEdgeNodeType(type) << "_" << index;
}

void printStepIndex(AsmPrinter& printer, int64_t stepIndex) {
  printer << "step_" << stepIndex;
}

void ConstantOp::print(OpAsmPrinter& p) {
  hlo::printConstantOp(p, getOperation(), getValue());
}

}  // namespace sdy
}  // namespace mlir
