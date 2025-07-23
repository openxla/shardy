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

// MLIR `opt` tool for driving MPMD transformations.
//
// Usage:
//   mpmd_opt <file> <llvm options>

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "shardy/dialect/mpmd/ir/register.h"
#include "shardy/dialect/sdy/transforms/passes.h"
#include "shardy/dialect/mpmd/transforms/passes.h"


int main(int argc, char** argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::mpmd::registerAllDialects(registry);
  registry.insert<mlir::quant::QuantDialect>();
  mlir::func::registerAllExtensions(registry);

  // Register all SDY passes and pipelines.
  mlir::sdy::registerAllSdyPassesAndPipelines();

  // Register all MPMD passes and pipelines.
  mlir::mpmd::registerAllMpmdPassesAndPipelines();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MPMD pass driver\n", registry));
}
