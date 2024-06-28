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

// MLIR `opt` tool for driving SDY transformations.
//
// Usage:
//   sdy_opt <file> <llvm options>

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/passes.h"
#include "stablehlo/dialect/StablehloOps.h"

int main(int argc, char** argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry dialects;
  dialects.insert<mlir::func::FuncDialect, mlir::quant::QuantizationDialect,
                  mlir::sdy::SdyDialect, mlir::stablehlo::StablehloDialect>();
  mlir::func::registerAllExtensions(dialects);

  // Register all SDY passes and pipelines.
  mlir::sdy::registerAllSdyPassesAndPipelines();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SDY pass driver\n", dialects));
}
