/* Copyright 2025 The Shardy Authors.

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

// MLIR `translate` tool for allowing SDY dialect bytecode emission.
//
// Usage:
//   sdy_translate <file.mlir> -serialize
//   sdy_translate <file.mlir.bc> -deserialize

#include <memory>
#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/dialect/sdy/ir/compatibility.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"

namespace mlir {

namespace {
llvm::cl::opt<bool> stripDebuginfoOption(
    "strip-debuginfo", llvm::cl::desc("Strip debug info from all operations"),
    llvm::cl::init(false));
llvm::cl::opt<std::string> targetVersionOption(
    "target-version", llvm::cl::desc("Target version for serialization"),
    llvm::cl::init(""));

void registerDialectsForSdy(DialectRegistry &registry) {
  mlir::sdy::registerAllDialects(registry);
  registry.insert<mlir::quant::QuantDialect>();
}

TranslateFromMLIRRegistration serializeRegistration(
    "serialize", "Serialize SDY program into a portable artifact",
    [](mlir::ModuleOp module, llvm::raw_ostream &os) -> llvm::LogicalResult {
      if (stripDebuginfoOption) {
        PassManager pm(module->getContext());
        pm.addPass(createStripDebugInfoPass());
        if (failed(pm.run(module)))
          return module.emitError("failed to strip debuginfo");
      }
      const auto *producer = "SDY";
      BytecodeWriterConfig writerConfig(producer);
      auto versionToUse = sdy::SdyDialectVersion::getCurrentVersion();
      if (std::string targetVersion = targetVersionOption.getValue();
          !targetVersion.empty()) {
        FailureOr<sdy::SdyDialectVersion> dialectVersion =
            sdy::SdyDialectVersion::fromString(targetVersion);
        if (failed(dialectVersion)) {
          return module.emitError("failed to parse target version");
        }
        versionToUse = dialectVersion.value();
      }
      // set version in config and manually downgrade the ModuleOp.
      writerConfig.setDialectVersion<sdy::SdyDialect>(
          std::make_unique<sdy::SdyDialectVersion>(versionToUse));
      if (versionToUse < sdy::SdyDialectVersion::getCurrentVersion()) {
        if (failed(sdy::downgradeModule(module, versionToUse))) {
          return module.emitError("failed to downgrade module");
        }
      }
      return writeBytecodeToFile(module, os, writerConfig);
    },
    [](DialectRegistry &registry) { registerDialectsForSdy(registry); });

TranslateToMLIRRegistration deserializeRegistration(
    "deserialize", "Deserialize a portable artifact into a SDY program",
    [](llvm::StringRef input, mlir::MLIRContext *context) {
      context->loadDialect<sdy::SdyDialect>();
      // will rely onto upgradeFromVersion to support older versions.
      auto module = parseSourceString<ModuleOp>(input, context);
      return module;
    },
    [](DialectRegistry &registry) { registerDialectsForSdy(registry); });
}  // namespace

}  // namespace mlir

int main(int argc, char **argv) {
  return mlir::asMainReturnCode(
      mlir::mlirTranslateMain(argc, argv, "SDY transformation driver\n"));
}
