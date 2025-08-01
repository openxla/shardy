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

#include "shardy/dialect/mpmd/ir/register.h"

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace mpmd {

void registerAllDialects(DialectRegistry& registry) {
  registry.insert<MpmdDialect, sdy::SdyDialect, func::FuncDialect,
                  tensor::TensorDialect, stablehlo::StablehloDialect>();
}

void loadAllRequiredDialects(MLIRContext* context) {
  DialectRegistry registry;
  func::registerAllExtensions(registry);
  sdy::registerAllDialects(registry);
  registerAllDialects(registry);
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
}

}  // namespace mpmd
}  // namespace mlir
