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

#ifndef SHARDY_DIALECT_MPMD_IR_REGISTER_H_
#define SHARDY_DIALECT_MPMD_IR_REGISTER_H_

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace mpmd {

// Add all required dialects to the provided registry.
void registerAllDialects(DialectRegistry& registry);

// Loads all required dialects to the provided context.
void loadAllRequiredDialects(MLIRContext* context);

}  // namespace mpmd
}  // namespace mlir

#endif  // SHARDY_DIALECT_MPMD_IR_REGISTER_H_
