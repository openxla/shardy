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

#ifndef SHARDY_DIALECT_SDY_IR_SERIALIZATION_H_
#define SHARDY_DIALECT_SDY_IR_SERIALIZATION_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir::sdy {

// Helper function to encapsulate the logic of downgrading a ModuleOp to a
// specific version. All passes that need to be run to downgrade the module
// should be added here.
// TODO(kostiantynl): b/406766999 add support in upstream interface.
LogicalResult downgradeModule(ModuleOp module, SdyDialectVersion version);

}  // namespace mlir::sdy

#endif  // SHARDY_DIALECT_SDY_IR_SERIALIZATION_H_
