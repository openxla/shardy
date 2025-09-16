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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_IMPORT_APPLY_SHARDING_CONSTRAINTS_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_IMPORT_APPLY_SHARDING_CONSTRAINTS_H_

#include <memory>

#include "mlir/Pass/Pass.h"
#include "shardy/dialect/sdy/transforms/common/propagation_options.h"

namespace mlir {
namespace sdy {

// TODO(b/445415899): Move this definition to the .td file.
// Creates a pass that applies the sharding of sdy.sharding_constraint ops to
// their operands, if the operand doesn't have a sharding yet.
std::unique_ptr<Pass> createApplyShardingConstraintsPass(
    const PropagationOptions& options);

// The no-argument version is declared in passes.h.inc

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_IMPORT_APPLY_SHARDING_CONSTRAINTS_H_
