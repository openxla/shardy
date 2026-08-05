/* Copyright 2026 The Shardy Authors.

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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_PASSES_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_PASSES_H_

#include <memory>            // IWYU pragma: keep
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "shardy/dialect/sdy/transforms/common/passes.h.inc"

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_PASSES_H_
