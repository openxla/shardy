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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_COMMON_TESTING_UTILS_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_COMMON_TESTING_UTILS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"

namespace mlir::mpmd {

// Finds an op of type `OpTy` that contains the attribute `attr_name`, as a
// means of identifying the specific place we'd like to test.
template <typename OpTy>
OpTy GetOpWithAttribute(func::FuncOp fn, StringRef attr_name) {
  OpTy found_op;
  fn.walk([&](OpTy op) {
    if (op->hasAttr(attr_name)) {
      found_op = op;
    }
  });
  SDY_CHECK(found_op);
  return found_op;
}

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_COMMON_TESTING_UTILS_H_
