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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_SHARDING_PROPAGATION_UTILS_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_SHARDING_PROPAGATION_UTILS_H_

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "mlir/IR/Value.h"

namespace mlir::mpmd {

inline void UpdateValueUserInShardings(
    Value value, mlir::sdy::TensorShardingAttr sharding) {
  for (mlir::Operation* user : value.getUsers()) {
    if (auto fragment_user = mlir::dyn_cast<mlir::mpmd::FragmentOp>(user)) {
      for (auto [operand_number, input] :
           llvm::enumerate(fragment_user.getInputs())) {
        if (input == value) {
          fragment_user.setInputSharding(operand_number, sharding);
        }
      }
    }
  }
}


}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_SHARDING_PROPAGATION_UTILS_H_

