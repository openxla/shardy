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

#include "shardy/dialect/sdy/transforms/export/utils.h"

#include "llvm/ADT/BitVector.h"    // IWYU pragma: keep
#include "llvm/ADT/SmallVector.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

SmallVector<AxisRefAttr> getGreatestCommonPrefix(ArrayRef<AxisRefAttr> first,
                                                 ArrayRef<AxisRefAttr> second) {
  SmallVector<AxisRefAttr> result;
  for (int i = 0; i < first.size() && i < second.size(); i++) {
    if (first[i] != second[i]) {
      auto prefix = first[i].getGreatestCommonPrefix(second[i]);
      if (prefix) {
        result.push_back(*prefix);
      }
      break;
    }
    result.push_back(first[i]);
  }
  return result;
}

}  // namespace sdy
}  // namespace mlir
