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

#include <cstdint>
#include <utility>

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {

using ::testing::DescribeMatcher;
using ::testing::IsEmpty;
using ::testing::PrintToString;

MATCHER_P(AxisRefIs, axisName,
          (negation ? "axis isn't " : "axis is ") + PrintToString(axisName)) {
  *result_listener << "where axis is " << arg.toString();
  return arg.getName() == axisName;
}

MATCHER_P3(SubAxisRefIs, axisName, preSize, size,
           (negation ? "sub-axis isn't " : "sub-axis is ") +
               PrintToString(axisName) + ":(" + PrintToString(preSize) + ")" +
               PrintToString(size)) {
  *result_listener << "where sub-axis is " << arg.toString();
  return arg.getName() == axisName && arg.getSubAxisInfo() &&
         arg.getSubAxisInfo().getPreSize() == preSize &&
         arg.getSubAxisInfo().getSize() == size;
}

class ShardyTestBase : public ::testing::Test {
 protected:
  void SetUp() override { loadAllRequiredDialects(&context); }

  AxisRefAttr createAxis(StringRef name) {
    return AxisRefAttr::get(&context, name);
  }

  AxisRefAttr createSubAxis(StringRef name, int64_t preSize, int64_t size) {
    return AxisRefAttr::get(&context, name, preSize, size);
  }

  MeshAttr createMesh(ArrayRef<std::pair<StringRef, int64_t>> axisNameAndSizes,
                      ArrayRef<int64_t> deviceIds = {}) {
    SmallVector<MeshAxisAttr> meshAxisAttrs;
    meshAxisAttrs.reserve(axisNameAndSizes.size());
    for (auto [axisName, axisSize] : axisNameAndSizes) {
      meshAxisAttrs.push_back(MeshAxisAttr::get(&context, axisName, axisSize));
    }
    return MeshAttr::get(&context, meshAxisAttrs, deviceIds);
  }

  MLIRContext context;
};

}  // namespace sdy
}  // namespace mlir
