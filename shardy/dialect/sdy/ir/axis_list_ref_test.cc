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

#include "shardy/dialect/sdy/ir/axis_list_ref.h"

#include <cstdint>

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {
namespace {

// TODO(enver): Expose a base test class with create axis methods.
class AxisListRefTest : public ::testing::Test {
 protected:
  void SetUp() override { loadAllRequiredDialects(&context); }

  AxisRefAttr createAxis(StringRef name) {
    return AxisRefAttr::get(&context, name);
  }

  AxisRefAttr createSubAxis(StringRef name, int64_t preSize, int64_t size) {
    return AxisRefAttr::get(&context, name, preSize, size);
  }

  MLIRContext context;
};

TEST_F(AxisListRefTest, Empty) { EXPECT_TRUE(AxisListRef().empty()); }

TEST_F(AxisListRefTest, NonEmpty) {
  SmallVector<AxisRefAttr> backingAxes({createAxis("a")});
  EXPECT_FALSE(AxisListRef(backingAxes).empty());
}

// TODO(enver): Add unit tests for all methods.

}  // namespace
}  // namespace sdy
}  // namespace mlir
