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

#include <cstdint>

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {
namespace {

using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

class ExportTest : public ::testing::Test {
 protected:
  void SetUp() override { context.loadDialect<SdyDialect>(); }

  AxisRefAttr createAxis(StringRef name) {
    return AxisRefAttr::get(&context, name);
  }

  AxisRefAttr createSubAxis(StringRef name, int64_t preSize, int64_t size) {
    return AxisRefAttr::get(&context, name, preSize, size);
  }

  MLIRContext context;
};

TEST_F(ExportTest, getGreatestCommonPrefix) {
  auto checkGreatestCommonPrefix = [](ArrayRef<AxisRefAttr> a,
                                      ArrayRef<AxisRefAttr> b,
                                      ArrayRef<AxisRefAttr> c) {
    EXPECT_THAT(getGreatestCommonPrefix(a, b), ElementsAreArray(c));
    EXPECT_THAT(getGreatestCommonPrefix(b, a), ElementsAreArray(c));
  };
  checkGreatestCommonPrefix(
      {createAxis("a"), createAxis("b"), createAxis("d"), createAxis("e")},
      {createAxis("a"), createAxis("b"), createAxis("c")},
      {createAxis("a"), createAxis("b")});
  checkGreatestCommonPrefix(
      {createAxis("a")}, {createAxis("a"), createAxis("b")}, {createAxis("a")});
  checkGreatestCommonPrefix(
      {createAxis("a"), createSubAxis("b", 2, 4), createAxis("d"),
       createAxis("e")},
      {createAxis("a"), createSubAxis("b", 2, 8), createAxis("d")},
      {createAxis("a"), createSubAxis("b", 2, 4)});
  checkGreatestCommonPrefix(
      {createSubAxis("a", 2, 6), createSubAxis("b", 2, 4), createAxis("d"),
       createAxis("e")},
      {createSubAxis("a", 2, 6), createSubAxis("b", 2, 8), createAxis("d")},
      {createSubAxis("a", 2, 6), createSubAxis("b", 2, 4)});
}

TEST_F(ExportTest, getGreatestCommonPrefix_AxesOrderMatters) {
  auto samePrefix = [](ArrayRef<AxisRefAttr> a) {
    EXPECT_THAT(getGreatestCommonPrefix(a, a), ElementsAreArray(a));
  };
  samePrefix({createAxis("a"), createAxis("d")});
  samePrefix({createAxis("d"), createAxis("a")});
}

TEST_F(ExportTest, getGreatestCommonPrefix_Empty) {
  auto emptyPrefix = [](ArrayRef<AxisRefAttr> a) {
    EXPECT_THAT(getGreatestCommonPrefix(a, {}), IsEmpty());
    EXPECT_THAT(getGreatestCommonPrefix({}, a), IsEmpty());
  };
  emptyPrefix({createAxis("a"), createAxis("c")});
  emptyPrefix({});
}

}  // namespace
}  // namespace sdy
}  // namespace mlir
