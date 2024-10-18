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

using ::testing::ElementsAre;
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

TEST_F(ExportTest, getGreatestCommonPrefix_Canonical) {
  EXPECT_THAT(
      getGreatestCommonPrefix(
          {createAxis("a"), createAxis("b"), createAxis("d"), createAxis("e")},
          {createAxis("a"), createAxis("b"), createAxis("c")}),
      ElementsAre(createAxis("a"), createAxis("b")));
}

TEST_F(ExportTest, getGreatestCommonPrefix_OrderMatters) {
  EXPECT_THAT(getGreatestCommonPrefix({createAxis("a"), createAxis("d")},
                                      {createAxis("a"), createAxis("d")}),
              ElementsAre(createAxis("a"), createAxis("d")));

  EXPECT_THAT(getGreatestCommonPrefix({createAxis("d"), createAxis("a")},
                                      {createAxis("d"), createAxis("a")}),
              ElementsAre(createAxis("d"), createAxis("a")));
}

TEST_F(ExportTest, getGreatestCommonPrefix_Empty) {
  EXPECT_THAT(getGreatestCommonPrefix({createAxis("a"), createAxis("c")}, {}),
              IsEmpty());

  EXPECT_THAT(getGreatestCommonPrefix({}, {createAxis("a"), createAxis("c")}),
              IsEmpty());

  EXPECT_THAT(getGreatestCommonPrefix({}, {}), IsEmpty());
}

TEST_F(ExportTest, getGreatestCommonPrefix_SubAxes) {
  EXPECT_THAT(getGreatestCommonPrefix(
                  {createAxis("a"), createSubAxis("b", 2, 4), createAxis("d"),
                   createAxis("e")},
                  {createAxis("a"), createSubAxis("b", 2, 8), createAxis("d")}),
              ElementsAre(createAxis("a"), createSubAxis("b", 2, 4)));

  EXPECT_THAT(getGreatestCommonPrefix(
                  {createSubAxis("a", 2, 6), createSubAxis("b", 2, 4),
                   createAxis("d"), createAxis("e")},
                  {createSubAxis("a", 2, 6), createSubAxis("b", 2, 8),
                   createAxis("d")}),
              ElementsAre(createSubAxis("a", 2, 6), createSubAxis("b", 2, 4)));
}

}  // namespace
}  // namespace sdy
}  // namespace mlir
