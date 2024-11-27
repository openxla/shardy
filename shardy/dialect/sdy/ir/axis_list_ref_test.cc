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

  AxisListRef createAxisListRef(SmallVector<AxisRefAttr> axisRefs) {
    if (axisRefs.empty()) {
      return AxisListRef();
    }
    backingData.push_back(axisRefs);
    return AxisListRef(backingData.back());
  }

  SmallVector<SmallVector<AxisRefAttr>> backingData;
  MLIRContext context;
};

TEST_F(AxisListRefTest, Empty) {
  AxisListRef axes = createAxisListRef({});
  EXPECT_TRUE(axes.empty());
}

TEST_F(AxisListRefTest, Empty_NotEmpty) {
  AxisListRef axes = createAxisListRef({createAxis("a")});
  EXPECT_FALSE(axes.empty());
}

TEST_F(AxisListRefTest, TruncateWithoutOverlap) {
  AxisListRef axes =
      createAxisListRef({createAxis("a"), createAxis("b"), createAxis("c"),
                         createAxis("d"), createAxis("e")});
  AxisListRef against =
      createAxisListRef({createAxis("c"), createAxis("e"), createAxis("f")});
  EXPECT_TRUE(axes.truncateWithoutOverlap(against));
  EXPECT_EQ(axes, createAxisListRef({createAxis("a"), createAxis("b")}));
}

TEST_F(AxisListRefTest, TruncateWithoutOverlap_AgainstOrderDoesNotMatter) {
  AxisListRef axes =
      createAxisListRef({createAxis("a"), createAxis("b"), createAxis("c"),
                         createAxis("d"), createAxis("e")});
  AxisListRef against =
      createAxisListRef({createAxis("f"), createAxis("e"), createAxis("c")});
  EXPECT_TRUE(axes.truncateWithoutOverlap(against));
  EXPECT_EQ(axes, createAxisListRef({createAxis("a"), createAxis("b")}));
}

TEST_F(AxisListRefTest, TruncateWithoutOverlap_NoOverlap) {
  AxisListRef axes = createAxisListRef({createAxis("a"), createAxis("b")});
  AxisListRef against = createAxisListRef({createAxis("c")});
  EXPECT_FALSE(axes.truncateWithoutOverlap(against));
  EXPECT_EQ(axes, createAxisListRef({createAxis("a"), createAxis("b")}));
}

TEST_F(AxisListRefTest, TruncateWithoutOverlap_EmptyTruncateOnEmpty) {
  AxisListRef axes = createAxisListRef({});
  AxisListRef against = createAxisListRef({});
  EXPECT_FALSE(axes.truncateWithoutOverlap(against));
  EXPECT_TRUE(axes.empty());
}

TEST_F(AxisListRefTest, TruncateWithoutOverlap_EmptyTruncateOnNonEmpty) {
  AxisListRef axes = createAxisListRef({});
  AxisListRef against = createAxisListRef({createAxis("a")});
  EXPECT_FALSE(axes.truncateWithoutOverlap(against));
  EXPECT_TRUE(axes.empty());
}

TEST_F(AxisListRefTest, TruncateWithoutOverlap_NonEmptyTruncateOnEmpty) {
  AxisListRef axes = createAxisListRef({createAxis("a")});
  AxisListRef against = createAxisListRef({});
  EXPECT_FALSE(axes.truncateWithoutOverlap(against));
  EXPECT_EQ(axes, createAxisListRef({createAxis("a")}));
}

TEST_F(AxisListRefTest, TruncateWithoutOverlap_AgainstSame) {
  AxisListRef axes = createAxisListRef({createAxis("a"), createAxis("b")});
  AxisListRef against = createAxisListRef({createAxis("a"), createAxis("b")});
  EXPECT_TRUE(axes.truncateWithoutOverlap(against));
  EXPECT_TRUE(axes.empty());
}

TEST_F(AxisListRefTest, TruncateWithoutOverlap_AxesIsSingleton) {
  AxisListRef axes = createAxisListRef({createAxis("a")});
  AxisListRef against = createAxisListRef({createAxis("a"), createAxis("b")});
  EXPECT_TRUE(axes.truncateWithoutOverlap(against));
  EXPECT_TRUE(axes.empty());
}

TEST_F(AxisListRefTest, TruncateWithoutOverlap_AgainstPrefix) {
  AxisListRef axes = createAxisListRef({createAxis("a"), createAxis("b")});
  AxisListRef against = createAxisListRef({createAxis("a")});
  EXPECT_TRUE(axes.truncateWithoutOverlap(against));
  EXPECT_TRUE(axes.empty());
}

TEST_F(AxisListRefTest, TruncateWithoutOverlap_AgainstSuffix) {
  AxisListRef axes = createAxisListRef({createAxis("a"), createAxis("b")});
  AxisListRef against = createAxisListRef({createAxis("b")});
  EXPECT_TRUE(axes.truncateWithoutOverlap(against));
  EXPECT_EQ(axes, createAxisListRef({createAxis("a")}));
}

TEST_F(AxisListRefTest, TruncateWithoutOverlap_SubAxis) {
  AxisListRef axes = createAxisListRef({createAxis("a"), createAxis("b")});
  AxisListRef against = createAxisListRef({createSubAxis("a", 4, 2)});
  EXPECT_TRUE(axes.truncateWithoutOverlap(against));
  EXPECT_EQ(axes, createAxisListRef({createSubAxis("a", 1, 4)}));
}

TEST_F(AxisListRefTest, TruncateWithoutOverlap_SubAxisAndAxesIsSingleton) {
  AxisListRef axes = createAxisListRef({createAxis("a")});
  AxisListRef against = createAxisListRef({createSubAxis("a", 4, 2)});
  EXPECT_TRUE(axes.truncateWithoutOverlap(against));
  EXPECT_EQ(axes, createAxisListRef({createSubAxis("a", 1, 4)}));
}

TEST_F(AxisListRefTest, TruncateWithoutOverlap_SubAxisOnLastElement) {
  AxisListRef axes = createAxisListRef({createAxis("a"), createAxis("b")});
  AxisListRef against = createAxisListRef({createSubAxis("b", 4, 2)});
  EXPECT_TRUE(axes.truncateWithoutOverlap(against));
  EXPECT_EQ(axes,
            createAxisListRef({createAxis("a"), createSubAxis("b", 1, 4)}));
}

TEST_F(AxisListRefTest, LessThan_SameSizeSmaller) {
  AxisListRef axes = createAxisListRef({createAxis("a"), createAxis("b")});
  AxisListRef against = createAxisListRef({createAxis("a"), createAxis("c")});
  EXPECT_TRUE(axes < against);
}

TEST_F(AxisListRefTest, LessThan_StrictPrefix) {
  AxisListRef axes = createAxisListRef({createAxis("a"), createAxis("b")});
  AxisListRef against = createAxisListRef({createAxis("b"), createAxis("c")});
  EXPECT_TRUE(axes < against);
}

TEST_F(AxisListRefTest, LessThan_SubAxisStrictPrefix) {
  AxisListRef axes =
      createAxisListRef({createAxis("a"), createSubAxis("b", 1, 2)});
  AxisListRef against = createAxisListRef({createAxis("a"), createAxis("b")});
  EXPECT_TRUE(axes < against);
}

TEST_F(AxisListRefTest, LessThan_SmallerSize) {
  AxisListRef axes = createAxisListRef({createAxis("a"), createAxis("c")});
  AxisListRef against =
      createAxisListRef({createAxis("a"), createAxis("b"), createAxis("c")});
  EXPECT_TRUE(axes < against);
}

TEST_F(AxisListRefTest, LessThan_Equal) {
  AxisListRef axes = createAxisListRef({createAxis("a"), createAxis("c")});
  AxisListRef against = createAxisListRef({createAxis("a"), createAxis("c")});
  EXPECT_FALSE(axes < against);
}

TEST_F(AxisListRefTest, LessThan_EmptyAgainstEmpty) {
  AxisListRef axes = createAxisListRef({});
  AxisListRef against = createAxisListRef({});
  EXPECT_FALSE(axes < against);
}

TEST_F(AxisListRefTest, LessThan_EmptyAgainstNonEmpty) {
  AxisListRef axes = createAxisListRef({});
  AxisListRef against = createAxisListRef({createAxis("a")});
  EXPECT_TRUE(axes < against);
}

TEST_F(AxisListRefTest, LessThan_NonEmptyAgainstEmpty) {
  AxisListRef axes = createAxisListRef({createAxis("a")});
  AxisListRef against = createAxisListRef({});
  EXPECT_FALSE(axes < against);
}

// TODO(enver): Add unit tests for all methods.

}  // namespace
}  // namespace sdy
}  // namespace mlir
