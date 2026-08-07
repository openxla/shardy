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

#include "shardy/dialect/sdy/transforms/export/utils.h"

#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/testing_utils.h"
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {
namespace {

class ExportUtilsTest : public ShardyTestBase {};

TEST_F(ExportUtilsTest, GetShardIndexInterleavedSubAxes) {
  // mesh = <["a"=4, "b"=2]>
  // axes = ["a":2(2), "b", "a":1(2)]
  MeshAttr mesh =
      MeshAttr::get(&context, {MeshAxisAttr::get(&context, "a", 4),
                               MeshAxisAttr::get(&context, "b", 2)});

  AxisRefAttr aSubLow =
      AxisRefAttr::get(&context, "a", /*subAxisPreSize=*/2, /*subAxisSize=*/2);
  AxisRefAttr bFull = AxisRefAttr::get(&context, "b");
  AxisRefAttr aSubHigh =
      AxisRefAttr::get(&context, "a", /*subAxisPreSize=*/1, /*subAxisSize=*/2);

  SmallVector<AxisRefAttr> axes = {aSubLow, bFull, aSubHigh};

  // Device IDs 0..7
  // id = a * 2 + b
  // a_high = a / 2
  // a_low = a % 2.
  // shardIndex = a_low * 4 + b * 2 + a_high
  EXPECT_EQ(getShardIndex(0, mesh, axes), 0);
  EXPECT_EQ(getShardIndex(1, mesh, axes), 2);
  EXPECT_EQ(getShardIndex(2, mesh, axes), 4);
  EXPECT_EQ(getShardIndex(3, mesh, axes), 6);
  EXPECT_EQ(getShardIndex(4, mesh, axes), 1);
  EXPECT_EQ(getShardIndex(5, mesh, axes), 3);
  EXPECT_EQ(getShardIndex(6, mesh, axes), 5);
  EXPECT_EQ(getShardIndex(7, mesh, axes), 7);
}

}  // namespace
}  // namespace sdy
}  // namespace mlir
