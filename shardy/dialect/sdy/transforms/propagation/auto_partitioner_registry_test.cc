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

#include "shardy/dialect/sdy/transforms/propagation/auto_partitioner_registry.h"

#include <gtest/gtest.h>

namespace mlir {
namespace sdy {

TEST(AutoPartitionerRegistryTest, RegisterAutoPartitioner) {
  AutoPartitionerRegistry registry;
  ASSERT_FALSE(registry.isRegistered());
  // Set NoOp callbacks.
  registry.setCallback(AutoPartitionerCallback());
  EXPECT_TRUE(registry.isRegistered());
  registry.clear();
  EXPECT_FALSE(registry.isRegistered());
}

}  // namespace sdy
}  // namespace mlir
