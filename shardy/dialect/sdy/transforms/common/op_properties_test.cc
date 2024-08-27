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

#include "shardy/dialect/sdy/transforms/common/op_properties.h"

#include <cassert>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {

namespace {

template <class OpTy>
OpTy getFirstOp(ModuleOp module) {
  auto mainFn = cast<func::FuncOp>(module.lookupSymbol("main"));
  auto ops = mainFn.getBody().front().getOps<OpTy>();
  assert(!ops.empty());
  return *ops.begin();
}

class IsElementwiseTest : public ::testing::Test {
 protected:
  void SetUp() override { loadAllRequiredDialects(&context); }

  MLIRContext context;
};

TEST_F(IsElementwiseTest, NonElementwiseOp) {
  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<256xf32>) -> tensor<32x256xf32> {
      %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<256xf32>) -> tensor<32x256xf32>
      return %0 : tensor<32x256xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);

  EXPECT_FALSE(
      isElementwise(getFirstOp<stablehlo::BroadcastInDimOp>(module.get())));
}

TEST_F(IsElementwiseTest, OpWithElementwiseTrait) {
  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
      %0 = stablehlo.add %arg0, %arg0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      return %0 : tensor<2x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);

  EXPECT_TRUE(isElementwise(getFirstOp<stablehlo::AddOp>(module.get())));
}

TEST_F(IsElementwiseTest, OpWithBroadcastingElementwiseTrait) {
  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<i64>, %arg1: tensor<5xi64>, %arg2: tensor<i64>)
        -> tensor<5xi64> {
      %0 = stablehlo.clamp %arg0, %arg1, %arg2 : (tensor<i64>, tensor<5xi64>, tensor<i64>) -> tensor<5xi64>
      return %0 : tensor<5xi64>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);

  EXPECT_TRUE(isElementwise(getFirstOp<stablehlo::ClampOp>(module.get())));
}

TEST_F(IsElementwiseTest, BitcastConvertOpSameBitWidth) {
  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<4xui64>) -> tensor<4xi64> {
      %0 = stablehlo.bitcast_convert %arg0 : (tensor<4xui64>) -> tensor<4xi64>
      return %0 : tensor<4xi64>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);

  EXPECT_TRUE(
      isElementwise(getFirstOp<stablehlo::BitcastConvertOp>(module.get())));
}

TEST_F(IsElementwiseTest, BitcastConvertOpDifferentBitWidth) {
  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<2xui64>) -> tensor<2x2xui32> {
      %0 = stablehlo.bitcast_convert %arg0 : (tensor<2xui64>) -> tensor<2x2xui32>
      return %0 : tensor<2x2xui32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);

  EXPECT_FALSE(
      isElementwise(getFirstOp<stablehlo::BitcastConvertOp>(module.get())));
}

}  // namespace

}  // namespace sdy
}  // namespace mlir
