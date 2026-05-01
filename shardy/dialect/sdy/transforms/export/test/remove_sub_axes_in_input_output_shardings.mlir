// RUN: sdy_opt %s -sdy-remove-sub-axes-in-input-output-shardings -split-input-file | FileCheck %s

// This test check that:
// 1. We remove sub-axes and the trailing axes in input and output shardings.
// 2. We do not modify the shardings for intermediate tensors.

sdy.mesh @mesh = <["x"=8, "y"=8, "z"=8]>

// CHECK-LABEL: func @main
// CHECK-SAME: %arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"z"}]>},
// CHECK-SAME: %arg1: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", ?}, {?}]>})
// CHECK-SAME: -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"x", "z", ?}]>},
// CHECK-SAME: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>})
func.func @main(
    %arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2, "y", ?}, {"z"}]>},
    %arg1: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x":(1)2, ?}, {?}]>})
    -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y":(1)2, ?}, {"x", "z", ?}]>},
        tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>}) {
  // CHECK: %0 = stablehlo.add      %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2, "y"}, {"z"}]>]>} : tensor<64x64xf32>
  // CHECK: %1 = stablehlo.multiply %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x":(1)2}, {}]>]>} : tensor<64x64xf32>
  %0 = stablehlo.add      %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2, "y"}, {"z"}]>]>} : tensor<64x64xf32>
  %1 = stablehlo.multiply %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x":(1)2}, {}]>]>} : tensor<64x64xf32>
  return %0, %1 : tensor<64x64xf32>, tensor<64x64xf32>
}

// -----

// This test check that it does not remove sub-axes in input and output
// shardings of non-main func but removes them for the main func.

sdy.mesh @mesh = <["x"=8, "y"=8, "z"=8]>

// CHECK-LABEL: func private @foo
// CHECK-SAME: %arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2, "y", ?}, {"z"}]>},
// CHECK-SAME: %arg1: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x":(1)2, ?}, {?}]>})
// CHECK-SAME: -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y":(1)2, ?}, {"x", "z", ?}]>},
// CHECK-SAME: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>}) {
func.func private @foo(
    %arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2, "y", ?}, {"z"}]>},
    %arg1: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x":(1)2, ?}, {?}]>})
    -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y":(1)2, ?}, {"x", "z", ?}]>},
        tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>}) {
  return %arg0, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>
}

// CHECK-LABEL: func @main
// CHECK-SAME: %arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"z"}]>},
// CHECK-SAME: %arg1: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", ?}, {?}]>})
// CHECK-SAME: -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"x", "z", ?}]>},
// CHECK-SAME: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>})
func.func @main(
    %arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2, "y", ?}, {"z"}]>},
    %arg1: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x":(1)2, ?}, {?}]>})
    -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y":(1)2, ?}, {"x", "z", ?}]>},
        tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>}) {
  // CHECK: %0:2 = call @foo(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y":(1)2, ?}, {"x", "z", ?}]>, <@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>]>}
  %0:2 = call @foo(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y":(1)2, ?}, {"x", "z", ?}]>, <@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>]>} : (tensor<64x64xf32>, tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>)
  return %0#0, %0#1 : tensor<64x64xf32>, tensor<64x64xf32>
}

// -----

// test for simple non flat functions.

sdy.mesh @mesh = <["x"=8, "y"=8, "z"=8]>

// CHECK-LABEL: func private @foo
// CHECK-SAME: %arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2, "y", ?}, {"z"}]>},
// CHECK-SAME: %arg1: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x":(1)2, ?}, {?}]>})
// CHECK-SAME: -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y":(1)2, ?}, {"x", "z", ?}]>},
// CHECK-SAME: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>}) {
func.func private @foo(
    %arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2, "y", ?}, {"z"}]>},
    %arg1: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x":(1)2, ?}, {?}]>})
    -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y":(1)2, ?}, {"x", "z", ?}]>},
        tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>}) {
  %0:2 = call @bar(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y":(1)2, ?}, {"x", "z", ?}]>, <@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>]>} : (tensor<64x64xf32>, tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>)
  return %0#0, %0#1 : tensor<64x64xf32>, tensor<64x64xf32>
}

// CHECK-LABEL: func private @bar
// CHECK-SAME: %arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2, "y", ?}, {"z"}]>},
// CHECK-SAME: %arg1: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x":(1)2, ?}, {?}]>})
// CHECK-SAME: -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y":(1)2, ?}, {"x", "z", ?}]>},
// CHECK-SAME: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>}) {
func.func private @bar(
    %arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2, "y", ?}, {"z"}]>},
    %arg1: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x":(1)2, ?}, {?}]>})
    -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y":(1)2, ?}, {"x", "z", ?}]>},
        tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>}) {
  return %arg0, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>
}

// CHECK-LABEL: func @main
// CHECK-SAME: %arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"z"}]>},
// CHECK-SAME: %arg1: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", ?}, {?}]>})
// CHECK-SAME: -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"x", "z", ?}]>},
// CHECK-SAME: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>})
func.func @main(
    %arg0: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2, "y", ?}, {"z"}]>},
    %arg1: tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x":(1)2, ?}, {?}]>})
    -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y":(1)2, ?}, {"x", "z", ?}]>},
        tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>}) {
  // CHECK: %0:2 = call @foo(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y":(1)2, ?}, {"x", "z", ?}]>, <@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>]>}
  %0:2 = call @foo(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y":(1)2, ?}, {"x", "z", ?}]>, <@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>]>} : (tensor<64x64xf32>, tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>)
  %1:2 = call @bar(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y":(1)2, ?}, {"x", "z", ?}]>, <@mesh, [{"y", "z":(4)2}, {"z":(2)2, "x"}]>]>} : (tensor<64x64xf32>, tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>)
  return %0#0, %1#1 : tensor<64x64xf32>, tensor<64x64xf32>
}
