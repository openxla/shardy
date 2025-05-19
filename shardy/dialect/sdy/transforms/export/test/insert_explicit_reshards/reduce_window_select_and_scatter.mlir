// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @reduce_window
func.func @reduce_window(%arg0: tensor<48x48x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<48x48x3xi32>, %arg2: tensor<f32>, %arg3: tensor<i32>)
    -> (tensor<16x48x1xf32>, tensor<16x48x1xi32>) {
  // CHECK-NOT: sdy.reshard
  %0:2 = "stablehlo.reduce_window"(%arg0, %arg1, %arg2, %arg3) ({
  ^bb0(%arg4: tensor<f32>, %arg5 : tensor<i32>, %arg6: tensor<f32>, %arg7 : tensor<i32>):
    %1 = stablehlo.maximum %arg4, %arg6 : tensor<f32>
    %2 = stablehlo.maximum %arg5, %arg7 : tensor<i32>
    stablehlo.return %1, %2 : tensor<f32>, tensor<i32>
  }) {window_dimensions = array<i64: 3, 1, 3>,
      window_strides = array<i64: 3, 1, 3>,
      padding = dense<[[0, 0], [2, -2], [0, 0]]> : tensor<3x2xi64>}
      : (tensor<48x48x3xf32>, tensor<48x48x3xi32>, tensor<f32>, tensor<i32>) -> (tensor<16x48x1xf32>, tensor<16x48x1xi32>)
  func.return %0#0, %0#1 : tensor<16x48x1xf32>, tensor<16x48x1xi32>
}

// CHECK-LABEL: func @select_and_scatter
func.func @select_and_scatter(%arg0: tensor<10x24x24x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}, %arg1: tensor<10x12x12x64xf32>, %arg2: tensor<f32>)
   -> tensor<10x24x24x64xf32> {
  %1 = "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
  // CHECK-NOT: sdy.reshard
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = stablehlo.compare GT, %arg3, %arg4 :(tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
  },  {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
    stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 1, 2, 2, 1>,
    window_strides = array<i64: 1, 2, 2, 1>
  } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  return %1 : tensor<10x24x24x64xf32>
}
