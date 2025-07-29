// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xyzt = <["x"=4, "y"=4, "z"=4, "t"=8]>

// CHECK-LABEL: func @reduce_window
func.func @reduce_window(%arg0: tensor<48x48x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>},
                         %arg1: tensor<48x48x3xi32>, %arg2: tensor<f32>, %arg3: tensor<i32>)
    -> (tensor<16x48x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, tensor<16x48x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) {
  // sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k], [], [])->([i, j, k], [i, j, k]) {i=16, j=48, k=1} permutation={i, j, k}>
  // CHECK-NEXT: %0 = sdy.reshard %arg1 <@mesh, [{"x"}, {}, {}]> : tensor<48x48x3xi32>
  // CHECK-NEXT: %1:2 = "stablehlo.reduce_window"(%arg0, %0, %arg2, %arg3)
  // CHECK: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>, <@mesh, [{"x"}, {}, {}]>]>
  %0:2 = "stablehlo.reduce_window"(%arg0, %arg1, %arg2, %arg3) ({
  ^bb0(%arg4: tensor<f32>, %arg5 : tensor<i32>, %arg6: tensor<f32>, %arg7 : tensor<i32>):
    %1 = stablehlo.maximum %arg4, %arg6 : tensor<f32>
    %2 = stablehlo.maximum %arg5, %arg7 : tensor<i32>
    stablehlo.return %1, %2 : tensor<f32>, tensor<i32>
  }) {window_dimensions = array<i64: 3, 1, 3>,
      window_strides = array<i64: 3, 1, 3>,
      padding = dense<[[0, 0], [2, -2], [0, 0]]> : tensor<3x2xi64>,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>, <@mesh, [{"x"}, {}, {}]>]>
      } : (tensor<48x48x3xf32>, tensor<48x48x3xi32>, tensor<f32>, tensor<i32>) -> (tensor<16x48x1xf32>, tensor<16x48x1xi32>)
  // CHECK-NEXT: return %1#0, %1#1 : tensor<16x48x1xf32>, tensor<16x48x1xi32>
  func.return %0#0, %0#1 : tensor<16x48x1xf32>, tensor<16x48x1xi32>
}

// CHECK-LABEL: func @select_and_scatter
func.func @select_and_scatter(%arg0: tensor<10x24x24x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>},
                              %arg1: tensor<10x12x12x64xf32>, %arg2: tensor<f32>)
   -> (tensor<10x24x24x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}) {
  // #sdy.op_sharding_rule<([i, k, m, n], [i, j, l, n], [])->([i, k, m, n]) {i=10, j=12, k=24, l=12, m=24, n=64} need_replication={j, l} permutation={k, m}>
  // CHECK-NEXT: %0 = sdy.reshard %arg1 <@mesh, [{"x"}, {}, {}, {}]> : tensor<10x12x12x64xf32>
  // CHECK-NEXT: %1 = "stablehlo.select_and_scatter"(%arg0, %0, %arg2)
  // CHECK: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>
  %1 = "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = stablehlo.compare GT, %arg3, %arg4 :(tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
  },  {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
    stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 1, 2, 2, 1>,
    window_strides = array<i64: 1, 2, 2, 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>
  } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  // CHECK-NEXT: return %1 : tensor<10x24x24x64xf32>
  return %1 : tensor<10x24x24x64xf32>
}

// CHECK-LABEL: func @select_and_scatter_passthrough_dimension
func.func @select_and_scatter_passthrough_dimension(%arg0: tensor<12x24x48xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x"}, {"y"}, {"z"}]>},
                               %arg1: tensor<11x12x48xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x"}, {"y"}, {"z"}]>}, %arg2: tensor<f32>)
   -> (tensor<12x24x48xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x"}, {"y"}, {"z"}]>}) {
  // #sdy.op_sharding_rule<([j, l, m], [i, k, m], [])->([j, l, m]) {i=11, j=12, k=12, l=24, m=48} need_replication={i, k} permutation={j, l}>
  // CHECK-NEXT: %0 = sdy.reshard %arg1 <@mesh_xyzt, [{}, {}, {"z"}]> : tensor<11x12x48xf32>
  // CHECK-NEXT: %1 = "stablehlo.select_and_scatter"(%arg0, %0, %arg2)
  // CHECK: sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"x"}, {"y"}, {"z"}]>]>
  %1 = "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = stablehlo.compare GT, %arg3, %arg4 :(tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
  },  {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
    stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 1, 2, 1>,
    padding = dense<[[-1, 0], [0, 0], [0, 0]]> : tensor<3x2xi64>,
    window_strides = array<i64: 1, 2, 1>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"x"}, {"y"}, {"z"}]>]>
  } : (tensor<12x24x48xf32>, tensor<11x12x48xf32>, tensor<f32>) -> tensor<12x24x48xf32>
  // CHECK-NEXT: return %1
  return %1 : tensor<12x24x48xf32>
}
