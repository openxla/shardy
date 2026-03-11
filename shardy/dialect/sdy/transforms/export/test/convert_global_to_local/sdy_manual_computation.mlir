// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_4 = <["x"=4]>
sdy.mesh @mesh_4 = <["x"=4]>
// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
// CHECK: sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>
sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>

// CHECK-LABEL: func @no_free_axes_two_manual_axes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"z"}]>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"z"}]>}) {
func.func @no_free_axes_two_manual_axes(%arg0 : tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"z"}]>})
  -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"z"}]>}) {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[ARG0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4_2, [{"x"}, {"z"}]>]>} : tensor<8x16xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[ABS]], %[[ABS]] : tensor<8x16xf32>
  // CHECK-NEXT: %[[TANH:.*]] = stablehlo.tanh %[[ADD]] : tensor<8x16xf32>
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4_2, [{"x"}, {"z"}]>]>} : tensor<16x32xf32>
  %1 = sdy.manual_computation(%0)
    in_shardings=[<@mesh_2_4_2, [{"x"}, {"z"}]>]
    out_shardings=[<@mesh_2_4_2, [{"x"}, {"z"}]>]
    manual_axes={"x", "z"}
    (%arg1: tensor<8x16xf32>) {
    %2 = stablehlo.add %arg1, %arg1 : tensor<8x16xf32>
    %3 = stablehlo.tanh %2  : tensor<8x16xf32>
    // CHECK-NEXT: return %[[TANH]] : tensor<8x16xf32>
    sdy.return %3 : tensor<8x16xf32>
  } : (tensor<16x32xf32>) -> (tensor<16x32xf32>)
  func.return %1 : tensor<16x32xf32>
}

// CHECK-LABEL: func @one_free_axis_one_manual_axis
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>})
// CHECK-SAME: -> (tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>}) {
func.func @one_free_axis_one_manual_axis(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>})
  -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>}) {
  // CHECK-NEXT: %[[TANH:.*]] = stablehlo.tanh %[[ARG0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"y"}, {}]>]>} : tensor<2x8xf32>
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
    out_shardings=[<@mesh_2_4, [{"x", "y"}, {}]>]
    manual_axes={"x"} (%arg1: tensor<8x8xf32>) {
    %1 = stablehlo.tanh %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"y"}, {}]>]>} : tensor<8x8xf32>
    sdy.return %1 : tensor<8x8xf32>
  } : (tensor<16x8xf32>) -> tensor<16x8xf32>
  // CHECK-NEXT: return %[[TANH]] : tensor<2x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @nested_manual_computations
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "z"}, {"y"}]>})
// CHECK-SAME: -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "z"}, {"y"}]>}) {
func.func @nested_manual_computations(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "z"}, {"y"}]>})
  -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "z"}, {"y"}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_2_4_2, [{"x", "z"}, {"y"}]>]
    out_shardings=[<@mesh_2_4_2, [{"x", "z"}, {"y"}]>]
    manual_axes={"x"} (%arg1: tensor<8x32xf32>) {
    // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[ARG0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4_2, [{"z"}, {"y"}]>]>} : tensor<4x8xf32>
    %1 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4_2, [{"z"}, {"y"}]>]>}: tensor<8x32xf32>
    %2 = sdy.manual_computation(%1)
      in_shardings=[<@mesh_2_4_2, [{"z"}, {"y"}]>]
      out_shardings=[<@mesh_2_4_2, [{"z"}, {"y"}]>]
      manual_axes={"z"} (%arg2: tensor<4x32xf32>) {
      // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[ABS]], %[[ABS]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4_2, [{}, {"y"}]>]>} : tensor<4x8xf32>
      %3 = stablehlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4_2, [{}, {"y"}]>]>}: tensor<4x32xf32>
      sdy.return %3 : tensor<4x32xf32>
    } : (tensor<8x32xf32>) -> tensor<8x32xf32>
    sdy.return %2 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // CHECK-NEXT: return %[[ADD]] : tensor<4x8xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func @stablehlo_all_gather
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {}]>}) {
func.func @stablehlo_all_gather(%arg0 : tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>})
  -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_4, [{"x"}, {}]>]
    out_shardings=[<@mesh_4, [{}, {}]>]
    manual_axes={"x"} (%arg1: tensor<4x8xf32>) {
      // CHECK-NEXT: %[[RES:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{
      // CHECK-SAME:   all_gather_dim = 0 : i64,
      // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
      // CHECK-SAME: }> : (tensor<4x8xf32>) -> tensor<16x8xf32>
      %1 = "stablehlo.all_gather"(%arg1) {
        all_gather_dim = 0 : i64,
        replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
        channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
      } : (tensor<4x8xf32>) -> tensor<16x8xf32>
      sdy.return %1 : tensor<16x8xf32>
  } : (tensor<16x8xf32>) -> (tensor<16x8xf32>)
  // CHECK-NEXT: return %[[RES]] : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @stablehlo_all_reduce
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {}]>}) {
func.func @stablehlo_all_reduce(%arg0 : tensor<64x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>})
  -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_4, [{"x"}, {}]>]
    out_shardings=[<@mesh_4, [{}, {}]>]
    manual_axes={"x"}
    (%arg1: tensor<16x8xf32>) {
      // CHECK-NEXT: %[[RES:.*]] = "stablehlo.all_reduce"(%[[ARG0]]) <{
      // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
      // CHECK-SAME: }> ({
      // CHECK-NEXT: ^bb0(%[[RED_ARG1:.*]]: tensor<f32>, %[[RED_ARG2:.*]]: tensor<f32>):
      // CHECK-NEXT:   %[[SUM:.*]] = stablehlo.add %[[RED_ARG1]], %[[RED_ARG2]] : tensor<f32>
      // CHECK-NEXT:   stablehlo.return %[[SUM]] : tensor<f32>
      // CHECK-NEXT: }) : (tensor<16x8xf32>) -> tensor<16x8xf32>
      %1 = "stablehlo.all_reduce"(%arg1) ({
        ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
          %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
          stablehlo.return %2 : tensor<f32>
      }) {
        replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
      } : (tensor<16x8xf32>) -> tensor<16x8xf32>
      sdy.return %1 : tensor<16x8xf32>
  } : (tensor<64x8xf32>) -> (tensor<16x8xf32>)
  // CHECK-NEXT: return %[[RES]] : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @stablehlo_all_to_all
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {"x"}]>}) {
func.func @stablehlo_all_to_all(%arg0 : tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>})
  -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {"x"}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_4, [{"x"}, {}]>]
    out_shardings=[<@mesh_4, [{}, {"x"}]>]
    manual_axes={"x"} (%arg1: tensor<4x8xf32>) {
    // CHECK-NEXT: %[[RES:.*]] = "stablehlo.all_to_all"(%[[ARG0]]) <{
    // CHECK-SAME:   concat_dimension = 0 : i64,
    // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    // CHECK-SAME:   split_count = 4 : i64,
    // CHECK-SAME:   split_dimension = 1 : i64
    // CHECK-SAME: }> : (tensor<4x8xf32>) -> tensor<16x2xf32>
    %1 = "stablehlo.all_to_all"(%arg1) {
      split_dimension = 1 : i64,
      concat_dimension = 0 : i64,
      split_count = 4 : i64,
      replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
    } : (tensor<4x8xf32>) -> tensor<16x2xf32>
    sdy.return %1 : tensor<16x2xf32>
  } : (tensor<16x8xf32>) -> (tensor<16x8xf32>)
  // CHECK-NEXT: return %[[RES]] : tensor<16x2xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @stablehlo_collective_permute
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>})
func.func @stablehlo_collective_permute(%arg0 : tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>})
  -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_4, [{"x"}, {}]>]
    out_shardings=[<@mesh_4, [{"x"}, {}]>]
    manual_axes={"x"} (%arg1: tensor<4x8xf32>) {
    // CHECK-SAME: -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) {
    // CHECK-NEXT: %[[RES:.*]] = "stablehlo.collective_permute"(%[[ARG0]]) <{
    // CHECK-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>
    // CHECK-SAME: }> : (tensor<4x8xf32>) -> tensor<4x8xf32>
    %1 = "stablehlo.collective_permute"(%arg1) {
       source_target_pairs = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>
    } : (tensor<4x8xf32>) -> tensor<4x8xf32>
    sdy.return %1 : tensor<4x8xf32>
  } : (tensor<16x8xf32>) -> (tensor<16x8xf32>)
  // CHECK-NEXT: return %[[RES]] : tensor<4x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @stablehlo_reduce_scatter
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {}]>})
// CHECK-SAME: -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) {
func.func @stablehlo_reduce_scatter(%arg0 : tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {}]>})
  -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_4, [{}, {}]>]
    out_shardings=[<@mesh_4, [{"x"}, {}]>]
    manual_axes={"x"} (%arg1: tensor<16x8xf32>) {
    // CHECK-NEXT: %[[RES:.*]] = "stablehlo.reduce_scatter"(%[[ARG0]]) <{
    // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 2, type = 0>,
    // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    // CHECK-SAME:   scatter_dimension = 0 : i64
    // CHECK-SAME: }> ({
    // CHECK-NEXT: ^bb0(%[[RED_ARG1:.*]]: tensor<f32>, %[[RED_ARG2:.*]]: tensor<f32>):
    // CHECK-NEXT:   %[[SUM:.*]] = stablehlo.add %[[RED_ARG1]], %[[RED_ARG2]] : tensor<f32>
    // CHECK-NEXT:   stablehlo.return %[[SUM]] : tensor<f32>
    // CHECK-NEXT: }) : (tensor<16x8xf32>) -> tensor<4x8xf32>
    %1 = "stablehlo.reduce_scatter"(%arg1) ({
      ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
        %3 = "stablehlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%3) : (tensor<f32>) -> ()
      }) {
      scatter_dimension = 0 : i64,
      replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 2, type = 0>
    } : (tensor<16x8xf32>) -> tensor<4x8xf32>
    sdy.return %1 : tensor<4x8xf32>
  } : (tensor<16x8xf32>) -> (tensor<16x8xf32>)
  // CHECK-NEXT: return %[[RES]] : tensor<4x8xf32>
  return %0 : tensor<16x8xf32>
}
