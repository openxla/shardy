// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s

// CHECK:  sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>
sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @no_pad
func.func @no_pad(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
  -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: return %[[ADD]] : tensor<4x8xf32>
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

sdy.mesh @mesh_4_2_4 = <["x"=4, "y"=2, "z"=4]>

// CHECK-LABEL: func private @pad_all_reduce
func.func private @pad_all_reduce(
  %arg0: tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}], unreduced={"y"}>})
  -> (tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[AR:.*]] = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: return %[[AR]] : tensor<8x8xf32>
  %0 = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x8xf32>
  return %0 : tensor<7x8xf32>
}

// CHECK-LABEL: func private @pad_collective_permute
func.func private @pad_collective_permute(
  %arg0: tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2_4, [{"x"}, {}]>})
  -> (tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2_4, [{"z"}, {}]>}) {
  // CHECK-NEXT: %[[CP:.*]] = sdy.collective_permute %arg0 out_sharding=<@mesh_4_2_4, [{"z"}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: return %[[CP]] : tensor<8x8xf32>
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_4_2_4, [{"z"}, {}]> : tensor<7x8xf32>
  return %0 : tensor<7x8xf32>
}

// CHECK-LABEL: func private @pad_sharded_to_unreduced
func.func private @pad_sharded_to_unreduced(
  %arg0: tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>})
  -> (tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}], unreduced={"y"}>}) {
  // CHECK-NEXT: %[[S2U:.*]] = sdy.sharded_to_unreduced [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}], unreduced={"y"}> : tensor<8x8xf32>
  // CHECK-NEXT: return %[[S2U]] : tensor<8x8xf32>
  %0 = sdy.sharded_to_unreduced [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}], unreduced={"y"}> : tensor<7x8xf32>
  return %0 : tensor<7x8xf32>
}

// CHECK-LABEL: func private @pad_replicated_to_unreduced
func.func private @pad_replicated_to_unreduced(
  %arg0: tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
  -> (tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}], unreduced={"y"}>}) {
  // CHECK-NEXT: %[[R2U:.*]] = sdy.replicated_to_unreduced {"y"} %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}], unreduced={"y"}> : tensor<8x8xf32>
  // CHECK-NEXT: return %[[R2U]] : tensor<8x8xf32>
  %0 = sdy.replicated_to_unreduced {"y"} %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}], unreduced={"y"}> : tensor<7x8xf32>
  return %0 : tensor<7x8xf32>
}

// CHECK-LABEL: func private @pad_manual_computation_free_axis
func.func private @pad_manual_computation_free_axis(
  %arg0: tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>})
  -> (tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[MC:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME:   in_shardings=[<@mesh_4_2, [{"x"}, {"y"}]>]
  // CHECK-SAME:   out_shardings=[<@mesh_4_2, [{"x"}, {"y"}]>]
  // CHECK-SAME:   manual_axes={"y"} (%arg1: tensor<8x4xf32>) {
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : tensor<8x4xf32>
  // CHECK-NEXT:   sdy.return %[[ADD]] : tensor<8x4xf32>
  // CHECK-NEXT: } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[MC]] : tensor<8x8xf32>
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_4_2, [{"x"}, {"y"}]>]
    out_shardings=[<@mesh_4_2, [{"x"}, {"y"}]>]
    manual_axes={"y"} (%arg1: tensor<7x4xf32>) {
    %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : tensor<7x4xf32>
    sdy.return %1 : tensor<7x4xf32>
  } : (tensor<7x8xf32>) -> tensor<7x8xf32>
  return %0 : tensor<7x8xf32>
}

// CHECK-LABEL: func private @pad_manual_computation_unannotated_return_operand
func.func private @pad_manual_computation_unannotated_return_operand(
  %arg0: tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>})
  -> (tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>},
      tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[MC:.*]]:2 = sdy.manual_computation(%arg0)
  // CHECK-SAME:   in_shardings=[<@mesh_4_2, [{"x"}, {"y"}]>]
  // CHECK-SAME:   out_shardings=[<@mesh_4_2, [{"x"}, {"y"}]>, <@mesh_4_2, [{"x"}, {"y"}]>]
  // CHECK-SAME:   manual_axes={"y"} (%arg1: tensor<8x4xf32>) {
  // CHECK-NEXT:   %[[CST:.*]] = sdy.constant dense<1.000000e+00> : tensor<8x4xf32>
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 : tensor<8x4xf32>
  // CHECK-NEXT:   sdy.return %[[ADD]], %[[CST]] : tensor<8x4xf32>, tensor<8x4xf32>
  // CHECK-NEXT: } : (tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>)
  // CHECK-NEXT: return %[[MC]]#0, %[[MC]]#1 : tensor<8x8xf32>, tensor<8x8xf32>
  %0:2 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_4_2, [{"x"}, {"y"}]>]
    out_shardings=[<@mesh_4_2, [{"x"}, {"y"}]>, <@mesh_4_2, [{"x"}, {"y"}]>]
    manual_axes={"y"} (%arg1: tensor<7x4xf32>) {
    %cst = sdy.constant dense<1.0> : tensor<7x4xf32>
    %1 = stablehlo.add %arg1, %arg1 : tensor<7x4xf32>
    sdy.return %1, %cst : tensor<7x4xf32>, tensor<7x4xf32>
  } : (tensor<7x8xf32>) -> (tensor<7x8xf32>, tensor<7x8xf32>)
  return %0#0, %0#1 : tensor<7x8xf32>, tensor<7x8xf32>
}

// CHECK-LABEL: func private @pad_map
// CHECK-SAME: (%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>},
// CHECK-SAME:  %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>})
// CHECK-SAME: -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>})
func.func private @pad_map(
  %arg0: tensor<7x7xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>},
  %arg1: tensor<7x7xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>})
  -> (tensor<7x7xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[MAP:.*]] = "stablehlo.map"(%arg0, %arg1) <{dimensions = array<i64: 0, 1>}> ({
  // CHECK-NEXT: ^bb0(%[[X:.*]]: tensor<f32>, %[[Y:.*]]: tensor<f32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[X]], %[[Y]] : tensor<f32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {"y"}]>]>} : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%x: tensor<f32>, %y: tensor<f32>):
      %sum = stablehlo.add %x, %y : tensor<f32>
      stablehlo.return %sum : tensor<f32>
  }) {dimensions = array<i64: 0, 1>, sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {"y"}]>]>} : (tensor<7x7xf32>, tensor<7x7xf32>) -> tensor<7x7xf32>
  // CHECK-NEXT: return %[[MAP]] : tensor<8x8xf32>
  return %0 : tensor<7x7xf32>
}

