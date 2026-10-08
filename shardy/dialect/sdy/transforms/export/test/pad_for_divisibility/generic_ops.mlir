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
