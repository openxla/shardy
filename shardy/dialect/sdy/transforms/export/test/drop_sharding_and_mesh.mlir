// RUN: sdy_opt %s -split-input-file -sdy-drop-sharding-and-mesh | FileCheck %s --check-prefixes=CHECK,DROP
// RUN: sdy_opt %s -split-input-file -sdy-drop-sharding-and-mesh="keep-module-sharding-metadata=true" | FileCheck %s --check-prefixes=CHECK,KEEP

// CHECK-NOT: sdy.mesh
sdy.mesh @mesh_2 = <["x"=2]>

// CHECK-LABEL: func @drop_sharding
// CHECK-SAME:    %arg0: tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NOT:     sdy.sharding
func.func @drop_sharding(%arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
    -> (tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  // CHECK-NEXT: stablehlo.add
  // CHECK-NOT:  sdy.sharding
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @drop_multiple_sharding
// CHECK-SAME:    %arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>)
// CHECK-NOT:     sdy.sharding
func.func @drop_multiple_sharding(
    %arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {"x"}]>},
    %arg1: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) -> tensor<2x4xf32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: sdy.mesh @single_dev_0 = <[], device_ids=[0]>
// CHECK-NOT:   sdy.mesh @mesh_2
sdy.mesh @mesh_2 = <["x"=2]>
sdy.mesh @single_dev_0 = <[], device_ids=[0]>

// CHECK-LABEL: func @keep_single_device_send_recv_sharding
func.func @keep_single_device_send_recv_sharding(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>},
    %token: !stablehlo.token) -> (tensor<8x16xf32>, !stablehlo.token) {
  // CHECK-NEXT: %[[SEND:.*]] = "stablehlo.send"(%arg0, %arg1)
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_0, []>]>}
  %0 = "stablehlo.send"(%arg0, %token) <{
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>,
    is_host_transfer = true
  }> {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>]>
  } : (tensor<8x16xf32>, !stablehlo.token) -> !stablehlo.token
  // CHECK-NEXT: %[[RECV:.*]]:2 = "stablehlo.recv"(%[[SEND]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_0, []>, <@single_dev_0, []>]>}
  %1:2 = "stablehlo.recv"(%0) <{
    channel_handle = #stablehlo.channel_handle<handle = 2, type = 3>,
    is_host_transfer = true
  }> {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>, #sdy.sharding<@single_dev_0, []>]>
  } : (!stablehlo.token) -> (tensor<8x16xf32>, !stablehlo.token)
  // CHECK-NEXT: return %[[RECV]]#0, %[[RECV]]#1
  return %1#0, %1#1 : tensor<8x16xf32>, !stablehlo.token
}

// -----

// CHECK-LABEL: module @module_input_output_shardings
// KEEP-SAME: sdy.output_shardings = #sdy.sharding_per_value<[<mesh<["x"=2]>, [{"x"}]>]>,
// KEEP-SAME: sdy.parameters_shardings = #sdy.sharding_per_value<[<mesh<["x"=2]>, [{"x"}]>]>
// DROP-NOT: sdy.output_shardings
// DROP-NOT: sdy.parameters_shardings
// CHECK-NOT: sdy.mesh
// CHECK:     func.func @main
module @module_input_output_shardings attributes {
  sdy.parameters_shardings = #sdy.sharding_per_value<[<mesh<["x"=2]>, [{"x"}]>]>,
  sdy.output_shardings = #sdy.sharding_per_value<[<mesh<["x"=2]>, [{"x"}]>]>
} {
  sdy.mesh @mesh_2 = <["x"=2]>
  func.func @main() {
    return
  }
}
