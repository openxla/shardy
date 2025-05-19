// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xyzt = <["x"=4, "y"=4, "z"=4, "t"=8]>

// CHECK-LABEL: func @manual_computation
func.func @manual_computation(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: sdy.manual_computation(%[[RESHARD]])
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<210xf32>) {
    %2 = stablehlo.abs %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: sdy.return %[[RESHARD]] : tensor<210xf32>
    sdy.return %2 : tensor<210xf32>
  } : (tensor<210xf32>) -> (tensor<210xf32>)
  %1 = stablehlo.negate %0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func @manual_computation_with_manual_axes
func.func @manual_computation_with_manual_axes(%arg0: tensor<208xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x","y"}]>}) -> (tensor<208xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x","z"}]>}) {
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_xyzt, [{"x","y"}]>] out_shardings=[<@mesh_xyzt, [{"x", "z"}]>] manual_axes={"x"} (%arg1: tensor<52xf32>) {
    // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyzt, [{"t"}]> : tensor<52xf32>
    // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"t"}]>]>} : tensor<52xf32>
    // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[ABS]] <@mesh_xyzt, [{"z"}]> : tensor<52xf32>
    // CHECK-NEXT: sdy.return %[[RESHARD2]] : tensor<52xf32>
    %2 = stablehlo.abs %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh_xyzt, [{"t"}]>]>} : tensor<52xf32>
    sdy.return %2 : tensor<52xf32>
  } : (tensor<208xf32>) -> (tensor<208xf32>)
  %1 = stablehlo.negate %0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_xyzt, [{"x","z"}]>]>} : tensor<208xf32>
  return %1 : tensor<208xf32>
}
