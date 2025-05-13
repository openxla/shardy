// RUN: sdy_opt -split-input-file %s -sdy-inline-meshes | FileCheck %s

// CHECK-LABEL: func @no_lifted_meshes(
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>}
func.func @no_lifted_meshes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

// -----

// CHECK-NOT: sdy.mesh @empty_mesh = <[]>
sdy.mesh @empty_mesh = <[]>

// CHECK-LABEL: func @lifted_empty_mesh(
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<[]>, [{}, {}]>})
func.func @lifted_empty_mesh(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<mesh<[]>, [{}, {}]>]>
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@empty_mesh, [{}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK-NOT: sdy.mesh @mesh = <["x"=2, "y"=4]>
sdy.mesh @mesh = <["x"=2, "y"=4]>

// CHECK-LABEL: func @lifted_and_inlined_meshes_two_functions
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2, "y"=4]>, [{"x"}, {}]>}
func.func @lifted_and_inlined_meshes_two_functions(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @another_function
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2, "y"=4]>, [{"x"}, {}]>}
func.func @another_function(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK-NOT: sdy.mesh @maximal_mesh_3 = <[], device_ids=[3]>
sdy.mesh @maximal_mesh_3 = <[], device_ids=[3]>
// CHECK-NOT: sdy.mesh @maximal_mesh_7 = <[], device_ids=[7]>
sdy.mesh @maximal_mesh_7 = <[], device_ids=[7]>

// CHECK-LABEL: func @lifted_maximal_mesh(
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<[], device_ids=[3]>, []>})
func.func @lifted_maximal_mesh(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@maximal_mesh_3, []>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<mesh<[], device_ids=[7]>, []>]>
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh_7, []>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK-NOT: sdy.mesh @mesh = <["x"=2]>
sdy.mesh @mesh = <["x"=2]>
// CHECK-NOT: sdy.mesh @mesh_0 = <["y"=4]>
sdy.mesh @mesh_0 = <["y"=4]>
// CHECK-NOT: sdy.mesh @mesh_1 = <["x"=2], device_ids=[1, 0]>
sdy.mesh @mesh_1 = <["x"=2], device_ids=[1, 0]>

// CHECK-LABEL: func @single_sharding_sdy_ops
func.func @single_sharding_sdy_ops(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[SC:.*]] = sdy.data_flow_edge %arg0 sharding=<mesh<["x"=2]>, [{"x"}, {}]>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[SC]] <mesh<["y"=4]>, [{}, {"y"}]>
  // CHECK-NEXT: sdy.sharding_constraint %[[RESHARD]] <mesh<["x"=2], device_ids=[1, 0]>, [{}, {"x"}]>
  %0 = sdy.data_flow_edge %arg0 sharding=<@mesh, [{"x"}, {}]> : tensor<8x8xf32>
  %1 = sdy.reshard %0 <@mesh_0, [{}, {"y"}]> : tensor<8x8xf32>
  %2 = sdy.sharding_constraint %1 <@mesh_1, [{}, {"x"}]> : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}

// -----

// CHECK-NOT: sdy.mesh @mesh = <["x"=2]>
sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @manual_computation
func.func @manual_computation(%arg0: tensor<16x32xf32>, %arg1: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK-NEXT: sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME:   in_shardings=[<mesh<["x"=2]>, [{"x"}, {}]>, <mesh<["x"=2]>, [{"x"}, {}]>]
  // CHECK-SAME:   out_shardings=[<mesh<["x"=2]>, [{"x"}, {}]>] manual_axes={"x"}
  %0 = sdy.manual_computation(%arg0, %arg1)
      in_shardings=[<@mesh, [{"x"}, {}]>, <mesh<["x"=2]>, [{"x"}, {}]>]
      out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"}
      (%arg2: tensor<8x32xf32>, %arg3: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg2, %arg3 : tensor<8x32xf32>
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

// CHECK-NOT: sdy.mesh @mesh = <["x"=4]>
sdy.mesh @mesh = <["x"=4]>

// CHECK-NOT: sdy.mesh @mesh_0 = <["y"=8]>
sdy.mesh @mesh_0 = <["y"=8]>

// CHECK-LABEL: func @named_computation
func.func @named_computation(%arg0: tensor<16x32xf32>, %arg1: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK-NEXT: sdy.named_computation<"foo">(%arg0, %arg1)
  // CHECK-SAME:   in_shardings=[<mesh<["x"=4]>, [{"x"}, {}]>, <mesh<["x"=4]>, [{?}, {?}]>]
  // CHECK-SAME:   out_shardings=[<mesh<["y"=8]>, [{}, {"y"}]>]
  %0 = sdy.named_computation<"foo">(%arg0, %arg1)
      in_shardings=[<@mesh, [{"x"}, {}]>, <mesh<["x"=4]>, [{?}, {?}]>]
      out_shardings=[<mesh<["y"=8]>, [{}, {"y"}]>]
      (%arg2: tensor<16x32xf32>, %arg3: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg2, %arg3 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}
