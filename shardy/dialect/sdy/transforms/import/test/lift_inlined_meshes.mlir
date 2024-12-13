// RUN: sdy_opt -split-input-file %s -sdy-lift-inlined-meshes | FileCheck %s

// CHECK: sdy.mesh @mesh1 = <["x"=2, "y"=2]>
sdy.mesh @mesh1 = <["x"=2, "y"=2]>

// CHECK: sdy.mesh @mesh2 = <["x"=2, "y"=4]>
sdy.mesh @mesh2 = <["x"=2, "y"=4]>

// CHECK-LABEL: func @no_inlined_meshes_or_duplicates(
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}]>},
// CHECK-SAME:    %arg1: tensor<8x8xf32>)
func.func @no_inlined_meshes_or_duplicates(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}]>}, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"x"}, {}]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"x"}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK: sdy.mesh @mesh1 = <["x"=2, "y"=2]>
sdy.mesh @mesh1 = <["x"=2, "y"=2]>

// CHECK: sdy.mesh @mesh2 = <["x"=2, "y"=4]>
sdy.mesh @mesh2 = <["x"=2, "y"=4]>

// CHECK-NOT: sdy.mesh @mesh3
sdy.mesh @mesh3 = <["x"=2, "y"=2]>

// CHECK-LABEL: func @no_inlined_meshes_with_duplicates(
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}]>},
// CHECK-SAME:    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{}, {}]>})
func.func @no_inlined_meshes_with_duplicates(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{}, {"y"}]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{}, {"y"}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK: sdy.mesh @mesh = <["x"=2, "y"=4]>
sdy.mesh @mesh = <["x"=2, "y"=4]>

// CHECK: sdy.mesh @mesh_0 = <["x"=2, "y"=2]>

// CHECK-LABEL: func @inlined_mesh
func.func @inlined_mesh(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"x"}, {}]>]>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK: sdy.mesh @empty_mesh = <[]>

// CHECK-LABEL: func @inlined_empty_mesh(
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>})
func.func @inlined_empty_mesh(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<[]>, [{}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@empty_mesh, [{}, {}]>]>
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<mesh<[]>, [{}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK: sdy.mesh @maximal_mesh_3 = <[], device_ids=[3]>
// CHECK: sdy.mesh @maximal_mesh_7 = <[], device_ids=[7]>

// CHECK-LABEL: func @inlined_maximal_mesh(
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@maximal_mesh_3, []>})
func.func @inlined_maximal_mesh(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<[], device_ids=[3]>, []>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh_7, []>]>
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<mesh<[], device_ids=[7]>, []>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK: sdy.mesh @mesh = <["x"=2, "y"=4]>
sdy.mesh @mesh = <["x"=2, "y"=4]>

// CHECK: sdy.mesh @mesh_0 = <["x"=2], device_ids=[1, 0]>
sdy.mesh @mesh_0 = <["x"=2], device_ids=[1, 0]>

// CHECK-NOT: @mesh_1
sdy.mesh @mesh_1 = <["x"=2, "y"=4]>

// CHECK: sdy.mesh @some_maximal_mesh = <[], device_ids=[5]>
sdy.mesh @some_maximal_mesh = <[], device_ids=[5]>

// CHECK-NOT: @copy_of_mesh
sdy.mesh @copy_of_mesh = <["x"=2, "y"=4]>

// CHECK-NOT: @copy_of_mesh_0
sdy.mesh @copy_of_mesh_0 = <["x"=2], device_ids=[1, 0]>

// CHECK: sdy.mesh @mesh_1 = <["x"=8]>
// CHECK: sdy.mesh @mesh_2 = <["a"=4]>
// CHECK: sdy.mesh @maximal_mesh_2 = <[], device_ids=[2]>

// CHECK-LABEL: func @many_inlined_meshes_and_duplicates(
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"x"}, {}]>},
// CHECK-SAME:    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}], replicated={"y"}>})
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{?}, {?}]>},
// CHECK-SAME:        tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>}) {
func.func @many_inlined_meshes_and_duplicates(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2], device_ids=[1, 0]>, [{"x"}, {}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}], replicated={"y"}>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=8]>, [{?}, {?}]>},
        tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=4]>, [{}, {}]>}) {
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[ADD_0]], %arg1 : tensor<8x8xf32>
  // CHECK-NEXT: %[[SC_0:.*]] = sdy.sharding_constraint %[[ADD_1]] <@mesh, [{}, {"y"}]>
  // CHECK-NEXT: %[[ADD_2:.*]] = stablehlo.add %[[SC_0]], %[[SC_0]] {sdy.sharding = #sdy.sharding_per_value<[<@some_maximal_mesh, []>]>}
  // CHECK-NEXT: %[[SC_1:.*]] = sdy.sharding_constraint %[[ADD_2]] <@mesh_2, [{}, {"a"}]>
  // CHECK-NEXT: %[[ADD_3:.*]] = stablehlo.add %[[ADD_2]], %[[SC_1]] : tensor<8x8xf32>
  // CHECK-NEXT: return %[[SC_1]], %[[ADD_3]]
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"x"}, {}]>]>} : tensor<8x8xf32>
  %1 = stablehlo.add %0, %arg1 : tensor<8x8xf32>
  %2 = sdy.sharding_constraint %1 <@copy_of_mesh, [{}, {"y"}]> : tensor<8x8xf32>
  %3 = stablehlo.add %2, %2 {sdy.sharding = #sdy.sharding_per_value<[<mesh<[], device_ids=[5]>, []>]>} : tensor<8x8xf32>
  %4 = sdy.sharding_constraint %3 <mesh<["a"=4]>, [{}, {"a"}]> : tensor<8x8xf32>
  %5 = stablehlo.add %3, %4 : tensor<8x8xf32>
  return %4, %5 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @another_func_in_module(
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}]>})
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}]>}) {
func.func @another_func_in_module(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@copy_of_mesh_0, [{}, {}]>}) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"a"}, {}]>]>}
  // CHECK-NEXT: %[[SC:.*]] = sdy.sharding_constraint %[[ADD]] <@maximal_mesh_2, []>
  // CHECK-NEXT: return %[[SC]]
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=4]>, [{"a"}, {}]>]>} : tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <mesh<[], device_ids=[2]>, []> : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// CHECK: sdy.mesh @mesh = <["x"=2]>
// CHECK: sdy.mesh @mesh_0 = <["y"=4]>
// CHECK: sdy.mesh @mesh_1 = <["x"=2], device_ids=[1, 0]>

// CHECK-LABEL: func @single_sharding_sdy_ops
func.func @single_sharding_sdy_ops(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[SC:.*]] = sdy.data_flow_edge %arg0 sharding=<@mesh, [{"x"}, {}]>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[SC]] <@mesh_0, [{}, {"y"}]>
  // CHECK-NEXT: sdy.sharding_constraint %[[RESHARD]] <@mesh_1, [{}, {"x"}]>
  %0 = sdy.data_flow_edge %arg0 sharding=<mesh<["x"=2]>, [{"x"}, {}]> : tensor<8x8xf32>
  %1 = sdy.reshard %0 <mesh<["y"=4]>, [{}, {"y"}]> : tensor<8x8xf32>
  %2 = sdy.sharding_constraint %1 <mesh<["x"=2], device_ids=[1, 0]>, [{}, {"x"}]> : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}

// -----

// CHECK: sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @manual_computation
func.func @manual_computation(%arg0: tensor<16x32xf32>, %arg1: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK-NEXT: sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{"x"}, {}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"}
  %0 = sdy.manual_computation(%arg0, %arg1)
      in_shardings=[<mesh<["x"=2]>, [{"x"}, {}]>, <mesh<["x"=2]>, [{"x"}, {}]>]
      out_shardings=[<mesh<["x"=2]>, [{"x"}, {}]>] manual_axes={"x"}
      (%arg2: tensor<8x32xf32>, %arg3: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg2, %arg3 : tensor<8x32xf32>
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

// CHECK: sdy.mesh @mesh = <["x"=4]>
sdy.mesh @mesh = <["x"=4]>

// CHECK: sdy.mesh @mesh_0 = <["y"=8]>

// CHECK-LABEL: func @named_computation
func.func @named_computation(%arg0: tensor<16x32xf32>, %arg1: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK-NEXT: sdy.named_computation<"foo">(%arg0, %arg1)
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{?}, {?}]>]
  // CHECK-SAME:   out_shardings=[<@mesh_0, [{}, {"y"}]>]
  %0 = sdy.named_computation<"foo">(%arg0, %arg1)
      in_shardings=[<@mesh, [{"x"}, {}]>, <mesh<["x"=4]>, [{?}, {?}]>]
      out_shardings=[<mesh<["y"=8]>, [{}, {"y"}]>]
      (%arg2: tensor<16x32xf32>, %arg3: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg2, %arg3 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}
