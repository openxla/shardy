// RUN: mpmd_opt %s 2>&1 | FileCheck %s

// CHECK-LABEL: func @transfer
func.func @transfer(%arg0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
    -> !mpmd.mesh_tensor<"mesh2", tensor<12x16xf32>, sharding=<@mesh, [{"z"}, {?}]>> attributes {
    "topology"=#mpmd.topology<
      <"mesh1": <["x"=2, "y"=4]>>,
      <"mesh2": <["z"=3]>>
    >} {
  // CHECK:      mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
  // CHECK-SAME:   -> !mpmd.mesh_tensor<"mesh2", tensor<12x16xf32>, sharding=<@mesh, [{"z"}, {?}]>>
  %0 = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>) -> !mpmd.mesh_tensor<"mesh2", tensor<12x16xf32>, sharding=<@mesh, [{"z"}, {?}]>>
  func.return %0 : !mpmd.mesh_tensor<"mesh2", tensor<12x16xf32>, sharding=<@mesh, [{"z"}, {?}]>>
}
