// RUN: mpmd_opt %s -mpmd-rename-meshes 2>&1 | FileCheck %s

// CHECK-LABEL: module @rename_meshes
module @rename_meshes {
  // CHECK-NOT: sdy.mesh @mesh
  // CHECK-NOT: sdy.mesh @mesh_0
  // CHECK-DAG: sdy.mesh @__mesh = <["x"=2, "y"=4]>
  // CHECK-DAG: sdy.mesh @__mesh_0 = <["z"=8]>
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  sdy.mesh @mesh_0 = <["z"=8]>

  // CHECK: %arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@__mesh, [{"x", "y"}]>}
  // CHECK: %arg1: tensor<16xf32> {sdy.sharding = #sdy.sharding<@__mesh_0, [{"z":(1)2}]>}
  func.func @main(
    %arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>},
    %arg1: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"z":(1)2}]>})
      -> (tensor<16xf32>) {
    return %arg0 : tensor<16xf32>
  }
}
