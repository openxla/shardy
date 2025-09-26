// RUN: mpmd_opt %s -mpmd-import-pipeline='name-to-mesh-assignment=f1@tpu,f2@cpu enable-heterogeneous-meshes' -split-input-file 2>&1 | FileCheck %s

module @multiple_input_meshes {
  // CHECK-DAG: sdy.mesh @tpu = <["tpu_x"=2, "tpu_y"=4]>
  // CHECK-DAG: sdy.mesh @cpu = <["cpu_z"=8]>
  // CHECK-DAG: sdy.mesh @empty_mesh = <[]>
  // CHECK-DAG: sdy.mesh @maximal_mesh = <[], device_ids=[0]>
  // CHECK-NOT: sdy.mesh @mesh
  // CHECK-NOT: sdy.mesh @mesh_0
  sdy.mesh @mesh = <["tpu_x"=8, "tpu_y"=8]>
  sdy.mesh @mesh_0 = <["cpu_z"=8]>
  sdy.mesh @empty_mesh = <[]>
  sdy.mesh @maximal_mesh = <[], device_ids=[0]>

  // CHECK: %arg0: !mpmd.mesh_tensor<"tpu", tensor<16xf32>> {sdy.sharding = #sdy.sharding<@tpu, [{"tpu_x", "tpu_y"}]>}
  // CHECK: %arg1: !mpmd.mesh_tensor<"cpu", tensor<16xf32>> {sdy.sharding = #sdy.sharding<@cpu, [{"cpu_z":(1)2}]>}
  // CHECK: %arg2: !mpmd.mesh_tensor<"tpu", tensor<16xf32>> {sdy.sharding = #sdy.sharding<@tpu, [{"tpu_x", "tpu_y"}]>}
  func.func @main(
    %arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tpu_x", "tpu_y"}]>},
    %arg1: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"cpu_z":(1)2}]>},
    %arg2: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tpu_x", "tpu_y"}]>})
      -> (tensor<16xf32>) attributes {
      topology = #mpmd.topology<<"tpu" : <["tpu_x"=2, "tpu_y"=4]>>, <"cpu" : <["cpu_z"=8]>>>} {
    %0 = mpmd.named_computation<"f1"> (%arg0, %arg2) (%arg3: tensor<16xf32>, %arg4: tensor<16xf32>) {
      %2 = stablehlo.add %arg4, %arg3 : tensor<16xf32>
      mpmd.return %2 : tensor<16xf32>
    } : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %1 = mpmd.named_computation<"f2"> (%arg1, %0) (%arg3: tensor<16xf32>, %arg4: tensor<16xf32>) {
      %2 = stablehlo.add %arg4, %arg3 : tensor<16xf32>
      mpmd.return %2 : tensor<16xf32>
    } : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    return %1 : tensor<16xf32>
  }
}
