// RUN: mpmd_opt %s -mpmd-import-pipeline='name-to-mesh-assignment=f1@tpu,f2@cpu enable-heterogeneous-meshes' -mpmd-optimize-pipeline -mpmd-sharding-propagation-pipeline -mpmd-export-pipeline 2>&1 | FileCheck %s

module @multiple_input_meshes {
  // CHECK-DAG: sdy.mesh @tpu = <["x"=2, "y"=4]>
  // CHECK-DAG: sdy.mesh @cpu = <["z"=8]>
  // CHECK-DAG: sdy.mesh @empty_mesh = <[]>
  // CHECK-DAG: sdy.mesh @maximal_mesh = <[], device_ids=[0]>
  // CHECK-NOT: sdy.mesh @mesh
  // CHECK-NOT: sdy.mesh @mesh_0
  sdy.mesh @mesh = <["tpu_x"=8, "tpu_y"=8]>
  sdy.mesh @mesh_0 = <["cpu_z"=8]>
  sdy.mesh @empty_mesh = <[]>
  sdy.mesh @maximal_mesh = <[], device_ids=[0]>

  // CHECK: %arg0: !mpmd.mesh_tensor<"tpu", tensor<16xf32>, sharding=<@tpu, [{"x", "y"}]>>
  // CHECK: %arg1: !mpmd.mesh_tensor<"cpu", tensor<16xf32>, sharding=<@cpu, [{"z"}]>>
  // CHECK: %arg2: !mpmd.mesh_tensor<"tpu", tensor<16xf32>, sharding=<@tpu, [{"x", "y"}]>>
  func.func @main(
    %arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tpu_x", "tpu_y"}]>},
    %arg1: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"cpu_z"}]>},
    %arg2: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tpu_x", "tpu_y"}]>})
      -> (tensor<16xf32>) attributes {
      topology = #mpmd.topology<<"tpu" : <["x"=2, "y"=4]>>, <"cpu" : <["z"=8]>>>} {
    // CHECK: %[[FRAGMENT_CALL1:.*]] = mpmd.fragment_call<mesh="tpu", origin=["f1"]> @p0_f1_fwd.multiple_input_meshes(%arg0, %arg2)
    %0 = mpmd.named_computation<"f1"> (%arg0, %arg2) (%arg3: tensor<16xf32>, %arg4: tensor<16xf32>) {
      %2 = stablehlo.add %arg4, %arg3 : tensor<16xf32>
      mpmd.return %2 : tensor<16xf32>
    } : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    // CHECK: %[[TRANSFER:.*]] = mpmd.transfer %[[FRAGMENT_CALL1]]
    // CHECK: %[[FRAGMENT_CALL2:.*]] = mpmd.fragment_call<mesh="cpu", origin=["f2"]> @p1_f2_fwd.multiple_input_meshes(%arg1, %[[TRANSFER]])
    %1 = mpmd.named_computation<"f2"> (%arg1, %0) (%arg3: tensor<16xf32>, %arg4: tensor<16xf32>) {
      %2 = stablehlo.add %arg4, %arg3 : tensor<16xf32>
      mpmd.return %2 : tensor<16xf32>
    } : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    return %1 : tensor<16xf32>
  }
}
// CHECK-LABEL: func.func @p0_f1_fwd.multiple_input_meshes
// CHECK-LABEL: func.func @p1_f2_fwd.multiple_input_meshes
