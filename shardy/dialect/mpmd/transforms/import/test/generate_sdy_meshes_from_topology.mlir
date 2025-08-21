// RUN: mpmd_opt %s -mpmd-generate-sdy-meshes-from-topology -split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: module @multiple_input_meshes
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

  // CHECK: %arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@tpu, [{"x", "y"}]>}
  // CHECK: %arg1: tensor<16xf32> {sdy.sharding = #sdy.sharding<@cpu, [{"z":(1)2}]>}
  // CHECK: %arg2: tensor<16xf32> {sdy.sharding = #sdy.sharding<@tpu, [{"x", "y"}]>}
  func.func @main(
    %arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tpu_x", "tpu_y"}]>},
    %arg1: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"cpu_z":(1)2}]>},
    %arg2: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tpu_x", "tpu_y"}]>})
      -> (tensor<16xf32>) attributes {
      topology = #mpmd.topology<<"tpu" : <["x"=2, "y"=4]>>, <"cpu" : <["z"=8]>>>} {
    %0 = mpmd.named_computation<"stage1"> (%arg0, %arg2) (%arg3: tensor<16xf32>, %arg4: tensor<16xf32>) {
      %2 = stablehlo.add %arg4, %arg3 : tensor<16xf32>
      mpmd.return %2 : tensor<16xf32>
    } : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %1 = mpmd.named_computation<"stage2"> (%arg1, %0) (%arg3: tensor<16xf32>, %arg4: tensor<16xf32>) {
      %2 = stablehlo.add %arg4, %arg3 : tensor<16xf32>
      mpmd.return %2 : tensor<16xf32>
    } : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    return %1 : tensor<16xf32>
  }
}

// -----

// CHECK-LABEL: module @empty_mesh
module @empty_mesh {
  // CHECK-DAG: sdy.mesh @tpu = <["x"=2]>
  // CHECK-DAG: sdy.mesh @empty_mesh = <[]>
  sdy.mesh @mesh = <["tpu_x"=2]>
  sdy.mesh @empty_mesh = <[]>

  // CHECK: %arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@tpu, [{"x"}]>}
  // CHECK: %arg1: tensor<16xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}
  func.func @main(
    %arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tpu_x"}]>},
    %arg1: tensor<16xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>})
      -> (tensor<16xf32>) attributes {
      topology = #mpmd.topology<<"tpu" : <["x"=2]>>>} {
    %0 = mpmd.named_computation<"stage1"> (%arg0, %arg1) (%arg2: tensor<16xf32>, %arg3: tensor<16xf32>) {
      %2 = stablehlo.add %arg3, %arg2 : tensor<16xf32>
      mpmd.return %2 : tensor<16xf32>
    } : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}

// -----

// CHECK-LABEL: module @maximal_mesh
module @maximal_mesh {
  // CHECK-DAG: sdy.mesh @tpu = <["x"=2]>
  // CHECK-DAG: sdy.mesh @maximal_mesh = <[], device_ids=[0]>
  sdy.mesh @mesh = <["tpu_x"=2]>
  sdy.mesh @maximal_mesh = <[], device_ids=[0]>

  // CHECK: %arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@tpu, [{"x"}]>}
  // CHECK: %arg1: tensor<16xf32> {sdy.sharding = #sdy.sharding<@maximal_mesh, []>}
  func.func @main(
    %arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tpu_x"}]>},
    %arg1: tensor<16xf32> {sdy.sharding = #sdy.sharding<@maximal_mesh, []>})
      -> (tensor<16xf32>) attributes {
      topology = #mpmd.topology<<"tpu" : <["x"=2]>>>} {
    %0 = mpmd.named_computation<"stage1"> (%arg0, %arg1) (%arg2: tensor<16xf32>, %arg3: tensor<16xf32>) {
      %2 = stablehlo.add %arg3, %arg2 : tensor<16xf32>
      mpmd.return %2 : tensor<16xf32>
    } : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}
