// RUN: mpmd_opt %s -mpmd-lower-to-fragment-calls='verbose-logging=true' -split-input-file 2>&1 | FileCheck %s

!mesh_tensor = !mpmd.mesh_tensor<"m", tensor<4x8xf32>>

func.func @main(%arg0: !mesh_tensor) -> !mesh_tensor
  attributes {"topology"=#mpmd.topology<<"m": <["x"=2, "y"=2]>>>} {
  %0 = mpmd.fragment<mesh="m", origin=["foo"]> (%arg0) (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  %1 = mpmd.fragment<mesh="m", origin=["bar"]> (%0) (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  return %1 : !mesh_tensor
}

// CHECK-LABEL: Module main on mesh m will execute
// CHECK: from program with origins [foo]
// CHECK: from program with origins [bar]
// CHECK-NOT: (fwd)
// CHECK-NOT: (bwd)

// -----

!mesh_tensor_2 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func @main(%arg0: !mesh_tensor_2) -> !mesh_tensor_2
  attributes {"topology"=#mpmd.topology<<"m2": <["x"=2, "y"=2]>>>} {
  %0 = mpmd.fragment<mesh="m2", origin=["foo"]> (%arg0) (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_tensor_2) -> !mesh_tensor_2
  %1 = mpmd.fragment<mesh="m2", origin=["bar"(1)]> (%0) (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_tensor_2) -> !mesh_tensor_2
  return %1 : !mesh_tensor_2
}

// CHECK-LABEL: Module main on mesh m2 will execute
// CHECK: from program with origins [foo(fwd)]
// CHECK: from program with origins [bar(bwd)]
