// RUN: mpmd_opt %s -mpmd-copy-topology-from-main -split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: sdy.mesh @mesh = <["a"=4, "b"=2]>
#topology = #mpmd.topology<<"mesh1": <["a"=4, "b"=2]>>, <"mesh2": <["a"=4, "b"=2]>>>

// CHECK-LABEL: func.func public @main(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>, tensor<5xf32>) attributes {topology = #mpmd.topology<<"mesh1" : <["a"=4, "b"=2]>>, <"mesh2" : <["a"=4, "b"=2]>>>}
func.func public @main(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>, tensor<5xf32>) attributes {
  topology=#topology
}{
  %0 = mpmd.call @shardy_mpmd_f(%arg0, %arg1) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
  %1 = mpmd.call @shardy_mpmd_g(%arg0, %arg1) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
  %2 = call @shardy_mpmd_i(%arg0, %arg1) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
  return %0, %1, %2 : tensor<5xf32>, tensor<5xf32>, tensor<5xf32>
}

// CHECK-LABEL: func.func private @shardy_mpmd_f(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<5xf32> attributes {topology = #mpmd.topology<<"mesh1" : <["a"=4, "b"=2]>>, <"mesh2" : <["a"=4, "b"=2]>>>}
func.func @shardy_mpmd_f(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<5xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<5xf32>
  return %0 : tensor<5xf32>
}

// CHECK-LABEL: func.func private @shardy_mpmd_g(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<5xf32> attributes {topology = #mpmd.topology<<"mesh1" : <["a"=4, "b"=2]>>, <"mesh2" : <["a"=4, "b"=2]>>>}
func.func private @shardy_mpmd_g(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<5xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<5xf32>
  return %0 : tensor<5xf32>
}

// Not referred by a call_op, so it doesn't get a topology or have visibility changed.
// CHECK-LABEL: func.func @shardy_mpmd_h(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<5xf32> {
func.func @shardy_mpmd_h(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<5xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<5xf32>
  return %0 : tensor<5xf32>
}

// Referred by a func.call_op, so it doesn't get a topology or have visibility changed.
// CHECK-LABEL: func.func @shardy_mpmd_i(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<5xf32> {
func.func @shardy_mpmd_i(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<5xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<5xf32>
  return %0 : tensor<5xf32>
}

// -----

// Verify that the homogeneous topology check ignores axes of size 1.
// CHECK-LABEL: func.func public @main(%arg0: tensor<8xf32>) -> tensor<8xf32>
func.func public @main(%arg0: tensor<8xf32>) -> tensor<8xf32> attributes {
  topology=#mpmd.topology<
    <"mesh1": <["a"=4, "b"=2]>>,
    <"mesh2": <["x"=1, "a"=4, "b"=2]>>,
    <"mesh3": <["a"=4, "b"=2, "y"=1]>>,
    <"mesh4": <["a"=4, "z"=1, "b"=2]>>
  >
}{
  return %arg0 : tensor<8xf32>
}
