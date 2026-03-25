// RUN: mpmd_opt %s -mpmd-optimize-pipeline='sink-transfers=true' -split-input-file 2>&1 | FileCheck %s --check-prefix=CHECK-SINK
// RUN: mpmd_opt %s -mpmd-optimize-pipeline='sink-transfers=false' -split-input-file 2>&1 | FileCheck %s --check-prefix=CHECK-NO-SINK

#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<16xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<16xf32>>

// CHECK-LABEL: func.func @sink_transfer
func.func @sink_transfer(%arg0: !mesh_1_tensor) -> !mesh_2_tensor attributes {topology=#topology} {
  // CHECK-SINK:      mpmd.fragment
  // CHECK-SINK:      mpmd.transfer
  // CHECK-SINK:      return

  // CHECK-NO-SINK:      mpmd.transfer
  // CHECK-NO-SINK:      mpmd.fragment
  // CHECK-NO-SINK:      return

  // Note: stablehlo.add on mesh_tensor might not support it directly without fragment.
  // Let's just use a TransferOp on argument directly to simplify, or wrap in fragment.
  %0 = mpmd.transfer %arg0 : (!mesh_1_tensor) -> !mesh_2_tensor

  %2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%a: tensor<16xf32>) {
    %mul = stablehlo.multiply %a, %a : tensor<16xf32>
    mpmd.return %mul : tensor<16xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor

  return %0 : !mesh_2_tensor
}
