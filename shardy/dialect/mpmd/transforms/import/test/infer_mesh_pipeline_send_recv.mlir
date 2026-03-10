// RUN: mpmd_opt %s -mpmd-infer-mesh-pipeline='infer-transfers=true infer-cross-mesh-reductions=True' 2>&1 | FileCheck %s

// Test that stablehlo.send and stablehlo.create_token ops (which produce
// !stablehlo.token types) are correctly partitioned into fragments by the
// MPMD pipeline.

// CHECK-LABEL: func @send_with_token(
// CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
// CHECK:       mpmd.fragment<mesh="m1", origin=[]>
// CHECK:         stablehlo.add
// CHECK:         mpmd.return
// CHECK:       mpmd.fragment<mesh="m1", origin=[]>
// CHECK:         stablehlo.create_token
// CHECK:         mpmd.return
// CHECK:       mpmd.fragment<mesh="m1", origin=[]>
// CHECK:         stablehlo.send
// CHECK:         mpmd.return
func.func @send_with_token(%arg0: tensor<4x8xf32>)
  -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>

  %token = "stablehlo.create_token"() : () -> !stablehlo.token
  "stablehlo.send"(%0, %token) {
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>,
    is_host_transfer = true
  } : (tensor<4x8xf32>, !stablehlo.token) -> !stablehlo.token

  func.return %0 : tensor<4x8xf32>
}
