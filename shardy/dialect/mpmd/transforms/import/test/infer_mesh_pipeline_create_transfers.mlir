// RUN: mpmd_opt %s -mpmd-infer-mesh-pipeline='infer-transfers=true' 2>&1 | FileCheck %s

!m1_8x16 = !mpmd.mesh_tensor<"m1", tensor<8x16xf32>>
!m2_8x16 = !mpmd.mesh_tensor<"m2", tensor<8x16xf32>>
!m3_8x16 = !mpmd.mesh_tensor<"m3", tensor<8x16xf32>>

// CHECK-LABEL: func @assign_outputs_with_empty_src_set(
func.func @assign_outputs_with_empty_src_set(%arg0: !m3_8x16, %arg1: !m2_8x16)
  -> (tensor<8x16xf32>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>,
      <"m3": <["z"=2]>>
    >} {
// CHECK:       %[[T0:.*]] = mpmd.transfer %arg0 {{.*}}m3{{.*}} -> {{.*}}m2
// CHECK-NEXT:  %[[ADD:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[T0]], %arg1)
// CHECK:       return %[[ADD]]
  %4 = mpmd.unassign %arg0 : (!m3_8x16) -> tensor<8x16xf32>
  %5 = mpmd.unassign %arg1 : (!m2_8x16) -> tensor<8x16xf32>

  %6 = stablehlo.add %4, %5 : tensor<8x16xf32>

  // The return value will have empty src_set. If
  // `infer-transfers = false`, we would have an error here. But instead, we
  // assign it to the first mesh and introduce transfers. Note that this isn't
  // optimal, but inferring the right mesh is hard. Until we improve the
  // inference algorithm, users will need to adjust their assignments manually
  // if they want optimal results.
  func.return %6 : tensor<8x16xf32>
}

// CHECK-LABEL: func @create_transfer_on_intermediates(
// CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m1", tensor<8x16xf32>>,
// CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m2", tensor<8x16xf32>>)
func.func @create_transfer_on_intermediates(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>)
  -> (tensor<8x16xf32>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// Note in this test we don't add the inputs directly, as transfers are always
// generated on inputs. What we want to check is that that transfers are created
// on intermediate values.
// CHECK-NEXT:  %[[ADD_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
// CHECK:       %[[ADD_2:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%arg1)
// CHECK:       %[[TRANSFER:.*]] = mpmd.transfer %[[ADD_2]] {{.*}}m2{{.*}} -> {{.*}}m1
// CHECK-NEXT:  %[[ADD_3:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ADD_1]], %[[TRANSFER]])
// CHECK:       return %[[ADD_3]]
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x16xf32>
  %1 = stablehlo.add %arg1, %arg1 : tensor<8x16xf32>


  %2 = mpmd.assign %0 : (tensor<8x16xf32>) -> !m1_8x16
  %3 = mpmd.assign %1 : (tensor<8x16xf32>) -> !m2_8x16

  %4 = mpmd.unassign %2 : (!m1_8x16) -> tensor<8x16xf32>
  %5 = mpmd.unassign %3 : (!m2_8x16) -> tensor<8x16xf32>

  %6 = stablehlo.add %4, %5 : tensor<8x16xf32>

  func.return %6 : tensor<8x16xf32>
}
