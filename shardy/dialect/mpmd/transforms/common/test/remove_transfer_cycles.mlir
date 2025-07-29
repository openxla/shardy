// RUN: mpmd_opt %s -mpmd-remove-transfer-cycles 2>&1 | FileCheck %s

// Reshards out of Call Op are merged.

!m1_4x8 = !mpmd.mesh_tensor<"mesh1", tensor<4x8xf32>>
!m2_4x8 = !mpmd.mesh_tensor<"mesh2", tensor<4x8xf32>, memory_kind="device">
!m2_4x8_host = !mpmd.mesh_tensor<"mesh2", tensor<4x8xf32>, memory_kind="pinned_host">
!m3_4x8 = !mpmd.mesh_tensor<"mesh3", tensor<4x8xf32>>

#topo = #mpmd.topology<<"mesh1" : <["x"=4]>>, <"mesh2" : <["x"=4]>>, <"mesh3" : <["x"=4]>>>

// CHECK: func.func public @single_cycle_is_removed
func.func public @single_cycle_is_removed(%arg0: !m1_4x8)
  -> (!m1_4x8, !m3_4x8) attributes {topology = #topo} {
// CHECK-NEXT: %[[T2:.*]] = mpmd.transfer %arg0
// CHECK-NEXT: %[[T3:.*]] = mpmd.transfer %[[T2]]
// CHECK-NEXT: return %arg0, %[[T3]]

  %t2 = mpmd.transfer %arg0 : (!m1_4x8) -> !m2_4x8
  %t3 = mpmd.transfer %t2 : (!m2_4x8) -> !m3_4x8

  %t1_1 = mpmd.transfer %t3 : (!m3_4x8) -> !m1_4x8

  return %t1_1, %t3 : !m1_4x8, !m3_4x8
}

// CHECK: func.func public @triple_cycle_is_removed
func.func public @triple_cycle_is_removed(%arg0: !m1_4x8)
  -> (!m1_4x8, !m2_4x8, !m3_4x8) attributes {topology = #topo} {
// CHECK-NEXT: %[[T2:.*]] = mpmd.transfer %arg0
// CHECK-NEXT: %[[T3:.*]] = mpmd.transfer %[[T2]]
// CHECK-NEXT: return %arg0, %[[T2]], %[[T3]]

  %t2 = mpmd.transfer %arg0 : (!m1_4x8) -> !m2_4x8
  %t3 = mpmd.transfer %t2 : (!m2_4x8) -> !m3_4x8

  %t1_1 = mpmd.transfer %t3 : (!m3_4x8) -> !m1_4x8
  %t2_1 = mpmd.transfer %t1_1 : (!m1_4x8) -> !m2_4x8
  %t3_1 = mpmd.transfer %t2_1 : (!m2_4x8) -> !m3_4x8

  %t1_2 = mpmd.transfer %t3_1 : (!m3_4x8) -> !m1_4x8
  %t2_2 = mpmd.transfer %t1_2 : (!m1_4x8) -> !m2_4x8
  %t3_2 = mpmd.transfer %t2_2 : (!m2_4x8) -> !m3_4x8

  %t1_3 = mpmd.transfer %t3_2 : (!m3_4x8) -> !m1_4x8

  return %t1_3, %t2_2, %t3_2 : !m1_4x8, !m2_4x8, !m3_4x8
}

// CHECK: func.func public @cycle_with_pinned_host_is_kept
func.func public @cycle_with_pinned_host_is_kept(%arg0: !m1_4x8)
  -> (!m1_4x8, !m3_4x8) attributes {topology = #topo} {
// CHECK-NEXT: %[[T2:.*]] = mpmd.transfer %arg0
// CHECK-NEXT: %[[T3:.*]] = mpmd.transfer %[[T2]]
// CHECK-NEXT: %[[T1:.*]] = mpmd.transfer %[[T3]]
// CHECK-NEXT: return %[[T1]], %[[T3]]

  %t2 = mpmd.transfer %arg0 : (!m1_4x8) -> !m2_4x8_host
  %t3 = mpmd.transfer %t2 : (!m2_4x8_host) -> !m3_4x8

  %t1_1 = mpmd.transfer %t3 : (!m3_4x8) -> !m1_4x8

  return %t1_1, %t3 : !m1_4x8, !m3_4x8
}
