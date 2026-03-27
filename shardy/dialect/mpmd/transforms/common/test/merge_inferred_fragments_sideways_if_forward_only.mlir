// RUN: mpmd_opt %s -mpmd-merge-inferred-fragments='merge-sideways-if-forward-only=true' 2>&1 | FileCheck %s

!m1_4x8 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!m2_4x8 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

#topo = #mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>

// Forward-only program: all origins have transpose_count=0.
// Sideways merging should be enabled.
// CHECK-LABEL: func @forward_only_merges_sideways
func.func @forward_only_merges_sideways(%arg0: !m1_4x8, %arg1: !m1_4x8)
  -> (!m1_4x8, !m1_4x8) attributes {topology=#topo} {
  // CHECK-NEXT: %[[FRAG:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f"]>
  // CHECK-NEXT:   stablehlo.add
  // CHECK-NEXT:   stablehlo.multiply
  // CHECK-NEXT:   mpmd.return
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[FRAG]]#0, %[[FRAG]]#1
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %arg0)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!m1_4x8, !m1_4x8) -> !m1_4x8
  %1 = mpmd.fragment<mesh="m1", origin=["f"(0)]> (%arg1)
    (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!m1_4x8) -> !m1_4x8

  func.return %0, %1 : !m1_4x8, !m1_4x8
}

// Non-forward-only program: one origin has transpose_count=1.
// Sideways merging should NOT be enabled.
// CHECK-LABEL: func @non_forward_only_no_sideways_merge
func.func @non_forward_only_no_sideways_merge(%arg0: !m1_4x8, %arg1: !m1_4x8)
  -> (!m1_4x8, !m1_4x8) attributes {topology=#topo} {
  // CHECK-NEXT: %[[FRAG0:.*]] = mpmd.fragment<mesh="m1", origin=[]>
  // CHECK-NEXT:   stablehlo.add
  // CHECK-NEXT:   mpmd.return
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[FRAG1:.*]] = mpmd.fragment<mesh="m1", origin=["f"(1)]>
  // CHECK-NEXT:   stablehlo.multiply
  // CHECK-NEXT:   mpmd.return
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[FRAG0]], %[[FRAG1]]
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %arg0)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!m1_4x8, !m1_4x8) -> !m1_4x8
  %1 = mpmd.fragment<mesh="m1", origin=["f"(1)]> (%arg1)
    (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!m1_4x8) -> !m1_4x8

  func.return %0, %1 : !m1_4x8, !m1_4x8
}
