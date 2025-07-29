// RUN: mpmd_opt %s -mpmd-merge-inferred-fragments='merge-sideways=true' 2>&1 | FileCheck %s

!m1_4x8 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!m2_4x8 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

#topo = #mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>

// CHECK-LABEL: func @simple
func.func @simple(%arg0: !m1_4x8, %arg1: !m1_4x8)
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
  %1 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg1)
    (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!m1_4x8) -> !m1_4x8

  func.return %0, %1 : !m1_4x8, !m1_4x8
}

// CHECK-LABEL: func @interleaved
func.func @interleaved(%arg0: !m1_4x8, %arg1: !m1_4x8)
  -> (!m1_4x8, !m1_4x8) attributes {topology=#topo} {
  // CHECK-NEXT: %[[FRAG_F:.*]] = mpmd.fragment<mesh="m1", origin=["f"]>
  // CHECK-NEXT:   stablehlo.add
  // CHECK-NEXT:   mpmd.return
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[FRAG_G:.*]]:2 = mpmd.fragment<mesh="m1", origin=["g"]>
  // CHECK-NEXT:   stablehlo.multiply
  // CHECK-NEXT:   stablehlo.subtract
  // CHECK-NEXT:   mpmd.return
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[FRAG_G]]#1, %[[FRAG_G]]#0
  //
  // We have interleaved fragments which don't rely on one another:
  // %arg0 -> %0 -> %2
  // %arg1 -> %1
  //
  // With sideways merging, %1 merges into %2.

  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!m1_4x8) -> !m1_4x8

  %1 = mpmd.fragment<mesh="m1", origin=[]> (%arg1)
    (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!m1_4x8) -> !m1_4x8


  %2 = mpmd.fragment<mesh="m1", origin=["g"]> (%0)
    (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.subtract %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!m1_4x8) -> !m1_4x8

  func.return %2, %1 : !m1_4x8, !m1_4x8
}
