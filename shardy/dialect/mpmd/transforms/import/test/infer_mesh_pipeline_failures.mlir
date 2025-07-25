// RUN: mpmd_opt %s -mpmd-infer-mesh-pipeline -verify-diagnostics -split-input-file

!m1_8x16 = !mpmd.mesh_tensor<"m1", tensor<8x16xf32>>
!m2_8x16 = !mpmd.mesh_tensor<"m2", tensor<8x16xf32>>
#topology =#mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["y"=2]>>>

func.func @operands_with_conflicting_meshes_non_reduce(%arg0: !m1_8x16, %arg1: !m2_8x16)
  -> (tensor<8x16xf32>) attributes {topology=#topology} {
  %2 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg3: tensor<8x16xf32>) {
    mpmd.return %arg3 : tensor<8x16xf32>
  } : (!m1_8x16) -> !m1_8x16

  %3 = mpmd.fragment<mesh="m2", origin=[]> (%arg1) (%arg3: tensor<8x16xf32>) {
    mpmd.return %arg3 : tensor<8x16xf32>
  } : (!m2_8x16) -> !m2_8x16

  %4 = mpmd.unassign %2 : (!m1_8x16) -> tensor<8x16xf32>
  %5 = mpmd.unassign %3 : (!m2_8x16) -> tensor<8x16xf32>

  // expected-error @+1 {{Mesh assignment is not possible for op as its operands are on conflicting meshes}}
  %6 = stablehlo.divide %4, %5 : tensor<8x16xf32>

  func.return %6 : tensor<8x16xf32>
}

// -----

!m1_8x16 = !mpmd.mesh_tensor<"m1", tensor<8x16xf32>>
!m2_8x16 = !mpmd.mesh_tensor<"m2", tensor<8x16xf32>>
#topology =#mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["y"=2]>>>

func.func @operands_with_conflicting_meshes_reduce(%arg0: !m1_8x16, %arg1: !m2_8x16)
  -> (tensor<8x16xf32>) attributes {topology=#topology} {
  %2 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg3: tensor<8x16xf32>) {
    mpmd.return %arg3 : tensor<8x16xf32>
  } : (!m1_8x16) -> !m1_8x16

  %3 = mpmd.fragment<mesh="m2", origin=[]> (%arg1) (%arg3: tensor<8x16xf32>) {
    mpmd.return %arg3 : tensor<8x16xf32>
  } : (!m2_8x16) -> !m2_8x16

  %4 = mpmd.unassign %2 : (!m1_8x16) -> tensor<8x16xf32>
  %5 = mpmd.unassign %3 : (!m2_8x16) -> tensor<8x16xf32>

  // expected-error @+1 {{Mesh assignment is not possible for op as its operands are on conflicting meshes}}
  %6 = stablehlo.add %4, %5 : tensor<8x16xf32>

  func.return %6 : tensor<8x16xf32>
}

// -----

!m1_3x5 = !mpmd.mesh_tensor<"m1", tensor<3x5xf32>>
!m2_3x5 = !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>
#topology = #mpmd.topology<<"m1": <["x"=8]>>, <"m2": <["x"=8]>>>

func.func public @src_set_empty_flowing_into_broadcast(%arg0: tensor<3x5xf32>) -> (tensor<3x5xf32>)
  attributes {topology = #topology}
{
  %0 = stablehlo.add %arg0, %arg0 : tensor<3x5xf32>
  %1 = mpmd.assign %0 : (tensor<3x5xf32>) -> !m1_3x5
  %2 = mpmd.assign %0 : (tensor<3x5xf32>) -> !m2_3x5

  %11 = mpmd.unassign %1 : (!m1_3x5) -> tensor<3x5xf32>
  %22 = mpmd.unassign %2 : (!m2_3x5) -> tensor<3x5xf32>

  // %3 will have src_set empty
  // expected-error @+1 {{Mesh assignment is not possible for op as its operands are on conflicting meshes}}
  %3 = stablehlo.divide %11, %22 : tensor<3x5xf32>
  %b = mpmd.broadcast %3 : tensor<3x5xf32>
  %bb = stablehlo.add %b, %b : tensor<3x5xf32>
  return %b : tensor<3x5xf32>
}
