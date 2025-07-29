// RUN: mpmd_opt %s -mpmd-infer-mesh-pipeline='infer-transfers=true infer-cross-mesh-reductions=True' 2>&1 | FileCheck %s

!mesh_1_tensor_ui32 = !mpmd.mesh_tensor<"m1", tensor<ui32>>
!mesh_1_tensor_1_ui32 = !mpmd.mesh_tensor<"m1", tensor<1xui32>>
!mesh_1_tensor_2_ui32 = !mpmd.mesh_tensor<"m1", tensor<2xui32>>
!mesh_1_tensor_5_5_ui32 = !mpmd.mesh_tensor<"m1", tensor<5x5xui32>>
!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_1_tensor_4_1_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x1x8xf32>>
!mesh_1_tensor_8_16_f32 = !mpmd.mesh_tensor<"m1", tensor<8x16xf32>>
!mesh_1_tensor_4_16_f32 = !mpmd.mesh_tensor<"m1", tensor<4x16xf32>>
!mesh_1_tensor_16_8_f32 = !mpmd.mesh_tensor<"m1", tensor<16x8xf32>>

!mesh_2_tensor_ui32 = !mpmd.mesh_tensor<"m2", tensor<ui32>>
!mesh_2_tensor_1_ui32 = !mpmd.mesh_tensor<"m2", tensor<1xui32>>
!mesh_2_tensor_2_ui32 = !mpmd.mesh_tensor<"m2", tensor<2xui32>>
!mesh_2_tensor_5_5_ui32 = !mpmd.mesh_tensor<"m2", tensor<5x5xui32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
!mesh_2_tensor_4_1_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x1x8xf32>>
!mesh_2_tensor_4_16_f32 = !mpmd.mesh_tensor<"m2", tensor<4x16xf32>>
!mesh_2_tensor_8_16_f32 = !mpmd.mesh_tensor<"m2", tensor<8x16xf32>>
!mesh_2_tensor_16_8_f32 = !mpmd.mesh_tensor<"m2", tensor<16x8xf32>>

#topology =#mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["y"=2]>>>


// CHECK-LABEL: func @plain_spmd_module(
// CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:    %arg2: !mpmd.mesh_tensor<"m1"
func.func @plain_spmd_module(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<4x8xf32>)
  -> (tensor<4x8xf32>, tensor<4x16xf32>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[INFERRED_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %arg2)
// CHECK-SAME:    (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg3, %arg4
// CHECK-NEXT:    %[[MUL:.*]] = stablehlo.multiply %[[ADD]], %[[ADD]]
// CHECK-NEXT:    mpmd.return %[[MUL]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[INFERRED_2:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[INFERRED_1]], %arg1)
// CHECK:       return %[[INFERRED_1]], %[[INFERRED_2]] :
// CHECK-SAME:    !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
// CHECK-SAME:    !mpmd.mesh_tensor<"m1", tensor<4x16xf32>>
  %0 = stablehlo.add %arg0, %arg2 : tensor<4x8xf32>
  %1 = stablehlo.multiply %0, %0 : tensor<4x8xf32>

  %3 = "stablehlo.dot"(%1, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  func.return %1, %3 : tensor<4x8xf32>, tensor<4x16xf32>
}

// CHECK-LABEL: func @push_unassign_forward_simple(
// CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
// CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m1", tensor<8x16xf32>>,
// CHECK-SAME:    %arg2: !mpmd.mesh_tensor<"m1", tensor<4x16xf32>>)
func.func @push_unassign_forward_simple(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<4x16xf32>)
  -> (tensor<4x16xf32>, tensor<4x16xf32>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[FRAGMENT_1:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1)
// CHECK:       %[[INFERRED_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg2, %[[FRAGMENT_1]])
// CHECK-SAME:    (%arg3: tensor<4x16xf32>, %arg4: tensor<4x16xf32>) {
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg3, %arg4
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[INFERRED_2:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[INFERRED_1]])
// CHECK-SAME:    (%arg3: tensor<4x16xf32>) {
// CHECK-NEXT:    %[[MUL:.*]] = stablehlo.multiply %arg3, %arg3
// CHECK-NEXT:    mpmd.return %[[MUL]]
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[INFERRED_1]], %[[INFERRED_2]] :
// CHECK-SAME:    !mpmd.mesh_tensor<"m1", tensor<4x16xf32>>,
// CHECK-SAME:    !mpmd.mesh_tensor<"m1", tensor<4x16xf32>>
  %0 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %1 = mpmd.assign %arg1 : (tensor<8x16xf32>) -> !mesh_1_tensor_8_16_f32

  %2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%0, %1)
    (%arg3: tensor<4x8xf32>, %arg4: tensor<8x16xf32>) {
    %6 = "stablehlo.dot"(%arg3, %arg4) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    mpmd.return %6 : tensor<4x16xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_8_16_f32) -> !mesh_1_tensor_4_16_f32

  %3 = mpmd.unassign %2 : (!mesh_1_tensor_4_16_f32) -> tensor<4x16xf32>

  %4 = stablehlo.add %arg2, %3 : tensor<4x16xf32>
  %5 = stablehlo.multiply %4, %4 : tensor<4x16xf32>

  func.return %4, %5 : tensor<4x16xf32>, tensor<4x16xf32>
}

// CHECK-LABEL: func @push_assign_backwards_simple(
// CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
// CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m1", tensor<8x16xf32>>,
// CHECK-SAME:    %arg2: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
func.func @push_assign_backwards_simple(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<4x8xf32>)
  -> (tensor<4x8xf32>, tensor<4x16xf32>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[INFERRED_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %arg2)
// CHECK-SAME:    (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg3, %arg4
// CHECK-NEXT:    %[[MUL:.*]] = stablehlo.multiply %[[ADD]], %[[ADD]]
// CHECK-NEXT:    mpmd.return %[[MUL]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%[[INFERRED_1]], %arg1)
// CHECK-SAME:    (%arg3: tensor<4x8xf32>, %arg4: tensor<8x16xf32>) {
// CHECK:       return %[[INFERRED_1]], %[[FRAGMENT]] :
// CHECK-SAME:    !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
// CHECK-SAME:    !mpmd.mesh_tensor<"m1", tensor<4x16xf32>>
  %0 = stablehlo.add %arg0, %arg2 : tensor<4x8xf32>
  %1 = stablehlo.multiply %0, %0 : tensor<4x8xf32>

  %2 = mpmd.assign %1 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %3 = mpmd.assign %arg1 : (tensor<8x16xf32>) -> !mesh_1_tensor_8_16_f32

  %4 = mpmd.fragment<mesh="m1", origin=["f1"]> (%2, %3)
    (%arg3: tensor<4x8xf32>, %arg4: tensor<8x16xf32>) {
    %6 = "stablehlo.dot"(%arg3, %arg4) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    mpmd.return %6 : tensor<4x16xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_8_16_f32) -> !mesh_1_tensor_4_16_f32

  %5 = mpmd.unassign %4 : (!mesh_1_tensor_4_16_f32) -> tensor<4x16xf32>

  func.return %1, %5 : tensor<4x8xf32>, tensor<4x16xf32>
}

// CHECK-LABEL: func @unused_and_identity_fragments(
// CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>,
// CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>,
// CHECK-SAME:    %arg2: !mpmd.mesh_tensor<"m1", tensor<8x16xf32>>)
func.func @unused_and_identity_fragments(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<8x16xf32>)
  -> (tensor<4x8xf32>, tensor<8x16xf32>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[FRAGMENT:.*]] = mpmd.fragment<mesh="m2", origin=["f"]> (%arg0)
// CHECK:       return %[[FRAGMENT]], %arg2
// CHECK-SAME:    !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>,
// CHECK-SAME:    !mpmd.mesh_tensor<"m1", tensor<8x16xf32>>
  %0 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m2", origin=["f"]> (%0)
    (%arg3: tensor<4x8xf32>) {
    mpmd.return %arg3 : tensor<4x8xf32>
  } : (!mesh_2_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32

  %2 = mpmd.unassign %1 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>

  func.return %2, %arg2 : tensor<4x8xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @assign_ops_unused_by_user_fragment(
// CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>,
// CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m2", tensor<8x16xf32>>,
// CHECK-SAME:    %arg2: !mpmd.mesh_tensor<"m1", tensor<4x16xf32>>)
func.func @assign_ops_unused_by_user_fragment(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<4x16xf32>)
  -> (tensor<4x16xf32>, tensor<4x16xf32>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[FRAGMENT_1:.*]] = mpmd.fragment<mesh="m2", origin=["f1"]> (%arg0, %arg1)
// CHECK:       %[[INFERRED_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg2)
// CHECK-SAME:    (%arg3: tensor<4x16xf32>) {
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg3, %arg3
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[INFERRED_1]], %[[FRAGMENT_1]] :
// CHECK-SAME:    !mpmd.mesh_tensor<"m1", tensor<4x16xf32>>,
// CHECK-SAME:    !mpmd.mesh_tensor<"m2", tensor<4x16xf32>>
  %0 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32
  %1 = mpmd.assign %arg1 : (tensor<8x16xf32>) -> !mesh_2_tensor_8_16_f32

  %2 = mpmd.fragment<mesh="m2", origin=["f1"]> (%0, %1)
    (%arg3: tensor<4x8xf32>, %arg4: tensor<8x16xf32>) {
    %6 = "stablehlo.dot"(%arg3, %arg4) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    mpmd.return %6 : tensor<4x16xf32>
  } : (!mesh_2_tensor_4_8_f32, !mesh_2_tensor_8_16_f32) -> !mesh_2_tensor_4_16_f32

  %3 = mpmd.unassign %2 : (!mesh_2_tensor_4_16_f32) -> tensor<4x16xf32>

  %4 = stablehlo.add %arg2, %arg2 : tensor<4x16xf32>

  func.return %4, %3 : tensor<4x16xf32>, tensor<4x16xf32>
}

// CHECK-LABEL: func @assign_of_ops_with_only_scalar_operands(
// CHECK-SAME:    %arg0: !mpmd.mesh_tensor<"m1", tensor<ui32>>)
func.func @assign_of_ops_with_only_scalar_operands(%arg0: tensor<ui32>)
  -> (tensor<ui32>, tensor<5x5xui32>, !mesh_1_tensor_5_5_ui32, !mesh_2_tensor_5_5_ui32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[INFERRED_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
// CHECK-SAME:    (%arg1: tensor<ui32>) {
// CHECK-NEXT:    %[[CONST_1:.*]] = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:    %[[ADD_1:.*]] = stablehlo.add %arg1, %[[CONST_1]]
// CHECK-NEXT:    mpmd.return %[[ADD_1]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[TRANSFER:.*]] = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<ui32>>) -> !mpmd.mesh_tensor<"m2", tensor<ui32>>
// CHECK-NEXT:  %[[INFERRED_2:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[TRANSFER]])
// CHECK-SAME:    (%arg1: tensor<ui32>) {
// CHECK-NEXT:    %[[CONST_2:.*]] = stablehlo.constant dense<1> : tensor<ui32>
// CHECK-NEXT:    %[[ADD_2:.*]] = stablehlo.add %arg1, %[[CONST_2]]
// CHECK-NEXT:    mpmd.return %[[ADD_2]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[INFERRED_3:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[INFERRED_1]])
// CHECK-SAME:    (%arg1: tensor<ui32>) {
// CHECK-NEXT:    %[[BROADCAST_1:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<5x5xui32>
// CHECK-NEXT:    mpmd.return %[[BROADCAST_1]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[INFERRED_4:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[INFERRED_2]])
// CHECK-SAME:    (%arg1: tensor<ui32>) {
// CHECK-NEXT:    %[[BROADCAST_2:.*]] = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<ui32>) -> tensor<5x5xui32>
// CHECK-NEXT:    mpmd.return %[[BROADCAST_2]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[INFERRED_5:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[INFERRED_4]])
// CHECK-SAME:    (%arg1: tensor<5x5xui32>) {
// CHECK-NEXT:    %[[ADD_3:.*]] = stablehlo.add %arg1, %arg1
// CHECK-NEXT:    mpmd.return %[[ADD_3]]
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[INFERRED_1]], %[[INFERRED_3]], %[[INFERRED_3]], %[[INFERRED_5]] :
// CHECK-SAME:    !mpmd.mesh_tensor<"m1", tensor<ui32>>,
// CHECK-SAME:    !mpmd.mesh_tensor<"m1", tensor<5x5xui32>>,
// CHECK-SAME:    !mpmd.mesh_tensor<"m1", tensor<5x5xui32>>,
// CHECK-SAME:    !mpmd.mesh_tensor<"m2", tensor<5x5xui32>>
  %0 = stablehlo.constant dense<1> : tensor<ui32>
  %1 = stablehlo.add %arg0, %0 : tensor<ui32>
  %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<ui32>) -> tensor<5x5xui32>

  %3 = mpmd.assign %2 : (tensor<5x5xui32>) -> !mesh_1_tensor_5_5_ui32

  %4 = stablehlo.add %2, %2 : tensor<5x5xui32>

  %5 = mpmd.assign %4 : (tensor<5x5xui32>) -> !mesh_2_tensor_5_5_ui32

  func.return %1, %2, %3, %5 : tensor<ui32>, tensor<5x5xui32>, !mesh_1_tensor_5_5_ui32, !mesh_2_tensor_5_5_ui32
}

// CHECK-LABEL: func @assign_of_non_scalar_const
func.func @assign_of_non_scalar_const()
  -> (!mesh_1_tensor_5_5_ui32, !mesh_2_tensor_5_5_ui32, !mesh_1_tensor_5_5_ui32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[INFERRED_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> () () {
// CHECK-NEXT:    %[[CONST_1:.*]] = stablehlo.constant dense<1> : tensor<5x5xui32>
// CHECK-NEXT:    mpmd.return %[[CONST_1]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[INFERRED_2:.*]] = mpmd.fragment<mesh="m2", origin=[]> () () {
// CHECK-NEXT:    %[[CONST_2:.*]] = stablehlo.constant dense<1> : tensor<5x5xui32>
// CHECK-NEXT:    mpmd.return %[[CONST_2]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[INFERRED_3:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[INFERRED_1]])
// CHECK-SAME:    (%arg0: tensor<5x5xui32>) {
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg0, %arg0
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK:       return %[[INFERRED_1]], %[[INFERRED_2]], %[[INFERRED_3]] :
// CHECK-SAME:    !mpmd.mesh_tensor<"m1", tensor<5x5xui32>>,
// CHECK-SAME:    !mpmd.mesh_tensor<"m2", tensor<5x5xui32>>,
// CHECK-SAME:    !mpmd.mesh_tensor<"m1", tensor<5x5xui32>>
  %0 = stablehlo.constant dense<1> : tensor<5x5xui32>

  %1 = mpmd.assign %0 : (tensor<5x5xui32>) -> !mesh_1_tensor_5_5_ui32
  %2 = mpmd.assign %0 : (tensor<5x5xui32>) -> !mesh_2_tensor_5_5_ui32

  %3 = stablehlo.add %0, %0 : tensor<5x5xui32>
  %4 = mpmd.assign %3 : (tensor<5x5xui32>) -> !mesh_1_tensor_5_5_ui32

  func.return %1, %2, %4 : !mesh_1_tensor_5_5_ui32, !mesh_2_tensor_5_5_ui32, !mesh_1_tensor_5_5_ui32
}

// CHECK-LABEL: func @push_unassign_forward_then_assign_backwards
func.func @push_unassign_forward_then_assign_backwards(%arg0: tensor<4x16xf32>, %arg1: tensor<4x16xf32>)
  -> (tensor<4x16xf32>, tensor<4x16xf32>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[FRAGMENT_1:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
// CHECK:       %[[INFERRED_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg1)
// CHECK-SAME:    (%arg2: tensor<4x16xf32>) {
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg2
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[INFERRED_2:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[FRAGMENT_1]], %[[INFERRED_1]])
// CHECK-SAME:    (%arg2: tensor<4x16xf32>, %arg3: tensor<4x16xf32>) {
// CHECK-NEXT:    %[[MUL:.*]] = stablehlo.multiply %arg2, %arg3
// CHECK-NEXT:    mpmd.return %[[MUL]]
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[INFERRED_1]], %[[INFERRED_2]]
  %0 = mpmd.assign %arg0 : (tensor<4x16xf32>) -> !mesh_1_tensor_4_16_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f1"]> (%0)
    (%arg2: tensor<4x16xf32>) {
    mpmd.return %arg2 : tensor<4x16xf32>
  } : (!mesh_1_tensor_4_16_f32) -> !mesh_1_tensor_4_16_f32

  %2 = mpmd.unassign %1 : (!mesh_1_tensor_4_16_f32) -> tensor<4x16xf32>

  %3 = stablehlo.add %arg1, %arg1 : tensor<4x16xf32>
  %4 = stablehlo.multiply %2, %3 : tensor<4x16xf32>

  func.return %3, %4 : tensor<4x16xf32>, tensor<4x16xf32>
}

// CHECK-LABEL: func @push_assign_backwards_then_unassign_forward
func.func @push_assign_backwards_then_unassign_forward(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<4x8xf32>)
  -> (tensor<4x8xf32>, tensor<4x16xf32>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[INFERRED_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %arg2)
// CHECK-SAME:    (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg3, %arg4
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[INFERRED_2:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[INFERRED_1]])
// CHECK-SAME:    (%arg3: tensor<4x8xf32>) {
// CHECK-NEXT:    %[[MUL:.*]] = stablehlo.multiply %arg3, %arg3
// CHECK-NEXT:    mpmd.return %[[MUL]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[FRAGMENT_1:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%[[INFERRED_1]], %arg1)
// CHECK:       return %[[INFERRED_2]], %[[FRAGMENT_1]]
  %0 = stablehlo.add %arg0, %arg2 : tensor<4x8xf32>
  %1 = stablehlo.multiply %0, %0 : tensor<4x8xf32>

  %2 = mpmd.assign %0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %3 = mpmd.assign %arg1 : (tensor<8x16xf32>) -> !mesh_1_tensor_8_16_f32

  %4 = mpmd.fragment<mesh="m1", origin=["f1"]> (%2, %3)
    (%arg3: tensor<4x8xf32>, %arg4: tensor<8x16xf32>) {
    %6 = "stablehlo.dot"(%arg3, %arg4) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    mpmd.return %6 : tensor<4x16xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_8_16_f32) -> !mesh_1_tensor_4_16_f32

  %5 = mpmd.unassign %4 : (!mesh_1_tensor_4_16_f32) -> tensor<4x16xf32>

  func.return %1, %5 : tensor<4x8xf32>, tensor<4x16xf32>
}

// CHECK-LABEL: func @op_between_unassign_and_assign
func.func @op_between_unassign_and_assign(%arg0: !mesh_1_tensor_4_8_f32, %arg1: tensor<4x8xf32>)
  -> !mesh_1_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[INFERRED_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %arg1)
// CHECK-SAME:    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:    %[[MUL:.*]] = stablehlo.multiply %[[ADD]], %[[ADD]]
// CHECK-NEXT:    mpmd.return %[[MUL]]
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[INFERRED_1]]
  %0 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>

  %1 = stablehlo.add %0, %arg1 : tensor<4x8xf32>
  %2 = stablehlo.multiply %1, %1 : tensor<4x8xf32>

  %3 = mpmd.assign %2 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32

  func.return %3 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @op_with_multiple_results
func.func @op_with_multiple_results(%arg0: tensor<4x16xf32>, %arg1: tensor<16x8xf32>)
  -> (tensor<4x16xf32>, tensor<16x8xf32>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[FRAGMENT_1:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
// CHECK:       %[[INFERRED:.*]]:2 = mpmd.fragment<mesh="m1", origin=[]> (%[[FRAGMENT_1]], %arg1)
// CHECK-SAME:    (%arg2: tensor<4x16xf32>, %arg3: tensor<16x8xf32>) {
// CHECK-NEXT:    %[[OPT_BARRIER:.*]]:2 = stablehlo.optimization_barrier %arg2, %arg3
// CHECK-NEXT:    mpmd.return %[[OPT_BARRIER]]#0, %[[OPT_BARRIER]]#1
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[INFERRED]]#0, %[[INFERRED]]#1
  %0 = mpmd.assign %arg0 : (tensor<4x16xf32>) -> !mesh_1_tensor_4_16_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f1"]> (%0)
    (%arg2: tensor<4x16xf32>) {
    mpmd.return %arg2 : tensor<4x16xf32>
  } : (!mesh_1_tensor_4_16_f32) -> !mesh_1_tensor_4_16_f32

  %2 = mpmd.unassign %1 : (!mesh_1_tensor_4_16_f32) -> tensor<4x16xf32>

  %3:2 = stablehlo.optimization_barrier %2, %arg1 : tensor<4x16xf32>, tensor<16x8xf32>

  func.return %3#0, %3#1 : tensor<4x16xf32>, tensor<16x8xf32>
}

// CHECK-LABEL: func @op_with_no_results
func.func @op_with_no_results(%arg0: tensor<4x16xf32>)
  -> (!mesh_2_tensor_4_16_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[INFERRED_1:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%arg0)
// CHECK-NEXT:    stablehlo.add %arg1, %arg1
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT:  }
// CHECK:       mpmd.fragment<mesh="m2", origin=[]> (%[[INFERRED_1]]) (%arg1
// CHECK-NEXT:    sdy.sharding_group %arg1
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[INFERRED_1]]
  %1 = stablehlo.add %arg0, %arg0 : tensor<4x16xf32>
  sdy.sharding_group %1 group_id=0 : tensor<4x16xf32>

  %2 = mpmd.assign %1 : (tensor<4x16xf32>) -> !mesh_2_tensor_4_16_f32
  func.return %2 : !mesh_2_tensor_4_16_f32
}

// CHECK-LABEL: func @op_with_no_results_multiple_meshes
func.func @op_with_no_results_multiple_meshes(%arg0: !mesh_2_tensor_4_16_f32)
  -> (!mesh_1_tensor_4_16_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[TRANSFER:.*]] = mpmd.transfer %arg0
// CHECK-NEXT:  %[[INFERRED_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[TRANSFER]])
// CHECK-NEXT:    stablehlo.add %arg1, %arg1
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[INFERRED_2:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%arg0)
// CHECK-NEXT:    stablehlo.add %arg1, %arg1
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT:  }
// CHECK-NEXT:  mpmd.fragment<mesh="m2", origin=[]> (%[[INFERRED_2]]) (%arg1
// CHECK-NEXT:    sdy.sharding_group %arg1
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT:  }
// CHECK-NEXT:  mpmd.fragment<mesh="m1", origin=[]> (%[[INFERRED_1]]) (%arg1
// CHECK-NEXT:    sdy.sharding_group %arg1
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[TRANSFER]]
  %0 = mpmd.unassign %arg0 : (!mesh_2_tensor_4_16_f32) -> tensor<4x16xf32>
  %t = mpmd.transfer %arg0 : (!mesh_2_tensor_4_16_f32) -> !mesh_1_tensor_4_16_f32

  %1 = stablehlo.add %0, %0 : tensor<4x16xf32>
  sdy.sharding_group %1 group_id=0 : tensor<4x16xf32>

  func.return %t : !mesh_1_tensor_4_16_f32
}

// CHECK-LABEL: func @push_unassign_forward_op_with_regions
func.func @push_unassign_forward_op_with_regions(%arg0: tensor<4x16xf32>, %arg1: tensor<4x16xf32>, %arg2: tensor<i32>)
  -> tensor<4x16xf32> attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[FRAGMENT_1:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0)
// CHECK:       %[[INFERRED:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg2, %[[FRAGMENT_1]], %arg1)
// CHECK-SAME:    (%arg3: tensor<i32>, %arg4: tensor<4x16xf32>, %arg5: tensor<4x16xf32>) {
// CHECK-NEXT:    %[[CASE:.*]] = "stablehlo.case"(%arg3) ({
// CHECK-NEXT:      %[[ADD:.*]] = stablehlo.add %arg4, %arg5
// CHECK-NEXT:      stablehlo.return %[[ADD]]
// CHECK-NEXT:    }, {
// CHECK-NEXT:      stablehlo.return %arg5
// CHECK-NEXT:    })
// CHECK-NEXT:    mpmd.return %[[CASE]]
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[INFERRED]]
  %0 = mpmd.assign %arg0 : (tensor<4x16xf32>) -> !mesh_1_tensor_4_16_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f1"]> (%0)
    (%arg3: tensor<4x16xf32>) {
    mpmd.return %arg3 : tensor<4x16xf32>
  } : (!mesh_1_tensor_4_16_f32) -> !mesh_1_tensor_4_16_f32

  %2 = mpmd.unassign %1 : (!mesh_1_tensor_4_16_f32) -> tensor<4x16xf32>

  %3 = "stablehlo.case"(%arg2) ({
    %4 = stablehlo.add %2, %arg1 : tensor<4x16xf32>
    stablehlo.return %4 : tensor<4x16xf32>
  }, {
    stablehlo.return %arg1 : tensor<4x16xf32>
  }) : (tensor<i32>) -> tensor<4x16xf32>

  func.return %3 : tensor<4x16xf32>
}

// CHECK-LABEL: func @push_assign_backwards_op_with_regions
func.func @push_assign_backwards_op_with_regions(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<i32>)
  -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[INFERRED:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1, %arg2)
// CHECK-SAME:    (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>, %arg5: tensor<i32>) {
// CHECK-NEXT:    %[[CASE:.*]] = "stablehlo.case"(%arg5) ({
// CHECK-NEXT:      %[[ADD:.*]] = stablehlo.add %arg3, %arg4
// CHECK-NEXT:      stablehlo.return %[[ADD]]
// CHECK-NEXT:    }, {
// CHECK-NEXT:      stablehlo.return %arg4
// CHECK-NEXT:    })
// CHECK-NEXT:    mpmd.return %[[CASE]]
// CHECK-NEXT:  }
// CHECK:       return %[[INFERRED]]
  %0 = "stablehlo.case"(%arg2) ({
    %4 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
    stablehlo.return %4 : tensor<4x8xf32>
  }, {
    stablehlo.return %arg1 : tensor<4x8xf32>
  }) : (tensor<i32>) -> tensor<4x8xf32>

  %1 = mpmd.assign %0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32

  %2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%1)
    (%arg3: tensor<4x8xf32>) {
    mpmd.return %arg3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %3 = mpmd.unassign %2 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>

  func.return %3 : tensor<4x8xf32>
}

// CHECK-LABEL: func @pushed_assign_of_unassign_deduped_with_transfer
func.func @pushed_assign_of_unassign_deduped_with_transfer(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_2_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, tensor<4x8xf32>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[TRANSFER:.*]] = mpmd.transfer %arg0
// CHECK-NEXT:  %[[INFERRED_1:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[TRANSFER]])
// CHECK-SAME:    (%arg1: tensor<4x8xf32>) {
// CHECK-NEXT:    %[[ADD_1:.*]] = stablehlo.add %arg1, %arg1
// CHECK-NEXT:    %[[ADD_2:.*]] = stablehlo.add %[[ADD_1]], %[[ADD_1]]
// CHECK-NEXT:    mpmd.return %[[ADD_2]]
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[TRANSFER]], %[[INFERRED_1]], %[[TRANSFER]]
  %0 = mpmd.transfer %arg0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  %1 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  %2 = stablehlo.add %1, %1 : tensor<4x8xf32>
  %3 = stablehlo.add %2, %2 : tensor<4x8xf32>
  %4 = mpmd.assign %3 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32
  func.return %0, %4, %1 : !mesh_2_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, tensor<4x8xf32>
}

// CHECK-LABEL: func @unassign_pushed_despite_transfer
func.func @unassign_pushed_despite_transfer(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_2_tensor_4_8_f32, tensor<4x8xf32>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[TRANSFER:.*]] = mpmd.transfer %arg0
// CHECK-NEXT:  %[[INFERRED:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
// CHECK-SAME:    (%arg1: tensor<4x8xf32>) {
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg1, %arg1
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[TRANSFER]], %[[INFERRED]]
  %0 = mpmd.transfer %arg0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32
  %1 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  %2 = stablehlo.add %1, %1 : tensor<4x8xf32>
  func.return %0, %2 : !mesh_2_tensor_4_8_f32, tensor<4x8xf32>
}

// CHECK-LABEL: func @multiple_meshes_complex
func.func @multiple_meshes_complex(%arg0: tensor<4x8xf32>,
                                   %arg1: tensor<8x16xf32>,
                                   %arg2: tensor<16x8xf32>,
                                   %arg3: tensor<4x16xf32>,
                                   %arg4: tensor<16x8xf32>)
  -> (tensor<4x16xf32>, tensor<4x8xf32>, tensor<16x8xf32>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[INFERRED_1:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%arg2, %arg4) (%arg5: tensor<16x8xf32>, %arg6: tensor<16x8xf32>) {
// CHECK-NEXT:    %[[ADD_1:.*]] = stablehlo.add %arg5, %arg6 : tensor<16x8xf32>
// CHECK-NEXT:    mpmd.return %[[ADD_1]] : tensor<16x8xf32>
// CHECK-NEXT:  } : (!mpmd.mesh_tensor<"m2", tensor<16x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<16x8xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<16x8xf32>>
// CHECK-NEXT:  %[[FRAGMENT_1:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1) (%arg5: tensor<4x8xf32>, %arg6: tensor<8x16xf32>) {
// CHECK-NEXT:    %[[DOT_1:.*]] = stablehlo.dot %arg5, %arg6 : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:    mpmd.return %[[DOT_1]] : tensor<4x16xf32>
// CHECK-NEXT:  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<8x16xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x16xf32>>
// CHECK-NEXT:  %[[TRANSFER:.*]] = mpmd.transfer %[[FRAGMENT_1]] : (!mpmd.mesh_tensor<"m1", tensor<4x16xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x16xf32>>
// CHECK-NEXT:  %[[FRAGMENT_2:.*]] = mpmd.fragment<mesh="m2", origin=["f2"]> (%[[TRANSFER]], %arg2) (%arg5: tensor<4x16xf32>, %arg6: tensor<16x8xf32>) {
// CHECK-NEXT:    %[[ADD_2:.*]] = stablehlo.add %arg6, %arg6 : tensor<16x8xf32>
// CHECK-NEXT:    %[[DOT_2:.*]] = stablehlo.dot %arg5, %[[ADD_2]] : (tensor<4x16xf32>, tensor<16x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:    mpmd.return %[[DOT_2]] : tensor<4x8xf32>
// CHECK-NEXT:  } : (!mpmd.mesh_tensor<"m2", tensor<4x16xf32>>, !mpmd.mesh_tensor<"m2", tensor<16x8xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
// CHECK-NEXT:  %[[INFERRED_2:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[FRAGMENT_1]], %arg3) (%arg5: tensor<4x16xf32>, %arg6: tensor<4x16xf32>) {
// CHECK-NEXT:    %[[ADD_3:.*]] = stablehlo.add %arg5, %arg6 : tensor<4x16xf32>
// CHECK-NEXT:    %[[ADD_4:.*]] = stablehlo.add %[[ADD_3]], %[[ADD_3]] : tensor<4x16xf32>
// CHECK-NEXT:    mpmd.return %[[ADD_4]] : tensor<4x16xf32>
// CHECK-NEXT:  } : (!mpmd.mesh_tensor<"m1", tensor<4x16xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x16xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x16xf32>>
// CHECK-NEXT:  return %[[INFERRED_2]], %[[FRAGMENT_2]], %[[INFERRED_1]] : !mpmd.mesh_tensor<"m1", tensor<4x16xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<16x8xf32>>
  %0 = stablehlo.add %arg2, %arg4 : tensor<16x8xf32>

  %1 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %2 = mpmd.assign %arg1 : (tensor<8x16xf32>) -> !mesh_1_tensor_8_16_f32

  %3 = mpmd.fragment<mesh="m1", origin=["f1"]> (%1, %2)
    (%arg5: tensor<4x8xf32>, %arg6: tensor<8x16xf32>) {
    %12 = "stablehlo.dot"(%arg5, %arg6) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    mpmd.return %12 : tensor<4x16xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_8_16_f32) -> (!mesh_1_tensor_4_16_f32)

  %4 = mpmd.transfer %3 : (!mesh_1_tensor_4_16_f32) -> !mesh_2_tensor_4_16_f32

  %5 = stablehlo.add %arg2, %arg2 : tensor<16x8xf32>

  %6 = mpmd.assign %5 : (tensor<16x8xf32>) -> !mesh_2_tensor_16_8_f32

  %7 = mpmd.fragment<mesh="m2", origin=["f2"]> (%4, %6)
    (%arg5: tensor<4x16xf32>, %arg6: tensor<16x8xf32>) {
    %12 = "stablehlo.dot"(%arg5, %arg6) : (tensor<4x16xf32>, tensor<16x8xf32>) -> tensor<4x8xf32>
    mpmd.return %12 : tensor<4x8xf32>
  } : (!mesh_2_tensor_4_16_f32, !mesh_2_tensor_16_8_f32) -> !mesh_2_tensor_4_8_f32

  %8 = mpmd.unassign %3 : (!mesh_1_tensor_4_16_f32) -> tensor<4x16xf32>

  %9 = stablehlo.add %8, %arg3 : tensor<4x16xf32>
  %10 = stablehlo.add %9, %9 : tensor<4x16xf32>

  %11 = mpmd.unassign %7 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>

  func.return %10, %11, %0 : tensor<4x16xf32>, tensor<4x8xf32>, tensor<16x8xf32>
}

// CHECK-LABEL: func @reduce_of_reduces(%arg0: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:                     %arg1: !mpmd.mesh_tensor<"m2"
func.func @reduce_of_reduces(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
  -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32) attributes {topology=#topology} {
// CHECK-NEXT:  %[[ADD_LOCAL_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %arg0
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[ADD_LOCAL_2:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%arg1, %arg1
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[TRANSFER_2:.*]] = mpmd.transfer %[[ADD_LOCAL_2]]
// CHECK-NEXT:  %[[ADD_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ADD_LOCAL_1]], %[[TRANSFER_2]])
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[TRANSFER_1:.*]] = mpmd.transfer %[[ADD_LOCAL_1]]
// CHECK-NEXT:  %[[ADD_2:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[TRANSFER_1]], %[[ADD_LOCAL_2]])
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[ADD_1]], %[[ADD_2]]
  %arg0_a = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %arg1_a = mpmd.assign %arg1 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32
  %1 = mpmd.unassign %arg0_a : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  %2 = mpmd.unassign %arg1_a : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>
  %41 = stablehlo.add %1, %2 : tensor<4x8xf32>
  %42 = stablehlo.add %1, %2 : tensor<4x8xf32>
  %5 = stablehlo.add %41, %42 : tensor<4x8xf32>
  %6 = mpmd.assign %5 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %7 = mpmd.assign %5 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32

  func.return %6, %7 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func @concat_reduce(%arg0: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m2"
func.func @concat_reduce(%arg0: tensor<4x1x8xf32>, %arg1: tensor<4x1x8xf32>)
  -> !mesh_1_tensor_4_8_f32 attributes {topology=#topology} {
// CHECK-NEXT:  %[[R1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0
// CHECK-NEXT:    stablehlo.reshape
// CHECK:       %[[R2:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%arg1
// CHECK-NEXT:    stablehlo.reshape
// CHECK:      %[[R3:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0
// CHECK-NEXT:    stablehlo.reshape
// CHECK:       %[[R4:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%arg1
// CHECK-NEXT:    stablehlo.reshape
// CHECK:       %[[MAX_LOCAL_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[R1]], %[[R3]]
// CHECK-NEXT:    %[[MAX:.*]] = stablehlo.maximum %arg2, %arg3
// CHECK-NEXT:    mpmd.return %[[MAX]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[MAX_LOCAL_2:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[R2]], %[[R4]]
// CHECK-NEXT:    %[[MAX:.*]] = stablehlo.maximum %arg2, %arg3
// CHECK-NEXT:    mpmd.return %[[MAX]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[TRANSFER_2:.*]] = mpmd.transfer %[[MAX_LOCAL_2]]
// CHECK-NEXT:  %[[MAX_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[MAX_LOCAL_1]], %[[TRANSFER_2]])
// CHECK-NEXT:    %[[MAX:.*]] = stablehlo.maximum %arg2, %arg3
// CHECK-NEXT:    mpmd.return %[[MAX]]
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[MAX_1]]
  %init = stablehlo.constant dense<1.0> : tensor<f32>
  %arg0_a = mpmd.assign %arg0 : (tensor<4x1x8xf32>) -> !mesh_1_tensor_4_1_8_f32
  %arg1_a = mpmd.assign %arg1 : (tensor<4x1x8xf32>) -> !mesh_2_tensor_4_1_8_f32
  %1 = mpmd.unassign %arg0_a : (!mesh_1_tensor_4_1_8_f32) -> tensor<4x1x8xf32>
  %2 = mpmd.unassign %arg1_a : (!mesh_2_tensor_4_1_8_f32) -> tensor<4x1x8xf32>
  %concat = "stablehlo.concatenate"(%1, %2, %1, %2) <{dimension = 1 : i64}> :
    (tensor<4x1x8xf32>, tensor<4x1x8xf32>, tensor<4x1x8xf32>, tensor<4x1x8xf32>) -> tensor<4x4x8xf32>
  %reduce = stablehlo.reduce(%concat init: %init) applies stablehlo.maximum across dimensions = [1] :
    (tensor<4x4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  %6 = mpmd.assign %reduce : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  func.return %6 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @src_set_empty_flowing_into_broadcast(%arg0: !mpmd.mesh_tensor<"m1"
func.func @src_set_empty_flowing_into_broadcast(%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>)
  attributes {topology = #topology}
{
// CHECK-NEXT:  %[[F1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0
// CHECK-NEXT:    stablehlo.add
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[T1:.*]] = mpmd.transfer %arg0
// CHECK-NEXT:  %[[F2:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[T1]])
// CHECK-NEXT:    stablehlo.add
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[T2:.*]] = mpmd.transfer %[[F2]]
// CHECK-NEXT:  %[[F3:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[F1]], %[[T2]])
// CHECK-NEXT:    stablehlo.divide
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[F3]]
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  %1 = mpmd.assign %0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %2 = mpmd.assign %0 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32

  %11 = mpmd.unassign %1 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  %22 = mpmd.unassign %2 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>

  // %3 will have src_set empty
  %3 = stablehlo.divide %11, %22 : tensor<4x8xf32>
  %b = mpmd.broadcast %3 : tensor<4x8xf32>
  %bb = stablehlo.add %b, %b : tensor<4x8xf32>
  return %b : tensor<4x8xf32>
}
