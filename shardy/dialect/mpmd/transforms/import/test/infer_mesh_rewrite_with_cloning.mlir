// RUN: mpmd_opt %s -mpmd-infer-mesh-rewrite-using-analysis='max-clones=3' 2>&1 | FileCheck %s

// CHECK-LABEL: func @wrap_as_used_by_return
func.func @wrap_as_used_by_return(%arg0: !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, %arg1: tensor<8x16xi32>)
  -> (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>)
  attributes {topology = #mpmd.topology<<"mesh1" : <["x"=4]>>>}
{
// CHECK-NEXT: %[[ASSIGN_ARG1:.*]] = mpmd.assign %arg1
// CHECK-NEXT: %[[INFERRED_0:.*]] = mpmd.fragment<mesh="mesh1", origin=[]> (%[[ASSIGN_ARG1]])
// CHECK-NEXT:   multiply
// CHECK-NEXT:   mpmd.return
// CHECK-NEXT: }
// CHECK-NEXT: %[[FRAG_UNASSIGN:.*]] = mpmd.unassign %[[INFERRED_0]]
// CHECK-NEXT: %[[FRAG_ASSIGN:.*]] = mpmd.assign %[[FRAG_UNASSIGN]]
// CHECK-NEXT: %[[FRAG:.*]] = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg0, %[[FRAG_ASSIGN]])
// CHECK-SAME:   (%arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>)
// CHECK-NEXT:     %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:     mpmd.return %[[ADD]]
// CHECK-NEXT: }
// CHECK-NEXT: %[[MULT_ASSIGN2:.*]] = mpmd.assign %[[FRAG_UNASSIGN]]
// CHECK-NEXT: return %[[FRAG]], %[[MULT_ASSIGN2]]

  // The multply op cannot be cloned into the fragment as it is used by an op
  // that is not a fragment, i.e. the return op.
  %m = stablehlo.multiply %arg1, %arg1 : tensor<8x16xi32>
  %am1 = mpmd.assign %m : (tensor<8x16xi32>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %1 = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg0, %am1) (%arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<8x16xi32>
    mpmd.return %2 : tensor<8x16xi32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %am2 = mpmd.assign %m : (tensor<8x16xi32>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  return %1, %am2 : !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
}

// CHECK-LABEL: @many_fragment_users_within_limit
func.func @many_fragment_users_within_limit(%arg0: tensor<8x16xi32>, %arg1: !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>)
  -> (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>)
  attributes {topology = #mpmd.topology<<"mesh1" : <["x"=4]>>>}
{
// CHECK:      %[[USER1:.*]] = mpmd.fragment<mesh="mesh1", origin=["m1"]>
// CHECK-NEXT:    stablehlo.subtract
// CHECK-NEXT:    stablehlo.add

// CHECK:      %[[USER2:.*]] = mpmd.fragment<mesh="mesh1", origin=["m1"]>
// CHECK-NEXT:    stablehlo.subtract
// CHECK-NEXT:    stablehlo.add

// CHECK:      %[[USER3:.*]] = mpmd.fragment<mesh="mesh1", origin=["m1"]>
// CHECK-NEXT:    stablehlo.subtract
// CHECK-NEXT:    stablehlo.add

// CHECK:      return %[[USER1]], %[[USER2]], %[[USER3]]
  %s = stablehlo.subtract %arg0, %arg0 : tensor<8x16xi32>
  %as = mpmd.assign %s : (tensor<8x16xi32>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %user1 = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %as) (%arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<8x16xi32>
    mpmd.return %2 : tensor<8x16xi32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %user2 = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %as) (%arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<8x16xi32>
    mpmd.return %2 : tensor<8x16xi32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %user3 = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %as) (%arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<8x16xi32>
    mpmd.return %2 : tensor<8x16xi32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  return %user1, %user2, %user3 : !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
}

// CHECK-LABEL: @many_fragment_users_above_limit
func.func @many_fragment_users_above_limit(%arg0: tensor<8x16xi32>, %arg1: !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>)
  -> (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>)
  attributes {topology = #mpmd.topology<<"mesh1" : <["x"=4]>>>}
{
// CHECK:      %[[INFERRED:.*]] = mpmd.fragment<mesh="mesh1", origin=[]>
// CHECK-NEXT:    stablehlo.subtract
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT: }
// CHECK-NEXT: %[[INF_UNASSIGN:.*]] = mpmd.unassign %[[INFERRED]]
// CHECK-NEXT: %[[INF_ASSIGN:.*]] = mpmd.assign %[[INF_UNASSIGN]]
// CHECK:      %[[USER1:.*]] = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %[[INF_ASSIGN]])
// CHECK-NEXT:    stablehlo.add
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT: }
// CHECK:      %[[USER2:.*]] = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %[[INF_ASSIGN]])
// CHECK-NEXT:    stablehlo.add
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT: }
// CHECK:      %[[USER3:.*]] = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %[[INF_ASSIGN]])
// CHECK-NEXT:    stablehlo.add
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT: }
// CHECK:      %[[USER4:.*]] = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %[[INF_ASSIGN]])
// CHECK-NEXT:    stablehlo.add
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT: }
// CHECK:      return %[[USER1]], %[[USER2]], %[[USER3]], %[[USER4]]
  %s = stablehlo.subtract %arg0, %arg0 : tensor<8x16xi32>
  %as = mpmd.assign %s : (tensor<8x16xi32>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %user1 = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %as) (%arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<8x16xi32>
    mpmd.return %2 : tensor<8x16xi32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %user2 = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %as) (%arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<8x16xi32>
    mpmd.return %2 : tensor<8x16xi32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %user3 = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %as) (%arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<8x16xi32>
    mpmd.return %2 : tensor<8x16xi32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %user4 = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %as) (%arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<8x16xi32>
    mpmd.return %2 : tensor<8x16xi32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  return %user1, %user2, %user3, %user4 : !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
}

// CHECK-LABEL: func @many_user_assigns_single_fragment_consumer
func.func @many_user_assigns_single_fragment_consumer(%arg0: tensor<8x16xi32>)
  -> (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>)
  attributes {topology = #mpmd.topology<<"mesh1" : <["x"=4]>>>}
{
// CHECK-NEXT: %[[ARG_ASSIGN:.*]] = mpmd.assign %arg0
// CHECK-NEXT: mpmd.fragment<mesh="mesh1", origin=[]> (%[[ARG_ASSIGN]])
// CHECK-NEXT:   stablehlo.multiply
// CHECK-NEXT:   add
// CHECK-NEXT:   add
// CHECK-NEXT:   add
// CHECK-NEXT:   mpmd.return
// CHECK-NEXT: }
// CHECK-NEXT: return
  %m = stablehlo.multiply %arg0, %arg0 : tensor<8x16xi32>
  %a1 = mpmd.assign %m : (tensor<8x16xi32>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %a2 = mpmd.assign %m : (tensor<8x16xi32>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %a3 = mpmd.assign %m : (tensor<8x16xi32>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %a4 = mpmd.assign %m : (tensor<8x16xi32>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %consumer = mpmd.fragment<mesh="mesh1", origin=[]> (%a1, %a2, %a3, %a4)
    (%arg1: tensor<8x16xi32>, %arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>, %arg4: tensor<8x16xi32>)
  {
    %a = stablehlo.add %arg1, %arg2 : tensor<8x16xi32>
    %b = stablehlo.add %a, %arg3 : tensor<8x16xi32>
    %c = stablehlo.add %b, %arg4 : tensor<8x16xi32>
    mpmd.return %c : tensor<8x16xi32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  return %consumer : !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
}

// CHECK-LABEL: func @inline_op_with_free_vars
func.func @inline_op_with_free_vars(%arg0: tensor<4x16xf32>, %arg1: tensor<4x16xf32>, %arg2: tensor<i32>)
  -> !mpmd.mesh_tensor<"mesh1", tensor<4x16xf32>> attributes {
    "topology"=#mpmd.topology<<"mesh1": <["x"=4]>>>}
{
// CHECK-DAG:  %[[ARG2_ASSIGN:.*]] = mpmd.assign %arg2
// CHECK-DAG:  %[[ARG1_ASSIGN:.*]] = mpmd.assign %arg1
// CHECK-DAG:  %[[ARG0_ASSIGN:.*]] = mpmd.assign %arg0

// CHECK-NEXT: mpmd.fragment<mesh="mesh1", origin=[]> (%[[ARG1_ASSIGN]], %[[ARG2_ASSIGN]], %[[ARG0_ASSIGN]])
// CHECK-SAME: (%arg3: tensor<4x16xf32>, %arg4: tensor<i32>, %arg5: tensor<4x16xf32>)
// CHECK-NEXT:   %[[MULT:.*]] = stablehlo.multiply %arg5, %arg5 : tensor<4x16xf32>
// CHECK-NEXT:   %[[CASE:.*]] = "stablehlo.case"(%arg4) ({
// CHECK-NEXT:     %[[ADD:.*]] = stablehlo.add %[[MULT]], %arg3
// CHECK-NEXT:     stablehlo.return %[[ADD]]
// CHECK-NEXT:   }, {
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<4x16xf32>
// CHECK-NEXT:   }) : (tensor<i32>) -> tensor<4x16xf32>
// CHECK-NEXT:   mpmd.return %[[CASE]] : tensor<4x16xf32>
// CHECK-NEXT: }
  %var = stablehlo.multiply %arg0, %arg0 : tensor<4x16xf32>
  %case = "stablehlo.case" (%arg2) ({
    %4 = stablehlo.add %var, %arg1 : tensor<4x16xf32>
    stablehlo.return %4 : tensor<4x16xf32>
  }, {
    stablehlo.return %arg1 : tensor<4x16xf32>
  }) : (tensor<i32>) -> tensor<4x16xf32>

  %a = mpmd.assign %case : (tensor<4x16xf32>) -> !mpmd.mesh_tensor<"mesh1", tensor<4x16xf32>>
  %single_consumer = mpmd.fragment<mesh="mesh1", origin=[]> (%a) (%arg3: tensor<4x16xf32>) {
    mpmd.return %arg3 : tensor<4x16xf32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<4x16xf32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<4x16xf32>>
  func.return %single_consumer : !mpmd.mesh_tensor<"mesh1", tensor<4x16xf32>>
}

// CHECK-LABEL: func @clone_op_with_free_vars
func.func @clone_op_with_free_vars(%arg0: tensor<4x16xf32>, %arg1: tensor<4x16xf32>, %arg2: tensor<i32>)
  -> (!mpmd.mesh_tensor<"mesh1", tensor<4x16xf32>>, !mpmd.mesh_tensor<"mesh1", tensor<4x16xf32>>) attributes {
    "topology"=#mpmd.topology<<"mesh1": <["x"=4]>>>}
{

// CHECK-DAG:  %[[ARG2_ASSIGN_C1:.*]] = mpmd.assign %arg2
// CHECK-DAG:  %[[ARG1_ASSIGN_C1:.*]] = mpmd.assign %arg1
// CHECK-DAG:  %[[ARG0_ASSIGN_C1:.*]] = mpmd.assign %arg0

// CHECK-NEXT: %[[CONSUMER1:.*]] = mpmd.fragment<mesh="mesh1", origin=[]> (%[[ARG1_ASSIGN_C1]], %[[ARG2_ASSIGN_C1]], %[[ARG0_ASSIGN_C1]])
// CHECK-SAME: (%arg3: tensor<4x16xf32>, %arg4: tensor<i32>, %arg5: tensor<4x16xf32>)
// CHECK-NEXT:   %[[MULT:.*]] = stablehlo.multiply %arg5, %arg5 : tensor<4x16xf32>
// CHECK-NEXT:   %[[CASE:.*]] = "stablehlo.case"(%arg4) ({
// CHECK-NEXT:     %[[ADD:.*]] = stablehlo.add %[[MULT]], %arg3
// CHECK-NEXT:     stablehlo.return %[[ADD]]
// CHECK-NEXT:   }, {
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<4x16xf32>
// CHECK-NEXT:   }) : (tensor<i32>) -> tensor<4x16xf32>
// CHECK-NEXT:   mpmd.return %[[CASE]] : tensor<4x16xf32>
// CHECK-NEXT: }

// CHECK-DAG:  %[[ARG2_ASSIGN_C2:.*]] = mpmd.assign %arg2
// CHECK-DAG:  %[[ARG1_ASSIGN_C2:.*]] = mpmd.assign %arg1
// CHECK-DAG:  %[[ARG0_ASSIGN_C2:.*]] = mpmd.assign %arg0

// CHECK-NEXT: %[[CONSUMER2:.*]] = mpmd.fragment<mesh="mesh1", origin=[]> (%[[ARG1_ASSIGN_C2]], %[[ARG2_ASSIGN_C2]], %[[ARG0_ASSIGN_C2]])
// CHECK-SAME: (%arg3: tensor<4x16xf32>, %arg4: tensor<i32>, %arg5: tensor<4x16xf32>)
// CHECK-NEXT:   %[[MULT:.*]] = stablehlo.multiply %arg5, %arg5 : tensor<4x16xf32>
// CHECK-NEXT:   %[[CASE:.*]] = "stablehlo.case"(%arg4) ({
// CHECK-NEXT:     %[[ADD:.*]] = stablehlo.add %[[MULT]], %arg3
// CHECK-NEXT:     stablehlo.return %[[ADD]]
// CHECK-NEXT:   }, {
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<4x16xf32>
// CHECK-NEXT:   }) : (tensor<i32>) -> tensor<4x16xf32>
// CHECK-NEXT:   mpmd.return %[[CASE]] : tensor<4x16xf32>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[CONSUMER1]], %[[CONSUMER2]]
  %var = stablehlo.multiply %arg0, %arg0 : tensor<4x16xf32>
  %case = "stablehlo.case" (%arg2) ({
    %4 = stablehlo.add %var, %arg1 : tensor<4x16xf32>
    stablehlo.return %4 : tensor<4x16xf32>
  }, {
    stablehlo.return %arg1 : tensor<4x16xf32>
  }) : (tensor<i32>) -> tensor<4x16xf32>

  %a = mpmd.assign %case : (tensor<4x16xf32>) -> !mpmd.mesh_tensor<"mesh1", tensor<4x16xf32>>
  %consumer1 = mpmd.fragment<mesh="mesh1", origin=[]> (%a) (%arg3: tensor<4x16xf32>) {
    mpmd.return %arg3 : tensor<4x16xf32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<4x16xf32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<4x16xf32>>
  %consumer2 = mpmd.fragment<mesh="mesh1", origin=[]> (%a) (%arg3: tensor<4x16xf32>) {
    mpmd.return %arg3 : tensor<4x16xf32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<4x16xf32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<4x16xf32>>
  func.return %consumer1, %consumer2 : !mpmd.mesh_tensor<"mesh1", tensor<4x16xf32>>, !mpmd.mesh_tensor<"mesh1", tensor<4x16xf32>>
}

// Trivial ops (single operand, single result, no regions) should be cloned
// into their fragment users even when the number of users exceeds max-clones.
// This avoids creating millions of wrapper fragments for ops like
// sdy.sharding_constraint.
// Here stablehlo.negate has 1 operand, 1 result, 0 regions, and 4 fragment
// users (above max-clones=3). It should be cloned, not wrapped.
// CHECK-LABEL: @trivial_op_above_limit_cloned_not_wrapped
func.func @trivial_op_above_limit_cloned_not_wrapped(%arg0: tensor<8x16xi32>, %arg1: !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>)
  -> (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>)
  attributes {topology = #mpmd.topology<<"mesh1" : <["x"=4]>>>}
{
// The negate should be absorbed (cloned) into each fragment, not wrapped.
// CHECK:      %[[USER1:.*]] = mpmd.fragment<mesh="mesh1", origin=["m1"]>
// CHECK-NEXT:    stablehlo.negate
// CHECK-NEXT:    stablehlo.add

// CHECK:      %[[USER2:.*]] = mpmd.fragment<mesh="mesh1", origin=["m1"]>
// CHECK-NEXT:    stablehlo.negate
// CHECK-NEXT:    stablehlo.add

// CHECK:      %[[USER3:.*]] = mpmd.fragment<mesh="mesh1", origin=["m1"]>
// CHECK-NEXT:    stablehlo.negate
// CHECK-NEXT:    stablehlo.add

// CHECK:      %[[USER4:.*]] = mpmd.fragment<mesh="mesh1", origin=["m1"]>
// CHECK-NEXT:    stablehlo.negate
// CHECK-NEXT:    stablehlo.add

// CHECK:      return %[[USER1]], %[[USER2]], %[[USER3]], %[[USER4]]
  %n = stablehlo.negate %arg0 : tensor<8x16xi32>
  %an = mpmd.assign %n : (tensor<8x16xi32>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %user1 = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %an) (%arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<8x16xi32>
    mpmd.return %2 : tensor<8x16xi32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %user2 = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %an) (%arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<8x16xi32>
    mpmd.return %2 : tensor<8x16xi32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %user3 = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %an) (%arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<8x16xi32>
    mpmd.return %2 : tensor<8x16xi32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %user4 = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %an) (%arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<8x16xi32>
    mpmd.return %2 : tensor<8x16xi32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  return %user1, %user2, %user3, %user4 : !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
}

// Trivial ops that have side effects should NOT be cloned even though they
// meet the shape criteria (1 operand, 1 result, 0 regions). Cloning an
// impure op would duplicate side effects. Here stablehlo.custom_call with
// has_side_effect=true has 1 operand, 1 result, 0 regions, and 4 fragment
// users (above max-clones=3). It should be wrapped, not cloned.
// CHECK-LABEL: @impure_trivial_op_above_limit_wrapped_not_cloned
func.func @impure_trivial_op_above_limit_wrapped_not_cloned(%arg0: tensor<8x16xi32>, %arg1: !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>)
  -> (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>)
  attributes {topology = #mpmd.topology<<"mesh1" : <["x"=4]>>>}
{
// The custom_call should be wrapped in its own fragment, not cloned.
// CHECK:      %[[INFERRED:.*]] = mpmd.fragment<mesh="mesh1", origin=[]>
// CHECK-NEXT:    stablehlo.custom_call
// CHECK-NEXT:    mpmd.return
// CHECK-NEXT: }
// CHECK-NEXT: %[[INF_UNASSIGN:.*]] = mpmd.unassign %[[INFERRED]]
// CHECK-NEXT: %[[INF_ASSIGN:.*]] = mpmd.assign %[[INF_UNASSIGN]]
// CHECK:      %[[USER1:.*]] = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %[[INF_ASSIGN]])
// CHECK-NEXT:    stablehlo.add
// CHECK:      %[[USER2:.*]] = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %[[INF_ASSIGN]])
// CHECK-NEXT:    stablehlo.add
// CHECK:      %[[USER3:.*]] = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %[[INF_ASSIGN]])
// CHECK-NEXT:    stablehlo.add
// CHECK:      %[[USER4:.*]] = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %[[INF_ASSIGN]])
// CHECK-NEXT:    stablehlo.add
// CHECK:      return %[[USER1]], %[[USER2]], %[[USER3]], %[[USER4]]
  %cc = stablehlo.custom_call @side_effecting_op(%arg0) {has_side_effect = true} : (tensor<8x16xi32>) -> tensor<8x16xi32>
  %acc = mpmd.assign %cc : (tensor<8x16xi32>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %user1 = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %acc) (%arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<8x16xi32>
    mpmd.return %2 : tensor<8x16xi32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %user2 = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %acc) (%arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<8x16xi32>
    mpmd.return %2 : tensor<8x16xi32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %user3 = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %acc) (%arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<8x16xi32>
    mpmd.return %2 : tensor<8x16xi32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  %user4 = mpmd.fragment<mesh="mesh1", origin=["m1"]> (%arg1, %acc) (%arg2: tensor<8x16xi32>, %arg3: tensor<8x16xi32>) {
    %2 = stablehlo.add %arg2, %arg3 : tensor<8x16xi32>
    mpmd.return %2 : tensor<8x16xi32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>) -> !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
  return %user1, %user2, %user3, %user4 : !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>, !mpmd.mesh_tensor<"mesh1", tensor<8x16xi32>>
}
