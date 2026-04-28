// RUN: mpmd_opt %s -mpmd-sink-create-token-into-fragments -allow-unregistered-dialect | FileCheck %s

#topology = #mpmd.topology<<"mesh0": <["x"=2]>>>

// CHECK-LABEL: func @sink_token_input
// CHECK-SAME: (%[[ARG0:.*]]: !stablehlo.token, %[[ARG1:.*]]: !mpmd.mesh_tensor<"mesh0", tensor<f32>>)
func.func @sink_token_input(%arg0: !stablehlo.token, %arg1: !mpmd.mesh_tensor<"mesh0", tensor<f32>>) -> !mpmd.mesh_tensor<"mesh0", tensor<f32>> attributes {topology = #topology} {

  // Verify that the token operand %[[ARG0]] is removed.
  // CHECK: %[[FRAG:.*]] = mpmd.fragment<mesh="mesh0", origin=[]> (%[[ARG1]])
  // CHECK-NOT: (%[[ARG0]]

  %0 = mpmd.fragment<mesh="mesh0", origin=[]> (%arg0, %arg1) (%arg2: !stablehlo.token, %arg3: tensor<f32>) {

    // Verify that a new token is created locally and used.
    // CHECK: %[[NEW_TOKEN:.*]] = stablehlo.create_token
    // CHECK: %[[RES:.*]] = "some.op"(%[[NEW_TOKEN]], %arg2)
    // CHECK: mpmd.return %[[RES]] : tensor<f32>

    %1 = "some.op"(%arg2, %arg3) : (!stablehlo.token, tensor<f32>) -> tensor<f32>
    mpmd.return %1 : tensor<f32>
  } : (!stablehlo.token, !mpmd.mesh_tensor<"mesh0", tensor<f32>>) -> (!mpmd.mesh_tensor<"mesh0", tensor<f32>>)
  return %0 : !mpmd.mesh_tensor<"mesh0", tensor<f32>>
}

// CHECK-LABEL: func @drop_token_output
func.func @drop_token_output(%arg0: !mpmd.mesh_tensor<"mesh0", tensor<f32>>) -> !mpmd.mesh_tensor<"mesh0", tensor<f32>> attributes {topology = #topology} {

  // Verify that the fragment now only returns a single tensor.
  // CHECK: %[[NEW_FRAG:.*]] = mpmd.fragment<mesh="mesh0", origin=[]> (%arg0)
  // CHECK-SAME: (%arg1: tensor<f32>) {

  %0:2 = mpmd.fragment<mesh="mesh0", origin=[]> (%arg0) (%arg1: tensor<f32>) {
    %token = stablehlo.create_token : !stablehlo.token

    // Verify that the token is dropped from the return.
    // CHECK: mpmd.return %arg1 : tensor<f32>

    mpmd.return %arg1, %token : tensor<f32>, !stablehlo.token
  } : (!mpmd.mesh_tensor<"mesh0", tensor<f32>>) -> (!mpmd.mesh_tensor<"mesh0", tensor<f32>>, !stablehlo.token)

  // Verify that uses of the non-token result are remapped to the new fragment result.
  // CHECK: return %[[NEW_FRAG]] : !mpmd.mesh_tensor<"mesh0", tensor<f32>>
  return %0#0 : !mpmd.mesh_tensor<"mesh0", tensor<f32>>
}

// CHECK-LABEL: func @sink_multiple_tokens
func.func @sink_multiple_tokens(%arg0: !stablehlo.token, %arg1: !stablehlo.token, %arg2: !mpmd.mesh_tensor<"mesh0", tensor<f32>>) -> !mpmd.mesh_tensor<"mesh0", tensor<f32>> attributes {topology = #topology} {
  // Verify that both token operands are removed.
  // CHECK: mpmd.fragment<mesh="mesh0", origin=[]> (%arg2)
  %0:3 = mpmd.fragment<mesh="mesh0", origin=[]> (%arg0, %arg1, %arg2) (%arg3: !stablehlo.token, %arg4: !stablehlo.token, %arg5: tensor<f32>) {
    // CHECK-DAG: %[[T1:.*]] = stablehlo.create_token
    // CHECK-DAG: %[[T2:.*]] = stablehlo.create_token
    // CHECK: %[[RES:.*]] = "some.op"(%[[T1]], %[[T2]], %arg3)
    %1 = "some.op"(%arg3, %arg4, %arg5) : (!stablehlo.token, !stablehlo.token, tensor<f32>) -> tensor<f32>
    // CHECK: mpmd.return %[[RES]] : tensor<f32>
    mpmd.return %1, %arg3, %arg4 : tensor<f32>, !stablehlo.token, !stablehlo.token
  } : (!stablehlo.token, !stablehlo.token, !mpmd.mesh_tensor<"mesh0", tensor<f32>>) -> (!mpmd.mesh_tensor<"mesh0", tensor<f32>>, !stablehlo.token, !stablehlo.token)

  return %0#0 : !mpmd.mesh_tensor<"mesh0", tensor<f32>>
}

// CHECK-LABEL: func @inter_fragment_tokens
func.func @inter_fragment_tokens(%arg0: !mpmd.mesh_tensor<"mesh0", tensor<f32>>) -> !mpmd.mesh_tensor<"mesh0", tensor<f32>> attributes {topology = #topology} {
  // CHECK: %[[FRAG1:.*]] = mpmd.fragment<mesh="mesh0", origin=[]> (%arg0)
  %0:2 = mpmd.fragment<mesh="mesh0", origin=[]> (%arg0) (%arg1: tensor<f32>) {
    %token = stablehlo.create_token : !stablehlo.token
    // CHECK: mpmd.return %arg1 : tensor<f32>
    mpmd.return %arg1, %token : tensor<f32>, !stablehlo.token
  } : (!mpmd.mesh_tensor<"mesh0", tensor<f32>>) -> (!mpmd.mesh_tensor<"mesh0", tensor<f32>>, !stablehlo.token)

  // Verify that the token input from Fragment A is removed and replaced by a local token.
  // CHECK: %[[FRAG2:.*]] = mpmd.fragment<mesh="mesh0", origin=[]> (%[[FRAG1]])
  %1 = mpmd.fragment<mesh="mesh0", origin=[]> (%0#1, %0#0) (%arg2: !stablehlo.token, %arg3: tensor<f32>) {
    // CHECK: %[[NEW_TOKEN:.*]] = stablehlo.create_token
    // CHECK: %[[RES:.*]] = "some.op"(%[[NEW_TOKEN]], %arg1)
    %2 = "some.op"(%arg2, %arg3) : (!stablehlo.token, tensor<f32>) -> tensor<f32>
    mpmd.return %2 : tensor<f32>
  } : (!stablehlo.token, !mpmd.mesh_tensor<"mesh0", tensor<f32>>) -> (!mpmd.mesh_tensor<"mesh0", tensor<f32>>)

  return %1 : !mpmd.mesh_tensor<"mesh0", tensor<f32>>
}
