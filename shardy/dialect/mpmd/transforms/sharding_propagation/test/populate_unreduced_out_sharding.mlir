// RUN: mpmd_opt %s -mpmd-populate-unreduced-out-sharding -split-input-file | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @unreduced_out_sharding
func.func @unreduced_out_sharding(%arg0: !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>) -> !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>
  attributes { "topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=2]>>> }
{
  // CHECK: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="mesh", origin=["f1"], in_shardings=[<@mesh, [{}, {}], unreduced={"x"}>], out_shardings=[<@mesh, [{}, {}], unreduced={"x"}>]>
  // CHECK-SAME: (%arg0) (%[[ARG1:.*]]: tensor<8x8xf32>) {
  %0 = mpmd.fragment<mesh="mesh", origin=["f1"], in_shardings=[<@mesh, [{}, {}], unreduced={"x"}>]> (%arg0) (%arg1: tensor<8x8xf32>) {
    // CHECK-NEXT: %[[CONSTRAINT:.*]] = sdy.sharding_constraint %[[ARG1]] <@mesh, [{}, {}], unreduced={"x"}> : tensor<8x8xf32>
    // CHECK-NEXT: mpmd.return %[[CONSTRAINT]] : tensor<8x8xf32>
    %1 = sdy.sharding_constraint %arg1 <@mesh, [{}, {}], unreduced={"x"}> : tensor<8x8xf32>
    mpmd.return %1 : tensor<8x8xf32>
  } : (!mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>) -> !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>
  return %0 : !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @existing_out_sharding
func.func @existing_out_sharding(%arg0: !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>) -> !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>
  attributes { "topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=2]>>> }
{
  // CHECK: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="mesh", origin=["f1"], in_shardings=[<@mesh, [{}, {"y"}], unreduced={"x"}>], out_shardings=[<@mesh, [{}, {"y"}], unreduced={"x"}>]>
  %0 = mpmd.fragment<mesh="mesh", origin=["f1"], in_shardings=[<@mesh, [{}, {"y"}], unreduced={"x"}>], out_shardings=[<@mesh, [{}, {"y"}]>]> (%arg0) (%arg1: tensor<8x8xf32>) {
    // CHECK-NEXT: %[[CONSTRAINT:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{}, {"y"}], unreduced={"x"}> : tensor<8x8xf32>
    // CHECK-NEXT: mpmd.return %[[CONSTRAINT]] : tensor<8x8xf32>
    %1 = sdy.sharding_constraint %arg1 <@mesh, [{}, {"y"}], unreduced={"x"}> : tensor<8x8xf32>
    mpmd.return %1 : tensor<8x8xf32>
  } : (!mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>) -> !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>
  return %0 : !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @multiple_out_shardings
func.func @multiple_out_shardings(%arg0: !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>) -> (!mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>, !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>)
  attributes { "topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=2]>>> }
{
  // CHECK: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="mesh", origin=["f1"], in_shardings=[<@mesh, [{}, {}], unreduced={"x"}>], out_shardings=[<@mesh, [{}, {}], unreduced={"x"}>, <@mesh, [{}, {"y"}]>]>
  %0:2 = mpmd.fragment<mesh="mesh", origin=["f1"], in_shardings=[<@mesh, [{}, {}], unreduced={"x"}>], out_shardings=[<@mesh, [{}, {}]>, <@mesh, [{}, {"y"}]>]> (%arg0) (%arg1: tensor<8x8xf32>) {
    // CHECK-NEXT: %[[CONSTRAINT:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{}, {}], unreduced={"x"}> : tensor<8x8xf32>
    // CHECK-NEXT: %[[CONSTRAINT2:.*]] = sdy.sharding_constraint %[[CONSTRAINT]] <@mesh, [{}, {"y"}]> : tensor<8x8xf32>
    // CHECK-NEXT: mpmd.return %[[CONSTRAINT]], %[[CONSTRAINT2]] : tensor<8x8xf32>, tensor<8x8xf32>
    %2 = sdy.sharding_constraint %arg1 <@mesh, [{}, {}], unreduced={"x"}> : tensor<8x8xf32>
    %3 = sdy.sharding_constraint %2 <@mesh, [{}, {"y"}]> : tensor<8x8xf32>
    mpmd.return %2, %3 : tensor<8x8xf32>, tensor<8x8xf32>
  } : (!mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>) -> (!mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>, !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>)
  return %0#0, %0#1 : !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>, !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @one_unreduced_one_not_specified_sharding
func.func @one_unreduced_one_not_specified_sharding(%arg0: !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>, %arg1: !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>) -> (!mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>, !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>)
  attributes { "topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=2]>>> }
{
  // CHECK: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="mesh", origin=["f1"], in_shardings=[<@mesh, [{}, {}], unreduced={"x"}>, <@mesh, [{}, {}]>], out_shardings=[<@mesh, [{}, {}], unreduced={"x"}>, <@mesh, [{}, {}]>]>
  %0:2 = mpmd.fragment<mesh="mesh", origin=["f1"], in_shardings=[<@mesh, [{}, {}], unreduced={"x"}>, <@mesh, [{}, {}]>]> (%arg0, %arg1) (%arg2: tensor<8x8xf32>, %arg3: tensor<8x8xf32>) {
    // CHECK-NEXT: %[[CONSTRAINT:.*]] = sdy.sharding_constraint %arg2 <@mesh, [{}, {}], unreduced={"x"}> : tensor<8x8xf32>
    // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg3, %arg3 : tensor<8x8xf32>
    // CHECK-NEXT: mpmd.return %[[CONSTRAINT]], %[[ADD]] : tensor<8x8xf32>, tensor<8x8xf32>
    %2 = sdy.sharding_constraint %arg2 <@mesh, [{}, {}], unreduced={"x"}> : tensor<8x8xf32>
    %3 = stablehlo.add %arg3, %arg3 : tensor<8x8xf32>
    mpmd.return %2, %3 : tensor<8x8xf32>, tensor<8x8xf32>
  } : (!mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>, !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>) -> (!mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>, !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>)
  return %0#0, %0#1 : !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>, !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @no_unreduced_axes
func.func @no_unreduced_axes(%arg0: !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>) -> !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>
  attributes { "topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=2]>>> }
{
  // CHECK: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="mesh", origin=["f1"], in_shardings=[<@mesh, [{}, {"y"}]>], out_shardings=[<@mesh, [{}, {"y"}]>]>
  %0 = mpmd.fragment<mesh="mesh", origin=["f1"], in_shardings=[<@mesh, [{}, {"y"}]>], out_shardings=[<@mesh, [{}, {"y"}]>]> (%arg0) (%arg1: tensor<8x8xf32>) {
    // CHECK-NEXT: %[[CONSTRAINT:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{}, {"y"}]> : tensor<8x8xf32>
    %1 = sdy.sharding_constraint %arg1 <@mesh, [{}, {"y"}]> : tensor<8x8xf32>
    mpmd.return %1 : tensor<8x8xf32>
  } : (!mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>) -> !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>
  return %0 : !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @no_sharding
func.func @no_sharding(%arg0: !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>) -> !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>
  attributes { "topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=2]>>> }
{
  // CHECK: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="mesh", origin=["f1"]>
  %0 = mpmd.fragment<mesh="mesh", origin=["f1"]> (%arg0) (%arg1: tensor<8x8xf32>) {
    // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg1, %arg1 : tensor<8x8xf32>
    %1 = stablehlo.add %arg1, %arg1 : tensor<8x8xf32>
    mpmd.return %1 : tensor<8x8xf32>
  } : (!mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>) -> !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>
  return %0 : !mpmd.mesh_tensor<"mesh", tensor<8x8xf32>>
}
