// RUN: mpmd_opt %s 2>&1 | FileCheck %s

// CHECK-LABEL: func @fragment_with_stage
func.func @fragment_with_stage(%arg0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
    -> !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>> attributes {
    "topology"=#mpmd.topology<<"mesh1": <["x"=2, "y"=4]>>>}
{
  // CHECK: mpmd.fragment<mesh="mesh1", origin=["f1"], stage=123> (%arg0) (%arg1: tensor<12x16xf32>)
  %0 = mpmd.fragment<mesh="mesh1", origin=["f1"], stage=123> (%arg0) (%arg1: tensor<12x16xf32>) {
    mpmd.return %arg1 : tensor<12x16xf32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
   -> (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
  func.return %0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>
}

// CHECK-LABEL: func @fragment_with_stage_and_in_shardings
func.func @fragment_with_stage_and_in_shardings(%arg0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
    -> !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>> attributes {
    "topology"=#mpmd.topology<<"mesh1": <["x"=2, "y"=4]>>>}
{
  // CHECK: mpmd.fragment<mesh="mesh1", origin=["f1"], stage=123, in_shardings=[<@mesh, [{"x"}, {?}]>]> (%arg0) (%arg1: tensor<12x16xf32>)
  %0 = mpmd.fragment<mesh="mesh1", origin=["f1"], stage=123, in_shardings=[<@mesh, [{"x"}, {?}]>]> (%arg0) (%arg1: tensor<12x16xf32>) {
    mpmd.return %arg1 : tensor<12x16xf32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
   -> (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
  func.return %0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>
}

// CHECK-LABEL: func @fragment_with_stage_and_out_shardings
func.func @fragment_with_stage_and_out_shardings(%arg0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
    -> !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>> attributes {
    "topology"=#mpmd.topology<<"mesh1": <["x"=2, "y"=4]>>>}
{
  // CHECK: mpmd.fragment<mesh="mesh1", origin=["f1"], stage=123, out_shardings=[<@mesh, [{"x"}, {?}]>]> (%arg0) (%arg1: tensor<12x16xf32>)
  %0 = mpmd.fragment<mesh="mesh1", origin=["f1"], stage=123, out_shardings=[<@mesh, [{"x"}, {?}]>]> (%arg0) (%arg1: tensor<12x16xf32>) {
    mpmd.return %arg1 : tensor<12x16xf32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
   -> (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
  func.return %0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>
}

// CHECK-LABEL: func @fragment_with_in_shardings_without_stage
func.func @fragment_with_in_shardings_without_stage(%arg0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
    -> !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>> attributes {
    "topology"=#mpmd.topology<<"mesh1": <["x"=2, "y"=4]>>>}
{
  // CHECK: mpmd.fragment<mesh="mesh1", origin=["f1"], in_shardings=[<@mesh, [{"x"}, {?}]>]> (%arg0) (%arg1: tensor<12x16xf32>)
  %0 = mpmd.fragment<mesh="mesh1", origin=["f1"], in_shardings=[<@mesh, [{"x"}, {?}]>]> (%arg0) (%arg1: tensor<12x16xf32>) {
    mpmd.return %arg1 : tensor<12x16xf32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
   -> (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
  func.return %0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>
}

// CHECK-LABEL: func @fragment_with_out_shardings_without_stage
func.func @fragment_with_out_shardings_without_stage(%arg0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
    -> !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>> attributes {
    "topology"=#mpmd.topology<<"mesh1": <["x"=2, "y"=4]>>>}
{
  // CHECK: mpmd.fragment<mesh="mesh1", origin=["f1"], out_shardings=[<@mesh, [{"x"}, {?}]>]> (%arg0) (%arg1: tensor<12x16xf32>)
  %0 = mpmd.fragment<mesh="mesh1", origin=["f1"], out_shardings=[<@mesh, [{"x"}, {?}]>]> (%arg0) (%arg1: tensor<12x16xf32>) {
    mpmd.return %arg1 : tensor<12x16xf32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
   -> (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
  func.return %0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>
}

// CHECK-LABEL: func @fragment_with_stage_and_in_shardings_and_out_shardings
func.func @fragment_with_stage_and_in_shardings_and_out_shardings(%arg0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
    -> !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>> attributes {
    "topology"=#mpmd.topology<<"mesh1": <["x"=2, "y"=4]>>>}
{
  // CHECK: mpmd.fragment<mesh="mesh1", origin=["f1"], stage=123, in_shardings=[<@mesh, [{"x"}, {?}]>], out_shardings=[<@mesh, [{"x"}, {?}]>]> (%arg0) (%arg1: tensor<12x16xf32>)
  %0 = mpmd.fragment<mesh="mesh1", origin=["f1"], stage=123, in_shardings=[<@mesh, [{"x"}, {?}]>], out_shardings=[<@mesh, [{"x"}, {?}]>]> (%arg0) (%arg1: tensor<12x16xf32>) {
    mpmd.return %arg1 : tensor<12x16xf32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
   -> (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
  func.return %0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>
}

// CHECK-LABEL: func @fragment_without_stage_with_in_shardings_and_out_shardings
func.func @fragment_without_stage_with_in_shardings_and_out_shardings(%arg0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
    -> !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>> attributes {
    "topology"=#mpmd.topology<<"mesh1": <["x"=2, "y"=4]>>>}
{
  %0 = mpmd.fragment<mesh="mesh1", origin=["f1"], in_shardings=[<@mesh, [{"x"}, {?}]>], out_shardings=[<@mesh, [{"x"}, {?}]>]> (%arg0) (%arg1: tensor<12x16xf32>) {
    mpmd.return %arg1 : tensor<12x16xf32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
   -> (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
  func.return %0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>
}
