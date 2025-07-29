// RUN: mpmd_opt %s 2>&1 | FileCheck %s

// The purpose of this test is to demonstrate that it's possible to reshard a
// tensor with a fragment, when in global_view.

module {

func.func @reshard(%arg0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
    -> !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>> attributes {
    "topology"=#mpmd.topology<<"mesh1": <["x"=2, "y"=4]>>>}
{
  // The fragment reshards a value (from sharded to replicated). This is only
  // possible when the module is in global view.
  // CHECK: (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>) -> !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>>
  %0 = mpmd.fragment<mesh="mesh1", origin=[]> (%arg0) (%arg1: tensor<12x16xf32>) {
    mpmd.return %arg1 : tensor<12x16xf32>
  } : (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
   -> !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>>
  func.return %0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>>
}

}
