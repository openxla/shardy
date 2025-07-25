// RUN: mpmd_opt %s -verify-diagnostics -split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: func @no_sharding_nor_memory_kind
// CHECK-SAME: !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>>
func.func @no_sharding_nor_memory_kind(%arg0 : !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>>)
    attributes {"topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=4]>>>}
{
  func.return
}

// -----

// CHECK-LABEL: func @sharding_but_no_memory_kind
// CHECK-SAME: !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>
func.func @sharding_but_no_memory_kind(%arg0 : !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>)
    attributes {"topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=4]>>>}
{
  func.return
}

// -----

// CHECK-LABEL: func @no_sharding_but_memory_kind
// CHECK-SAME
func.func @no_sharding_but_memory_kind(%arg0 : !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>, memory_kind="device">)
    attributes {"topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=4]>>>}
{
  func.return
}

// -----

// CHECK-LABEL: func @sharding_and_memory_kind
// CHECK-SAME: !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>, memory_kind="pinned_host">
func.func @sharding_and_memory_kind(%arg0 : !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>, memory_kind="pinned_host">)
    attributes {"topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=4]>>>}
{
  func.return
}

// -----

// expected-error @+1 {{expected '>'}}
func.func @missing_comma(%arg0 : !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]> memory_kind="pinned_host">)
    attributes {"topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=4]>>>}
{
  func.return
}

// -----

// expected-error @+1 {{unbalanced '<' character in pretty dialect name}}
func.func @last_gt_missing(%arg0 : !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>)
    attributes {"topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=4]>>>}
{
  func.return
}

// -----

// expected-error @+1 {{expected '>'}}
func.func @too_many_commas(%arg0 : !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>, memory_kind="pinned_host",>)
    attributes {"topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=4]>>>}
{
  func.return
}

// -----

// expected-error @+1 {{expected '>'}}
func.func @too_many_commas(%arg0 : !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>, memory_kind="pinned_host",>)
    attributes {"topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=4]>>>}
{
  func.return
}

// -----

// expected-error @+1 {{expected 'memory_kind'}}
func.func @too_many_commas(%arg0 : !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>, >)
    attributes {"topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=4]>>>}
{
  func.return
}

// -----

// expected-error @+1 {{expected '<'}}
func.func @invalid_sharding_without_memory_kind(%arg0 : !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>, sharding="not_a_sharding">)
    attributes {"topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=4]>>>}
{
  func.return
}

// -----

// expected-error @+1 {{expected '<'}}
func.func @invalid_sharding_without_comma_before_memory_kind(%arg0 : !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>, sharding="not_a_sharding" memory_kind="pinned_host">)
    attributes {"topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=4]>>>}
{
  func.return
}


// -----

// expected-error @+1 {{expected '<'}}
func.func @invalid_sharding(%arg0 : !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>, sharding="not_a_sharding", memory_kind="pinned_host">)
    attributes {"topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=4]>>>}
{
  func.return
}
