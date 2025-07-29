// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>


// CHECK-LABEL: func @all_gather
func.func @all_gather(%arg0: tensor<2x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg1: tensor<2x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<2x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, tensor<2x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  // #sdy.op_sharding_rule<([i, j], [i, j])->([i, k], [i, k]) {i=2, j=2, k=4} need_replication={j, k}>
  // CHECK-NEXT: %0 = sdy.reshard %arg1 <@mesh, [{"y"}, {}]> : tensor<2x2xi64>
  // CHECK-NEXT: %1:2 = "stablehlo.all_gather"(%arg0, %0)
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>, <@mesh, [{"y"}, {}]>]>}
  %0:2 = "stablehlo.all_gather"(%arg0, %arg1) {
    all_gather_dim = 1 : i64,
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>, <@mesh, [{}, {"y"}]>]>
  } : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
  // CHECK-NEXT: %2 = sdy.reshard %1#1 <@mesh, [{}, {"y"}]> : tensor<2x4xi64>
  // CHECK-NEXT: return %1#0, %2
  return %0#0, %0#1 :  tensor<2x4xi64>, tensor<2x4xi64>
}


// CHECK-LABEL: func @all_reduce
func.func @all_reduce(%arg0: tensor<2x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg1: tensor<2x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<2x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, tensor<2x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  // sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j], [i, j]) {i=2, j=2}>
  // CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<2x2xi64>
  // CHECK-NEXT: %1:2 = "stablehlo.all_reduce"(%0, %arg1)
  // CHECK: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>, <@mesh, [{}, {"y"}]>]>}
  %0:2 = "stablehlo.all_reduce"(%arg0, %arg1) ({
    ^bb0(%arg3: tensor<i64>, %arg4: tensor<i64>):
      %0 = "stablehlo.add"(%arg3, %arg4) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%0) : (tensor<i64>) -> ()
  }) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>, <@mesh, [{}, {"y"}]>]>
  } : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x2xi64>, tensor<2x2xi64>)
  // CHECK-NEXT: %2 = sdy.reshard %1#0 <@mesh, [{"y"}, {}]> : tensor<2x2xi64>
  // CHECK-NEXT: return %2, %1#1
  return %0#0, %0#1 : tensor<2x2xi64>, tensor<2x2xi64>
}

// CHECK-LABEL: func @all_to_all_same_dimension
func.func @all_to_all_same_dimension(%arg0: tensor<2x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg1: tensor<2x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> (tensor<2x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, tensor<2x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  // sdy.sharding_rule = #sdy.op_sharding_rule<([j, i], [j, i])->([k, i], [k, i]) {i=4, j=2, k=2} need_replication={j, k}>
  // CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{}, {"y"}]> : tensor<2x4xi64>
  // CHECK-NEXT: %1:2 = "stablehlo.all_to_all"(%0, %arg1)
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>, <@mesh, [{}, {"y"}]>]>
  %0:2 = "stablehlo.all_to_all"(%arg0, %arg1) {
    split_dimension = 0 : i64,
    concat_dimension = 0 : i64,
    split_count = 2 : i64,
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>, <@mesh, [{}, {"y"}]>]>
  } : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
  // CHECK-NEXT: %2 = sdy.reshard %1#0 <@mesh, [{"y"}, {}]> : tensor<2x4xi64>
  // CHECK-NEXT: return %2, %1#1
  return %0#0, %0#1 :  tensor<2x4xi64>, tensor<2x4xi64>
}


// CHECK-LABEL: func @collective_broadcast
func.func @collective_broadcast(%arg0 : tensor<2x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<2x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  // #sdy.op_sharding_rule<([i, j])->([i, j]) {i=2, j=4}>
  // CHECK-NEXT: %0 = "stablehlo.collective_broadcast"(%arg0)
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>}
  %0 = "stablehlo.collective_broadcast"(%arg0) {
    replica_groups = dense<[[2, 1]]> : tensor<1x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>
  } : (tensor<2x4xi64>) -> tensor<2x4xi64>
  // CHECK-NEXT: return %0 : tensor<2x4xi64>
  return %0 : tensor<2x4xi64>
}

// CHECK-LABEL: func @collective_permute
func.func @collective_permute(%arg0 : tensor<2x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<2x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  // #sdy.op_sharding_rule<([i, j])->([i, j]) {i=2, j=4}>
  // CHECK-NEXT: %0 = "stablehlo.collective_permute"(%arg0)
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>}
  %0 = "stablehlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>
  } : (tensor<2x4xi64>) -> tensor<2x4xi64>
  // CHECK-NEXT: return %0 : tensor<2x4xi64>
  return %0 : tensor<2x4xi64>
}

// CHECK-LABEL: func @reduce_scatter
func.func @reduce_scatter(%arg0: tensor<2x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) -> (tensor<2x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // #sdy.op_sharding_rule<([i, j])->([i, k]) {i=2, j=4, k=2} need_replication={j, k}>
  // CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{"y"}, {}]> : tensor<2x4xi64>
  // CHECK-NEXT: %1 = "stablehlo.reduce_scatter"(%0)
  // CHECK: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>}
  %0 = "stablehlo.reduce_scatter"(%arg0) ({
    ^bb0(%arg3: tensor<i64>, %arg4: tensor<i64>):
    %0 = "stablehlo.add"(%arg3, %arg4) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    "stablehlo.return"(%0) : (tensor<i64>) -> ()
  }) {
    scatter_dimension = 1 : i64,
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x":(1)2}]>]>
  } : (tensor<2x4xi64>) -> tensor<2x2xi64>
  // CHECK-NEXT: %2 = sdy.reshard %1 <@mesh, [{"y"}, {"x":(1)2}]> : tensor<2x2xi64>
  // CHECK-NEXT: return %2 : tensor<2x2xi64>
  return %0 : tensor<2x2xi64>
}
