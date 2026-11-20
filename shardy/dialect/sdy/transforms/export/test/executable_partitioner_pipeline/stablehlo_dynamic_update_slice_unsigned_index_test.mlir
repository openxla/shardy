// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_x2 = <["x"=2]>

// Unsigned ui8 constant 128 on an evenly divisible 260 element sharded
// dimension (shard size 130) crosses shard boundary 130 for a 4 element update.
func.func @parallel_dus_divisible_u8_msb(%arg0: tensor<260xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}]>}, %upd: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}]>}) -> (tensor<260xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}]>}) {
  %c128 = stablehlo.constant dense<128> : tensor<ui8>
  %sharded = sdy.reshard %arg0 <@mesh_x2, [{"x"}]> : tensor<260xi32>
  %0 = stablehlo.dynamic_update_slice %sharded, %upd, %c128 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}]>]>} : (tensor<260xi32>, tensor<4xi32>, tensor<ui8>) -> tensor<260xi32>
  %neg = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}]>]>} : tensor<260xi32>
  %1 = sdy.reshard %neg <@mesh_x2, [{}]> : tensor<260xi32>
  return %1 : tensor<260xi32>
}

func.func @sequential_dus_divisible_u8_msb(%arg0: tensor<260xi32>, %upd: tensor<4xi32>) -> tensor<260xi32> {
  %c128 = stablehlo.constant dense<128> : tensor<ui8>
  %0 = stablehlo.dynamic_update_slice %arg0, %upd, %c128 : (tensor<260xi32>, tensor<4xi32>, tensor<ui8>) -> tensor<260xi32>
  %neg = stablehlo.negate %0 : tensor<260xi32>
  return %neg : tensor<260xi32>
}

// Unsigned ui64 constant UINT64_MAX clamps to 2 on an indivisible 5 element
// sharded dimension (shard size 3), crossing shard boundary 3 for size 3.
func.func @parallel_dus_indivisible_u64_max(%arg0: tensor<5xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}]>}, %upd: tensor<3xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}]>}) -> (tensor<5xi32> {sdy.sharding = #sdy.sharding<@mesh_x2, [{}]>}) {
  %cmax = stablehlo.constant dense<18446744073709551615> : tensor<ui64>
  %sharded = sdy.reshard %arg0 <@mesh_x2, [{"x"}]> : tensor<5xi32>
  %0 = stablehlo.dynamic_update_slice %sharded, %upd, %cmax {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}]>]>} : (tensor<5xi32>, tensor<3xi32>, tensor<ui64>) -> tensor<5xi32>
  %neg = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2, [{"x"}]>]>} : tensor<5xi32>
  %1 = sdy.reshard %neg <@mesh_x2, [{}]> : tensor<5xi32>
  return %1 : tensor<5xi32>
}

func.func @sequential_dus_indivisible_u64_max(%arg0: tensor<5xi32>, %upd: tensor<3xi32>) -> tensor<5xi32> {
  %cmax = stablehlo.constant dense<18446744073709551615> : tensor<ui64>
  %0 = stablehlo.dynamic_update_slice %arg0, %upd, %cmax : (tensor<5xi32>, tensor<3xi32>, tensor<ui64>) -> tensor<5xi32>
  %neg = stablehlo.negate %0 : tensor<5xi32>
  return %neg : tensor<5xi32>
}

//--- part2.mlir

func.func @main() {
  %input1 = stablehlo.iota dim = 0 : tensor<260xi32>
  %upd1 = stablehlo.constant dense<[10, 20, 30, 40]> : tensor<4xi32>
  %seq1 = func.call @sequential_dus_divisible_u8_msb(%input1, %upd1) : (tensor<260xi32>, tensor<4xi32>) -> tensor<260xi32>
  %res1:2 = "interpreter.run_parallel"(%input1, %upd1, %input1, %upd1) {programs = [[@parallel_dus_divisible_u8_msb, @parallel_dus_divisible_u8_msb]]} : (tensor<260xi32>, tensor<4xi32>, tensor<260xi32>, tensor<4xi32>) -> (tensor<260xi32>, tensor<260xi32>)
  "check.expect_eq"(%res1#0, %seq1) : (tensor<260xi32>, tensor<260xi32>) -> ()
  "check.expect_eq"(%res1#1, %seq1) : (tensor<260xi32>, tensor<260xi32>) -> ()

  %input2 = stablehlo.constant dense<[5, 6, 7, 8, 9]> : tensor<5xi32>
  %upd2 = stablehlo.constant dense<[11, 22, 33]> : tensor<3xi32>
  %seq2 = func.call @sequential_dus_indivisible_u64_max(%input2, %upd2) : (tensor<5xi32>, tensor<3xi32>) -> tensor<5xi32>
  %res2:2 = "interpreter.run_parallel"(%input2, %upd2, %input2, %upd2) {programs = [[@parallel_dus_indivisible_u64_max, @parallel_dus_indivisible_u64_max]]} : (tensor<5xi32>, tensor<3xi32>, tensor<5xi32>, tensor<3xi32>) -> (tensor<5xi32>, tensor<5xi32>)
  "check.expect_eq"(%res2#0, %seq2) : (tensor<5xi32>, tensor<5xi32>) -> ()
  "check.expect_eq"(%res2#1, %seq2) : (tensor<5xi32>, tensor<5xi32>) -> ()
  return
}
