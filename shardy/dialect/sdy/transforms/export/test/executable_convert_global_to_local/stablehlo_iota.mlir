// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

// Performs the same iota as in sequential_iota, but on 2 devices in parallel.
//
// We will use sdy-opt to convert this to a device local program that
// stablehlo interpreter can execute.
//
func.func @parallel_iota()
  -> (tensor<2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) {
  %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}]>]>} : tensor<2xi32>
  return %0 : tensor<2xi32>
}

//--- part2.mlir

// Main Orchestrator: executes the sequential and parallel iota and checks that
// they are equivalent.
func.func @main() {
  %seq = func.call @sequential_iota() : () -> tensor<2xi32>

  %pars:2 = "interpreter.run_parallel"() {
    programs = [[@parallel_iota, @parallel_iota]]
  } : () -> (tensor<1xi32>, tensor<1xi32>)
  %par = "stablehlo.concatenate"(%pars#0, %pars#1) {
    dimension = 0 : i64
  } : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>

  "check.expect_eq"(%seq, %par) : (tensor<2xi32>, tensor<2xi32>) -> ()

  return
}
