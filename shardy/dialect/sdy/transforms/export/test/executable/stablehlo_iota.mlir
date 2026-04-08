// RUN: split-file %s %t

// 1. Generate the partitioned (localized) version of the function.
// RUN: sdy_opt %t/part1.mlir --sdy-convert-global-to-local --allow-unregistered-dialect | \
// RUN: sdy_opt --sdy-drop-sharding-and-mesh --allow-unregistered-dialect > %t/part1_lowered.mlir

// 2. Generate the sequential reference by stripping shardings and renaming the function.
// RUN: sdy_opt %t/part1.mlir --sdy-drop-sharding-and-mesh --allow-unregistered-dialect | \
// RUN: sed 's/parallel_iota/sequential_iota/' > %t/part1_sequential.mlir

// 3. Assemble the final module by unwrapping the bodies of both generated modules.
// RUN: sed '1d; /^}/,$d' %t/part1_lowered.mlir > %t/combined.mlir
// RUN: sed '1d; /^}/,$d' %t/part1_sequential.mlir >> %t/combined.mlir
// RUN: cat %t/part2.mlir >> %t/combined.mlir
// RUN: cat %t/combined.mlir

// 4. Execute the combined test.
// RUN: stablehlo-translate --interpret %t/combined.mlir


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
