// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_4 = <["x"=4]>

func.func @parallel_iota_bf16()
  -> (tensor<1028xbf16> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}]>}) {
  %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>} : tensor<1028xbf16>
  return %0 : tensor<1028xbf16>
}

func.func @parallel_iota_f16()
  -> (tensor<65516xf16> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}]>}) {
  %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>} : tensor<65516xf16>
  return %0 : tensor<65516xf16>
}

func.func @parallel_iota_f8e4m3fn()
  -> (tensor<456xf8E4M3FN> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}]>}) {
  %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>} : tensor<456xf8E4M3FN>
  return %0 : tensor<456xf8E4M3FN>
}

func.func @parallel_iota_f8e8m0fnu()
  -> (tensor<8xf8E8M0FNU> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}]>}) {
  %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>} : tensor<8xf8E8M0FNU>
  return %0 : tensor<8xf8E8M0FNU>
}

//--- part2.mlir

func.func @main() {
  %seq_bf16 = func.call @sequential_iota_bf16() : () -> tensor<1028xbf16>
  %pars_bf16:4 = "interpreter.run_parallel"() {
    programs = [[@parallel_iota_bf16, @parallel_iota_bf16, @parallel_iota_bf16, @parallel_iota_bf16]]
  } : () -> (tensor<257xbf16>, tensor<257xbf16>, tensor<257xbf16>, tensor<257xbf16>)
  %par_bf16 = "stablehlo.concatenate"(%pars_bf16#0, %pars_bf16#1, %pars_bf16#2, %pars_bf16#3) {
    dimension = 0 : i64
  } : (tensor<257xbf16>, tensor<257xbf16>, tensor<257xbf16>, tensor<257xbf16>) -> tensor<1028xbf16>
  "check.expect_eq"(%seq_bf16, %par_bf16) : (tensor<1028xbf16>, tensor<1028xbf16>) -> ()

  %seq_f16 = func.call @sequential_iota_f16() : () -> tensor<65516xf16>
  %pars_f16:4 = "interpreter.run_parallel"() {
    programs = [[@parallel_iota_f16, @parallel_iota_f16, @parallel_iota_f16, @parallel_iota_f16]]
  } : () -> (tensor<16379xf16>, tensor<16379xf16>, tensor<16379xf16>, tensor<16379xf16>)
  %par_f16 = "stablehlo.concatenate"(%pars_f16#0, %pars_f16#1, %pars_f16#2, %pars_f16#3) {
    dimension = 0 : i64
  } : (tensor<16379xf16>, tensor<16379xf16>, tensor<16379xf16>, tensor<16379xf16>) -> tensor<65516xf16>
  "check.expect_eq"(%seq_f16, %par_f16) : (tensor<65516xf16>, tensor<65516xf16>) -> ()

  %seq_f8 = func.call @sequential_iota_f8e4m3fn() : () -> tensor<456xf8E4M3FN>
  %pars_f8:4 = "interpreter.run_parallel"() {
    programs = [[@parallel_iota_f8e4m3fn, @parallel_iota_f8e4m3fn, @parallel_iota_f8e4m3fn, @parallel_iota_f8e4m3fn]]
  } : () -> (tensor<114xf8E4M3FN>, tensor<114xf8E4M3FN>, tensor<114xf8E4M3FN>, tensor<114xf8E4M3FN>)
  %par_f8 = "stablehlo.concatenate"(%pars_f8#0, %pars_f8#1, %pars_f8#2, %pars_f8#3) {
    dimension = 0 : i64
  } : (tensor<114xf8E4M3FN>, tensor<114xf8E4M3FN>, tensor<114xf8E4M3FN>, tensor<114xf8E4M3FN>) -> tensor<456xf8E4M3FN>
  "check.expect_eq"(%seq_f8, %par_f8) : (tensor<456xf8E4M3FN>, tensor<456xf8E4M3FN>) -> ()

  %seq_e8m0 = func.call @sequential_iota_f8e8m0fnu() : () -> tensor<8xf8E8M0FNU>
  %pars_e8m0:4 = "interpreter.run_parallel"() {
    programs = [[@parallel_iota_f8e8m0fnu, @parallel_iota_f8e8m0fnu, @parallel_iota_f8e8m0fnu, @parallel_iota_f8e8m0fnu]]
  } : () -> (tensor<2xf8E8M0FNU>, tensor<2xf8E8M0FNU>, tensor<2xf8E8M0FNU>, tensor<2xf8E8M0FNU>)
  %par_e8m0 = "stablehlo.concatenate"(%pars_e8m0#0, %pars_e8m0#1, %pars_e8m0#2, %pars_e8m0#3) {
    dimension = 0 : i64
  } : (tensor<2xf8E8M0FNU>, tensor<2xf8E8M0FNU>, tensor<2xf8E8M0FNU>, tensor<2xf8E8M0FNU>) -> tensor<8xf8E8M0FNU>
  %seq_e8m0_bits = stablehlo.bitcast_convert %seq_e8m0 : (tensor<8xf8E8M0FNU>) -> tensor<8xi8>
  %par_e8m0_bits = stablehlo.bitcast_convert %par_e8m0 : (tensor<8xf8E8M0FNU>) -> tensor<8xi8>
  "check.expect_eq"(%seq_e8m0_bits, %par_e8m0_bits) : (tensor<8xi8>, tensor<8xi8>) -> ()

  return
}
