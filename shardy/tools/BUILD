# Shardy tools.

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "sdy_opt",
    srcs = ["sdy_opt_main.cc"],
    deps = [
        "//shardy/dialect/sdy/ir:register",
        "//shardy/dialect/sdy/transforms:passes",
        "//shardy/round_trip_import:import_sdy_custom_calls",
        "//shardy/round_trip_import:import_shardy_attrs",
        "//shardy/round_trip_import:import_uninlineable_func_calls",
        "//shardy/round_trip_import:pipelines",
        "//shardy/round_trip_import:shard_map_import",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:QuantOps",
    ],
)

cc_binary(
    name = "sdy_translate",
    srcs = ["sdy_translate_main.cc"],
    deps = [
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/ir:register",
        "//shardy/dialect/sdy/transforms:passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:TranslateLib",
    ],
)
