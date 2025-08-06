# Shardy tools.

# load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "sdy_opt",
    srcs = ["sdy_opt_main.cc"],
    deps = [
        "//shardy/dialect/sdy/ir:register",
        "//shardy/dialect/sdy/transforms:passes",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:RegisterAllPasses",
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

cc_binary(
    name = "mpmd_opt",
    srcs = ["mpmd_opt_main.cc"],
    deps = [
        "//shardy/dialect/mpmd/ir:register",
        "//shardy/dialect/mpmd/transforms:passes",
        "//shardy/dialect/sdy/transforms:passes",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:RegisterAllPasses",
    ],
)
