package(default_visibility = ["//visibility:public"])

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
    deps = ["@llvm-project//llvm:Support"],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        ":constants",
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/ir:register",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "import_sdy_custom_calls",
    srcs = ["import_sdy_custom_calls.cc"],
    hdrs = ["import_sdy_custom_calls.h"],
    deps = [
        ":constants",
        ":utils",
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "import_backend_func_calls",
    srcs = ["import_backend_func_calls.cc"],
    hdrs = ["import_backend_func_calls.h"],
    deps = [
        ":constants",
        ":utils",
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "import_constants",
    srcs = ["import_constants.cc"],
    hdrs = ["import_constants.h"],
    deps = [
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "open_while_free_vars_sharding",
    srcs = ["open_while_free_vars_sharding.cc"],
    hdrs = ["open_while_free_vars_sharding.h"],
    deps = [
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "import_shardy_attrs",
    srcs = ["import_shardy_attrs.cc"],
    hdrs = ["import_shardy_attrs.h"],
    deps = [
        "constants",
        ":utils",
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "remove_size_one_axes",
    srcs = ["remove_size_one_axes.cc"],
    hdrs = ["remove_size_one_axes.h"],
    deps = [
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/transforms/common:sharding_walker",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "import_callback_custom_calls",
    srcs = ["import_callback_custom_calls.cc"],
    hdrs = ["import_callback_custom_calls.h"],
    deps = [
        ":utils",
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "shard_map_import",
    srcs = ["shard_map_import.cc"],
    hdrs = ["shard_map_import.h"],
    deps = [
        ":constants",
        ":utils",
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "pipelines",
    srcs = ["pipelines.cc"],
    hdrs = ["pipelines.h"],
    deps = [
        ":import_backend_func_calls",
        ":import_callback_custom_calls",
        ":import_constants",
        ":import_sdy_custom_calls",
        ":import_shardy_attrs",
        ":open_while_free_vars_sharding",
        ":remove_size_one_axes",
        ":shard_map_import",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:stablehlo_passes",
    ],
)
