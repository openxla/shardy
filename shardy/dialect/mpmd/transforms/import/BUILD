# The MPMD import passes and pipeline.

# load("@rules_cc//cc:cc_library.bzl", "cc_library")
# load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(default_visibility = ["//visibility:public"])

td_library(
    name = "passes_td_files",
    srcs = [
        "passes.td",
    ],
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

gentbl_cc_library(
    name = "passes_inc",
    tbl_outs = {
        "passes.h.inc": [
            "-gen-pass-decls",
            "-name=MpmdImport",
        ],
        "g3doc/mpmd_import_passes.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [":passes_td_files"],
)

cc_library(
    name = "passes",
    srcs = [
        "add_sdy_meshes_from_topology_pass.cc",
        "copy_topology_from_main.cc",
        "enforce_equisharding.cc",
        "generate_sdy_meshes_from_topology_pass.cc",
        "import_pipeline.cc",
        "infer_mesh_assignment.cc",
        "infer_mesh_validation.cc",
        "insert_nameless_clones_of_negligible_ops.cc",
        "introduce_transfers.cc",
        "map_input_output_to_mesh.cc",
        "map_named_ops_to_mpmd_ops.cc",
        "remove_internal_sdy_meshes.cc",
        "rename_meshes_pass.cc",
        "simplify_named_computations.cc",
        "validate_named_ops_in_mpmd_func.cc",
    ],
    hdrs = [
        "infer_mesh_assignment.h",
        "passes.h",
    ],
    deps = [
        ":mesh_assignment_map",
        ":mesh_inference_origins",
        ":mesh_inference_utils",
        ":meshes_with_origins",
        ":passes_inc",
        ":sharding_constraints",
        "//shardy/common:logging",
        "//shardy/dialect/mpmd/ir:dialect",
        "//shardy/dialect/mpmd/transforms/common:distributed_function_pass",
        "//shardy/dialect/mpmd/transforms/common:passes",
        "//shardy/dialect/mpmd/transforms/common:simplify_region_op_base",
        "//shardy/dialect/mpmd/transforms/common:utils",
        "//shardy/dialect/sdy/ir:dialect",
        "//shardy/dialect/sdy/transforms/common:sharding_walker",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:stablehlo_passes_optimization",
    ],
)

cc_library(
    name = "mesh_assignment_map",
    srcs = ["mesh_assignment_map.cc"],
    hdrs = ["mesh_assignment_map.h"],
    deps = ["@llvm-project//llvm:Support"],
)

cc_test(
    name = "enforce_equisharding_test",
    srcs = ["enforce_equisharding_test.cc"],
    deps = [
        ":passes",
        ":sharding_constraints",
        "//shardy/common:logging",
        "//shardy/dialect/mpmd/ir:dialect",
        "//shardy/dialect/mpmd/ir:register",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "sharding_constraints",
    srcs = ["sharding_constraints.cc"],
    hdrs = ["sharding_constraints.h"],
    deps = ["@llvm-project//llvm:Support"],
)

cc_library(
    name = "mesh_inference_utils",
    srcs = ["mesh_inference_utils.cc"],
    hdrs = ["mesh_inference_utils.h"],
    deps = [
        ":meshes_with_origins",
        "//shardy/dialect/mpmd/ir:dialect",
        "//shardy/dialect/mpmd/ir:fragment_arg_res_attrs",
        "//shardy/dialect/mpmd/transforms/common:utils",
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "mesh_inference_origins",
    srcs = ["mesh_inference_origins.cc"],
    hdrs = ["mesh_inference_origins.h"],
    deps = [
        "//shardy/common:logging",
        "//shardy/dialect/mpmd/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "mesh_inference_utils_test",
    srcs = ["mesh_inference_utils_test.cc"],
    deps = [
        ":mesh_inference_utils",
        ":meshes_with_origins",
        "//shardy/dialect/mpmd/ir:dialect",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "meshes_with_origins",
    srcs = ["meshes_with_origins.cc"],
    hdrs = ["meshes_with_origins.h"],
    deps = [
        ":mesh_inference_origins",
        "//shardy/common:logging",
        "//shardy/dialect/mpmd/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "meshes_with_origins_test",
    srcs = ["meshes_with_origins_test.cc"],
    deps = [
        ":mesh_inference_origins",
        ":meshes_with_origins",
        "//shardy/dialect/mpmd/ir:dialect",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
