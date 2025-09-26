/* Copyright 2025 The MPMD Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "shardy/dialect/mpmd/ir/utils.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ScopedPrinter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/register.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::mlir::func::FuncOp;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::FieldsAre;
using ::testing::IsEmpty;
using ::testing::Optional;

namespace mlir::mpmd {
namespace {

std::optional<std::string> GetMeshAttrString(Operation* op) {
  FailureOr<sdy::MeshAttr> mesh_attr = GetMeshAttr(op);
  if (failed(mesh_attr)) {
    return std::nullopt;
  }
  return llvm::to_string(*mesh_attr);
}

TEST(ExtractFunctionIOShardingSpecsAndMeshes, FunctionWithSingleTransfer) {
  const std::string kProgram = R"mlir(
    func.func @main(%arg0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>
        {mhlo.memory_kind = "arg"})
        -> (!mpmd.mesh_tensor<"mesh2", tensor<12x16xf32>, sharding=<@mesh, [{"x"}, {?}]>> {mhlo.memory_kind = "res"}) attributes {
        "topology"=#mpmd.topology<
          <"mesh1": <["x"=2, "y"=4]>>,
          <"mesh2": <["x"=2, "y"=4]>>
        >} {
      %0 = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>) -> !mpmd.mesh_tensor<"mesh2", tensor<12x16xf32>, sharding=<@mesh, [{"x"}, {?}]>>
      func.return %0 : !mpmd.mesh_tensor<"mesh2", tensor<12x16xf32>, sharding=<@mesh, [{"x"}, {?}]>>
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  SDY_CHECK(module);
  auto func_op = GetMainFunction(*module);
  SDY_CHECK(func_op);

  FunctionIOShardingSpecsAndMeshes specs_and_mesh =
      ExtractFunctionIOShardingSpecsAndMeshes(func_op);

  EXPECT_THAT(specs_and_mesh.input_specs,
              ElementsAre(FieldsAre("mesh1", SpmdTensorPartitionSpec{{}, {"y"}},
                                    "arg")));
  EXPECT_THAT(specs_and_mesh.output_specs,
              ElementsAre(FieldsAre("mesh2", SpmdTensorPartitionSpec{{"x"}, {}},
                                    "res")));
}

TEST(ExtractFunctionIOShardingSpecsAndMeshes, MultipleInputMultipleOutput) {
  const std::string kProgram = R"mlir(
    func.func @main(
      %arg0 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>,
      %arg1 : !mpmd.mesh_tensor<"mesh1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>
    ) -> (!mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>,
          !mpmd.mesh_tensor<"mesh1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>)
      attributes {
        "topology"=#mpmd.topology<
          <"mesh1": <["x"=2, "y"=4]>>
        >
      }
    {
      func.return %arg0, %arg1 : !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>,
                                 !mpmd.mesh_tensor<"mesh1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  SDY_CHECK(module);
  auto func_op = GetMainFunction(*module);
  SDY_CHECK(func_op);

  FunctionIOShardingSpecsAndMeshes specs_and_mesh =
      ExtractFunctionIOShardingSpecsAndMeshes(func_op);
  EXPECT_THAT(specs_and_mesh.input_specs,
              ElementsAre(FieldsAre("mesh1", SpmdTensorPartitionSpec{{}, {"y"}},
                                    std::nullopt),
                          FieldsAre("mesh1", SpmdTensorPartitionSpec{{"x"}, {}},
                                    std::nullopt)));
  EXPECT_THAT(specs_and_mesh.output_specs,
              ElementsAre(FieldsAre("mesh1", SpmdTensorPartitionSpec{{}, {"y"}},
                                    std::nullopt),
                          FieldsAre("mesh1", SpmdTensorPartitionSpec{{"x"}, {}},
                                    std::nullopt)));
}

TEST(ExtractFunctionIOShardingSpecsAndMeshes, IOTypesHaveMemoryKinds) {
  const std::string kProgram = R"mlir(
    !input_type = !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, memory_kind="device">
    !output_type = !mpmd.mesh_tensor<"mesh1", tensor<12x16xf32>, memory_kind="pinned_host">
    func.func @main(%arg0 : !input_type {mhlo.memory_kind = "will_be_ignored"}) -> !output_type attributes {
        "topology"=#mpmd.topology<<"mesh1": <["x"=2, "y"=4]>> >} {
      %0 = mpmd.transfer %arg0 : (!input_type) -> !output_type
      func.return %0 : !output_type
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  SDY_CHECK(module);
  auto func_op = GetMainFunction(*module);

  FunctionIOShardingSpecsAndMeshes specs_and_mesh =
      ExtractFunctionIOShardingSpecsAndMeshes(func_op);

  EXPECT_THAT(
      specs_and_mesh.input_specs,
      ElementsAre(FieldsAre("mesh1", SpmdTensorPartitionSpec{}, "device")));

  EXPECT_THAT(specs_and_mesh.output_specs,
              ElementsAre(FieldsAre("mesh1", SpmdTensorPartitionSpec{},
                                    "pinned_host")));
}

SmallVector<SmallVector<OpResult>> GetCallOpResults(FuncOp func_op) {
  SmallVector<SmallVector<OpResult>> produced_values;
  for (CallOp op : func_op.getBody().getOps<CallOp>()) {
    produced_values.push_back(llvm::to_vector(op.getResults()));
  }
  return produced_values;
}

TEST(GetMeshAttrTest, NotInTheScopeOfAFragment) {
  MLIRContext context;
  loadAllRequiredDialects(&context);

  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {
      mesh_shape = #sdy.mesh<["x"=2]>
    } {
      %0 = stablehlo.add %arg0, %arg0 {focus} : tensor<16xf32>
      func.return %0 : tensor<16xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  Operation* op = FindAnnotatedOperation(*module, "focus");
  SDY_CHECK(op);
  EXPECT_THAT(GetMeshAttrString(op), Eq("#sdy.mesh<[\"x\"=2]>"));
}

TEST(GetMeshAttrTest, OperationMissingMeshShape) {
  MLIRContext context;
  loadAllRequiredDialects(&context);

  // This program is missing the mesh_shape attribute.
  const std::string program = R"mlir(
  func.func @main(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {
  } {
    %0 = stablehlo.add %arg0, %arg0 {focus} : tensor<16xf32>
    func.return %0 : tensor<16xf32>
  })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);

  Operation* op = FindAnnotatedOperation(*module, "focus");
  SDY_CHECK(op);

  EXPECT_THAT(GetMeshAttrString(op), Eq(std::nullopt));
}

TEST(GetMeshAttrTest, ValueNotInTheScopeOfAFragment) {
  MLIRContext context;
  loadAllRequiredDialects(&context);

  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {
      mesh_shape = #sdy.mesh<["x"=2]>
    } {
      %0 = stablehlo.add %arg0, %arg0 {focus} : tensor<16xf32>
      func.return %0 : tensor<16xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  Operation* op = FindAnnotatedOperation(*module, "focus");
  SDY_CHECK(op);
  EXPECT_THAT(GetMeshAttrString(op), Eq("#sdy.mesh<[\"x\"=2]>"));
}

TEST(GetMeshAttrTest, NotInTheScopeOfAFragmentWithMeshAndTopology) {
  MLIRContext context;
  loadAllRequiredDialects(&context);

  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {
      mesh_shape = #sdy.mesh<["x"=2]>,
      topology = #mpmd.topology<
          <"mesh1": <["y"=2]>>
        >
    } {
      %0 = stablehlo.add %arg0, %arg0 {focus} : tensor<16xf32>
      func.return %0 : tensor<16xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  Operation* op = FindAnnotatedOperation(*module, "focus");
  SDY_CHECK(op);
  // Expects the mesh to be the one in mesh_shape.
  EXPECT_THAT(GetMeshAttrString(op), Eq("#sdy.mesh<[\"x\"=2]>"));
}

TEST(GetMeshAttrTest, InTheScopeOfAFragment) {
  MLIRContext context;
  loadAllRequiredDialects(&context);

  const std::string program = R"mlir(
    func.func @main(%arg0: !mpmd.mesh_tensor<"mesh2", tensor<16xf32>>)
      -> !mpmd.mesh_tensor<"mesh2", tensor<16xf32>> attributes {
        topology = #mpmd.topology<
          <"mesh1": <["x"=2]>>,
          <"mesh2": <["x"=4]>>,
          <"mesh3": <["y"=2]>>
        >
    } {
      %1 = mpmd.fragment<mesh="mesh2", origin=["f1"]> (%arg0) (%arg1: tensor<16xf32>) {
        %0 = stablehlo.add %arg1, %arg1 {focus} : tensor<16xf32>
        mpmd.return %0 : tensor<16xf32>
      } : (!mpmd.mesh_tensor<"mesh2", tensor<16xf32>>)
       -> !mpmd.mesh_tensor<"mesh2", tensor<16xf32>>
      func.return %1 : !mpmd.mesh_tensor<"mesh2", tensor<16xf32>>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  Operation* op = FindAnnotatedOperation(*module, "focus");
  SDY_CHECK(op);
  EXPECT_THAT(GetMeshAttrString(op), Eq("#sdy.mesh<[\"x\"=4]>"));
}

// This test is different from InTheScopeOfAFragment because it contains both
// a topology and a mesh_shape in the function's attributes.
TEST(GetMeshAttrTest, InTheScopeOfAFragmentWithTopologyAndMesh) {
  MLIRContext context;
  loadAllRequiredDialects(&context);

  const std::string program = R"mlir(
    func.func @main(%arg0: !mpmd.mesh_tensor<"mesh1", tensor<16xf32>>)
      -> !mpmd.mesh_tensor<"mesh1", tensor<16xf32>> attributes {
        topology = #mpmd.topology<
          <"mesh1": <["x"=2]>>
        >,
        mesh_shape = #sdy.mesh<["y"=4]>
    } {
      %1 = mpmd.fragment<mesh="mesh1", origin=["f"]> (%arg0) (%arg1: tensor<16xf32>) {
        %0 = stablehlo.add %arg1, %arg1 {focus} : tensor<16xf32>
        mpmd.return %0 : tensor<16xf32>
      } : (!mpmd.mesh_tensor<"mesh1", tensor<16xf32>>)
       -> !mpmd.mesh_tensor<"mesh1", tensor<16xf32>>
      func.return %1 : !mpmd.mesh_tensor<"mesh1", tensor<16xf32>>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  Operation* op = FindAnnotatedOperation(*module, "focus");
  SDY_CHECK(op);
  // Expects the mesh to be the one in topology.
  EXPECT_THAT(GetMeshAttrString(op), Eq("#sdy.mesh<[\"x\"=2]>"));
}

TEST(GetMeshAttrTest, OperationMissingMeshShapeAndTopology) {
  MLIRContext context;
  loadAllRequiredDialects(&context);

  // This program is missing both the mesh_shape and topology attributes.
  const std::string program = R"mlir(
  func.func @main(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {
  } {
    %0 = stablehlo.add %arg0, %arg0 {focus} : tensor<16xf32>
    func.return %0 : tensor<16xf32>
  })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);

  Operation* op = FindAnnotatedOperation(*module, "focus");
  SDY_CHECK(op);

  EXPECT_THAT(GetMeshAttrString(op), Eq(std::nullopt));
}

TEST(GetMeshAttrTest, ValueInTheScopeOfAFragment) {
  MLIRContext context;
  loadAllRequiredDialects(&context);

  const std::string program = R"mlir(
    func.func @main(%arg0: !mpmd.mesh_tensor<"mesh1", tensor<16xf32>>)
      -> !mpmd.mesh_tensor<"mesh1", tensor<16xf32>> attributes {
        topology = #mpmd.topology<
          <"mesh1": <["x"=2]>>,
          <"mesh2": <["x"=4]>>,
          <"mesh3": <["y"=2]>>
        >
    } {
      %1 = mpmd.fragment<mesh="mesh1", origin=["f1"]> (%arg0) (%arg1: tensor<16xf32>) {
        %0 = stablehlo.add %arg1, %arg1 {focus} : tensor<16xf32>
        mpmd.return %0 : tensor<16xf32>
      } : (!mpmd.mesh_tensor<"mesh1", tensor<16xf32>>)
       -> !mpmd.mesh_tensor<"mesh1", tensor<16xf32>>
      func.return %1 : !mpmd.mesh_tensor<"mesh1", tensor<16xf32>>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  Operation* op = FindAnnotatedOperation(*module, "focus");
  SDY_CHECK(op);
  EXPECT_THAT(GetMeshAttrString(op), Eq("#sdy.mesh<[\"x\"=2]>"));
}

TEST(SetTopology, WhenUndefined) {
  MLIRContext context;
  loadAllRequiredDialects(&context);

  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
      %1 = stablehlo.add %arg0, %arg0 : tensor<16xf32>
      func.return %1 : tensor<16xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  auto main_fn = cast<FuncOp>(module->lookupSymbol("main"));
  std::vector<std::pair<std::string, FlatMesh>> topology_shape = {
      {"mesh1", {{"x", 2}, {"y", 4}}},
      {"mesh2", {{"x", 8}}},
  };
  SetTopology(topology_shape, main_fn);
  SDY_CHECK(main_fn->hasAttr("topology"));
  auto topology_attr = cast<TopologyAttr>(main_fn->getAttr("topology"));

  // Expected meshes in the topology.
  std::vector<sdy::MeshAxisAttr> mesh1 = {
      sdy::MeshAxisAttr::get(&context, "x", 2),
      sdy::MeshAxisAttr::get(&context, "y", 4)};
  std::vector<sdy::MeshAxisAttr> mesh2 = {
      sdy::MeshAxisAttr::get(&context, "x", 8)};
  EXPECT_THAT(
      topology_attr.getMeshes(),
      ElementsAre(NamedMeshAttr::get(&context, "mesh1",
                                     sdy::MeshAttr::get(&context, mesh1)),
                  NamedMeshAttr::get(&context, "mesh2",
                                     sdy::MeshAttr::get(&context, mesh2))));
}

TEST(SetTopology, WhenDefined) {
  MLIRContext context;
  loadAllRequiredDialects(&context);

  const std::string program = R"mlir(
    func.func @main(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> attributes {
      topology = #mpmd.topology<<"mesh1": <["a"=4, "b"=2]>>, <"mesh2": <["b"=2]>>>
    } {
      %1 = stablehlo.add %arg0, %arg0 : tensor<16xf32>
      func.return %1 : tensor<16xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  auto main_fn = cast<FuncOp>(module->lookupSymbol("main"));
  std::vector<std::pair<std::string, FlatMesh>> topology_shape = {
      {"m", {{"x", 8}}}};
  SetTopology(topology_shape, main_fn);
  SDY_CHECK(main_fn->hasAttr("topology"));
  auto topology_attr = cast<TopologyAttr>(main_fn->getAttr("topology"));

  // Expected mesh in the topology.
  std::vector<sdy::MeshAxisAttr> mesh = {
      sdy::MeshAxisAttr::get(&context, "x", 8)};
  EXPECT_THAT(topology_attr.getMeshes(),
              ElementsAre(NamedMeshAttr::get(
                  &context, "m", sdy::MeshAttr::get(&context, mesh))));
}

using Sources = SmallVector<OpOperand*>;
using Targets = SmallVector<Value>;

TEST(GetMpmdDataflowEdge, FuncOp) {
  const std::string kProgram = R"mlir(
    module {
      func.func public @main(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>, %arg2: tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>) attributes {topology = #mpmd.topology<<"mesh1" : <["x"=1]>>, <"mesh2" : <["x"=1]>>>} {
        %0:2 = mpmd.call @f(%arg0, %arg1, %arg2) : (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)
        %1:2 = mpmd.call @f(%arg0, %arg1, %arg2) : (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)
        return %0#0, %1#1 : tensor<3x5xf32>, tensor<3x5xf32>
      }
      func.func private @f(%arg3: tensor<3x5xf32>, %arg4: tensor<3x5xf32>, %arg5: tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>) attributes {topology = #mpmd.topology<<"mesh1" : <["x"=1]>>, <"mesh2" : <["x"=1]>>>} {
        %2 = stablehlo.add %arg3, %arg4 : tensor<3x5xf32>
        %3 = stablehlo.add %2, %arg5 : tensor<3x5xf32>
        return %2, %3 : tensor<3x5xf32>, tensor<3x5xf32>
      }
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  SDY_CHECK(module);
  auto main_func = GetMainFunction(*module);
  SDY_CHECK(main_func);

  EXPECT_THAT(GetMpmdDataflowEdges(main_func), IsEmpty());

  FuncOp private_func = dyn_cast_or_null<FuncOp>(module->lookupSymbol("f"));
  SDY_CHECK(private_func);

  auto return_op =
      cast<func::ReturnOp>(private_func.getBody().front().getTerminator());
  SmallVector<SmallVector<OpResult>> call_results = GetCallOpResults(main_func);
  auto call_ops = llvm::to_vector(main_func.getBody().getOps<CallOp>());
  CallOp c0 = call_ops[0];
  CallOp c1 = call_ops[1];

  EXPECT_THAT(GetMpmdDataflowEdges(private_func),
              ElementsAre(
                  // {%2} -> {%0#0, %1#0}
                  FieldsAre(Sources{&return_op->getOpOperand(0)},
                            Targets{call_results[0][0], call_results[1][0]}),
                  // {%3} -> {%0#1, %1#1}
                  FieldsAre(Sources{&return_op->getOpOperand(1)},
                            Targets{call_results[0][1], call_results[1][1]}),
                  // {%arg0 use 0, %arg0 use 1} -> {%arg3}
                  FieldsAre(Sources{&c0->getOpOperand(0), &c1->getOpOperand(0)},
                            Targets{private_func.getArgument(0)}),
                  // {%arg1 use 0, %arg1 use 1} -> {%arg4}
                  FieldsAre(Sources{&c0->getOpOperand(1), &c1->getOpOperand(1)},
                            Targets{private_func.getArgument(1)}),
                  // {%arg2 use 0, %arg2 use 1} -> {%arg5}
                  FieldsAre(Sources{&c0->getOpOperand(2), &c1->getOpOperand(2)},
                            Targets{private_func.getArgument(2)})));
}

TEST(TryToFindSingleTransposeCount, NoTransposeCount) {
  const std::string kProgram = R"mlir(
    !mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
      -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0)(%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      return %0 : !mesh_1_tensor_4_8_f32
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  SDY_CHECK(module);
  auto main_func = GetMainFunction(*module);
  SDY_CHECK(main_func);
  FragmentOp fragment_op = cast<FragmentOp>(*main_func.getOps().begin());
  EXPECT_THAT(TryToFindSingleTransposeCount(fragment_op), Eq(std::nullopt));
}

TEST(TryToFindSingleTransposeCount, OneTransposeCount) {
  const std::string kProgram = R"mlir(
    !mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
      -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(123)]> (%arg0)(%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      return %0 : !mesh_1_tensor_4_8_f32
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  SDY_CHECK(module);
  auto main_func = GetMainFunction(*module);
  SDY_CHECK(main_func);
  FragmentOp fragment_op = cast<FragmentOp>(*main_func.getOps().begin());
  EXPECT_THAT(TryToFindSingleTransposeCount(fragment_op), Optional(123));
}

TEST(TryToFindSingleTransposeCount, OneTransposeCountInMultipleOrigins) {
  const std::string kProgram = R"mlir(
    !mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
      -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(123), "f2"(123)]> (%arg0)(%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      return %0 : !mesh_1_tensor_4_8_f32
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  SDY_CHECK(module);
  auto main_func = GetMainFunction(*module);
  SDY_CHECK(main_func);
  FragmentOp fragment_op = cast<FragmentOp>(*main_func.getOps().begin());
  EXPECT_THAT(TryToFindSingleTransposeCount(fragment_op), Optional(123));
}

TEST(TryToFindSingleTransposeCount,
     DifferentTransposeCountsShouldReturnNullopt) {
  const std::string kProgram = R"mlir(
    !mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
      -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(123), "f2"(321)]> (%arg0)(%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      return %0 : !mesh_1_tensor_4_8_f32
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  SDY_CHECK(module);
  auto main_func = GetMainFunction(*module);
  SDY_CHECK(main_func);
  FragmentOp fragment_op = cast<FragmentOp>(*main_func.getOps().begin());
  EXPECT_THAT(TryToFindSingleTransposeCount(fragment_op), Eq(std::nullopt));
}

TEST(TryToFindMaxTransposeCount,
     MergedRematFragmentShouldReturnMaxTransposeCount) {
  const std::string kProgram = R"mlir(
    !mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
      -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(123), "f2"(321)]> (%arg0) {remat} (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      return %0 : !mesh_1_tensor_4_8_f32
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  SDY_CHECK(module);
  auto main_func = GetMainFunction(*module);
  SDY_CHECK(main_func);
  FragmentOp fragment_op = cast<FragmentOp>(*main_func.getOps().begin());
  EXPECT_THAT(TryToFindMaxTransposeCount(fragment_op), Eq(321));
}

TEST(IsExecutedImmediatelyAfter,
     ShouldReturnTrueIfBackwardIsImmediatelyAfterForwardFragmentInSameMesh) {
  const char kProgram[] = R"mlir(
    !mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
      -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %forward_result = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      %transfer_result = mpmd.transfer %forward_result : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      %backward_result = mpmd.fragment<mesh="m1", origin=[]> (%forward_result) (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

      return %backward_result : !mesh_1_tensor_4_8_f32
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  Region::OpIterator it =
      func_op.getOps().begin();  // it points to the forward fragment.
  SDY_CHECK(it != func_op.getOps().end());
  FragmentOp fwd_fragment = cast<FragmentOp>(*it);
  SDY_CHECK(it != func_op.getOps().end());
  ++it;  // it points to the transfer op.
  SDY_CHECK(it != func_op.getOps().end());
  ++it;  // it points to the backward fragment.
  SDY_CHECK(it != func_op.getOps().end());
  FragmentOp bwd_fragment = cast<FragmentOp>(*it);

  EXPECT_TRUE(IsExecutedImmediatelyAfter(fwd_fragment, bwd_fragment));
}

TEST(IsExecutedImmediatelyAfter,
     ShouldReturnFalseIfAnotherFragmentInSameMeshBetweenTwoFragments) {
  const char kProgram[] = R"mlir(
    !mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
      -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %forward_result = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0) (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
       %another_fragment_result = mpmd.fragment<mesh="m1", origin=["f1"(0)]> (%arg0) (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      %backward_result = mpmd.fragment<mesh="m1", origin=["f2"(1)]> (%forward_result) (%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

      return %backward_result : !mesh_1_tensor_4_8_f32
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  FuncOp func_op = GetMainFunction(*module);

  Region::OpIterator it =
      func_op.getOps().begin();  // it points to the forward fragment.
  SDY_CHECK(it != func_op.getOps().end());
  FragmentOp fwd_fragment = cast<FragmentOp>(*it);
  SDY_CHECK(it != func_op.getOps().end());
  ++it;  // it points to the other fragment.
  SDY_CHECK(it != func_op.getOps().end());
  ++it;  // it points to the backward fragment.
  SDY_CHECK(it != func_op.getOps().end());
  FragmentOp bwd_fragment = cast<FragmentOp>(*it);

  EXPECT_FALSE(IsExecutedImmediatelyAfter(fwd_fragment, bwd_fragment));
}

TEST(IsLoweredWithSdy, LoweredWithSdyIfModuleHasSdyLoweredAttr) {
  const std::string kProgram = R"mlir(
    module attributes {mpmd.sdy_lowered} {}
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);

  EXPECT_TRUE(IsLoweredWithSdy(*module));
}

TEST(IsLoweredWithSdy, NotLoweredWithSdyIfModuleHasNoSdyLoweredAttr) {
  const std::string kProgram = R"mlir(
    module {}
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);

  EXPECT_FALSE(IsLoweredWithSdy(*module));
}

TEST(IsLoweredWithSdy, NotLoweredWithSdyIfNoModuleAttr) {
  const std::string kProgram = R"mlir(
    func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
      return %arg0 : tensor<f32>
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);

  EXPECT_FALSE(IsLoweredWithSdy(*module));
}

TEST(SdyGetSharding, ShouldGetCorrectFragmentArgsSharding) {
  const std::string kProgram = R"mlir(
   sdy.mesh @mesh = <["x"=4, "y"=2]>
   !mesh_tensor_1 = !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>
    #topology = #mpmd.topology<<"mesh1" : <["x"=4, "y"=2]>>>
   func.func public @main(%arg1: !mesh_tensor_1, %arg2: !mesh_tensor_1) -> (!mesh_tensor_1)
    attributes {topology = #topology} {
      %0 = mpmd.fragment<mesh="mesh1", origin=["stage1"], in_shardings=[<@mesh, [{"x", ?}, {?}, {?}]>, <@mesh, [{"y", ?}, {?}, {?}]>]> (%arg1, %arg2) (%arg3: tensor<16x10x3xf32>, %arg4: tensor<16x10x3xf32>) {
       %r = stablehlo.add %arg3, %arg4 : tensor<16x10x3xf32>
       mpmd.return %r : tensor<16x10x3xf32>
      } : (!mesh_tensor_1, !mesh_tensor_1) -> !mesh_tensor_1
      return %0 : !mesh_tensor_1
  }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  SDY_CHECK(module);
  auto main_func = GetMainFunction(*module);
  SDY_CHECK(main_func);

  FragmentOp fragment = cast<FragmentOp>(*main_func.getOps().begin());
  Value first_arg = (*fragment.getBody()->args_begin());
  Value second_arg = (*(fragment.getBody()->args_rbegin()));

  sdy::TensorShardingAttr sharded_on_x =
      sdy::TensorShardingAttr::getFullyOpen(&context, 3, "mesh")
          .getSharded(0, "x");

  sdy::TensorShardingAttr sharded_on_y =
      sdy::TensorShardingAttr::getFullyOpen(&context, 3, "mesh")
          .getSharded(0, "y");

  EXPECT_EQ(sharded_on_x, sdy::getSharding(first_arg));
  EXPECT_EQ(sharded_on_y, sdy::getSharding(second_arg));
}

TEST(SdySetSharding, ShouldSetCorrectFragmentArgsSharding) {
  const std::string kProgram = R"mlir(
   sdy.mesh @mesh = <["x"=4, "y"=2]>
   !mesh_tensor_1 = !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>
    #topology = #mpmd.topology<<"mesh1" : <["x"=4, "y"=2]>>>
   func.func public @main(%arg1: !mesh_tensor_1, %arg2: !mesh_tensor_1) -> (!mesh_tensor_1)
    attributes {topology = #topology} {
      %0 = mpmd.fragment<mesh="mesh1", origin=["stage1"]> (%arg1, %arg2) (%arg3: tensor<16x10x3xf32>, %arg4: tensor<16x10x3xf32>) {
       %r = stablehlo.add %arg3, %arg4 : tensor<16x10x3xf32>
       mpmd.return %r : tensor<16x10x3xf32>
      } : (!mesh_tensor_1, !mesh_tensor_1) -> !mesh_tensor_1
      return %0 : !mesh_tensor_1
  }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  SDY_CHECK(module);
  auto main_func = GetMainFunction(*module);
  SDY_CHECK(main_func);

  FragmentOp fragment = cast<FragmentOp>(*main_func.getOps().begin());
  Value first_arg = (*fragment.getBody()->args_begin());
  Value second_arg = (*(fragment.getBody()->args_rbegin()));

  sdy::TensorShardingAttr sharded_on_x =
      sdy::TensorShardingAttr::getFullyOpen(&context, 3, "mesh")
          .getSharded(0, "x");
  sdy::TensorShardingAttr sharded_on_y =
      sdy::TensorShardingAttr::getFullyOpen(&context, 3, "mesh")
          .getSharded(0, "y");

  // There should be no sharding before setting them.
  EXPECT_EQ(sdy::TensorShardingAttr(), sdy::getSharding(first_arg));
  EXPECT_EQ(sdy::TensorShardingAttr(), sdy::getSharding(second_arg));

  // Setting the sharding of the first argument should only set the first and
  // the second fully open.
  sdy::setSharding(first_arg, sharded_on_x);
  EXPECT_EQ(sharded_on_x, sdy::getSharding(first_arg));
  EXPECT_EQ(sdy::TensorShardingAttr::getFullyOpen(&context, 3, "mesh"),
            sdy::getSharding(second_arg));

  // Setting the sharding of the second argument should only set the second and
  // leave the first unchanged.
  sdy::setSharding(second_arg, sharded_on_y);
  EXPECT_EQ(sharded_on_x, sdy::getSharding(first_arg));
  EXPECT_EQ(sharded_on_y, sdy::getSharding(second_arg));

  // // Overriding existing sharding should be successful.
  sdy::setSharding(first_arg, sharded_on_y);
  EXPECT_EQ(sharded_on_y, sdy::getSharding(first_arg));
}

TEST(SdyGetArgsShardings, ShouldGetShardingsForAllArgs) {
  const std::string kProgram = R"mlir(
   sdy.mesh @mesh = <["x"=4, "y"=2]>
   !mesh_tensor_1 = !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>
    #topology = #mpmd.topology<<"mesh1" : <["x"=4, "y"=2]>>>
   func.func public @main(%arg1: !mesh_tensor_1, %arg2: !mesh_tensor_1) -> (!mesh_tensor_1)
    attributes {topology = #topology} {
      %0 = mpmd.fragment<mesh="mesh1", origin=["stage1"], in_shardings=[<@mesh, [{"x", ?}, {?}, {?}]>, <@mesh, [{"y", ?}, {?}, {?}]>]> (%arg1, %arg2) (%arg3: tensor<16x10x3xf32>, %arg4: tensor<16x10x3xf32>) {
       %r = stablehlo.add %arg3, %arg4 : tensor<16x10x3xf32>
       mpmd.return %r : tensor<16x10x3xf32>
      } : (!mesh_tensor_1, !mesh_tensor_1) -> !mesh_tensor_1
      return %0 : !mesh_tensor_1
  }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  SDY_CHECK(module);
  auto main_func = GetMainFunction(*module);
  SDY_CHECK(main_func);
  FragmentOp fragment = cast<FragmentOp>(*main_func.getOps().begin());

  auto expected_shardings = SmallVector<sdy::TensorShardingAttr>{
      sdy::TensorShardingAttr::getFullyOpen(&context, 3, "mesh")
          .getSharded(0, "x"),
      sdy::TensorShardingAttr::getFullyOpen(&context, 3, "mesh")
          .getSharded(0, "y")};

  for (const auto& [i, sharding] :
       llvm::enumerate(fragment.getBlockArgumentEdgeOwnerShardings())) {
    EXPECT_EQ(expected_shardings[i], sharding);
  }
}

TEST(SdySetArgShardings, ShouldSetShardingsForAllArgs) {
  const std::string kProgram = R"mlir(
   sdy.mesh @mesh = <["x"=4, "y"=2]>
   !mesh_tensor_1 = !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>
    #topology = #mpmd.topology<<"mesh1" : <["x"=4, "y"=2]>>>
   func.func public @main(%arg1: !mesh_tensor_1, %arg2: !mesh_tensor_1) -> (!mesh_tensor_1)
    attributes {topology = #topology} {
      %0 = mpmd.fragment<mesh="mesh1", origin=["stage1"]> (%arg1, %arg2) (%arg3: tensor<16x10x3xf32>, %arg4: tensor<16x10x3xf32>) {
       %r = stablehlo.add %arg3, %arg4 : tensor<16x10x3xf32>
       mpmd.return %r : tensor<16x10x3xf32>
      } : (!mesh_tensor_1, !mesh_tensor_1) -> !mesh_tensor_1
      return %0 : !mesh_tensor_1
  }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  SDY_CHECK(module);
  auto main_func = GetMainFunction(*module);
  SDY_CHECK(main_func);

  FragmentOp fragment = cast<FragmentOp>(*main_func.getOps().begin());
  sdy::TensorShardingAttr sharded_on_x =
      sdy::TensorShardingAttr::getFullyOpen(&context, 3, "mesh")
          .getSharded(0, "x");
  sdy::TensorShardingAttr sharded_on_y =
      sdy::TensorShardingAttr::getFullyOpen(&context, 3, "mesh")
          .getSharded(1, "y");
  Value first_arg = (*fragment.getBody()->args_begin());
  Value second_arg = (*(fragment.getBody()->args_rbegin()));

  fragment.setBlockArgumentEdgeOwnerShardings({sharded_on_x, sharded_on_y});

  EXPECT_EQ(sharded_on_x, sdy::getSharding(first_arg));
  EXPECT_EQ(sharded_on_y, sdy::getSharding(second_arg));
}

TEST(GetEdgeOwners, ShouldGetCorrectBlockArgumentAndResultEdgeOwners) {
  const std::string kProgram = R"mlir(
   sdy.mesh @mesh = <["x"=4, "y"=2]>
   !mesh_tensor_1 = !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>
    #topology = #mpmd.topology<<"mesh1" : <["x"=4, "y"=2]>>>
   func.func public @main(%arg1: !mesh_tensor_1, %arg2: !mesh_tensor_1) -> (!mesh_tensor_1)
    attributes {topology = #topology} {
      %0 = mpmd.fragment<mesh="mesh1", origin=["stage1"]> (%arg1, %arg2) (%arg3: tensor<16x10x3xf32>, %arg4: tensor<16x10x3xf32>) {
       %r = stablehlo.add %arg3, %arg4 : tensor<16x10x3xf32>
       mpmd.return %r : tensor<16x10x3xf32>
      } : (!mesh_tensor_1, !mesh_tensor_1) -> !mesh_tensor_1
      return %0 : !mesh_tensor_1
  }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  SDY_CHECK(module);
  FuncOp main_func = GetMainFunction(*module);
  SDY_CHECK(main_func);

  auto fragment = cast<FragmentOp>(*main_func.getOps().begin());

  EXPECT_EQ(fragment.getBlockArgumentEdgeOwners(),
            fragment.getBody()->getArguments());
  EXPECT_EQ(fragment.getOpResultEdgeOwners(), fragment.getResults());
}

TEST(GetEdgeOwner, ShouldGetCorrectOwnerFromSourceOrTarget) {
  const std::string kProgram = R"mlir(
   sdy.mesh @mesh = <["x"=4, "y"=2]>
   !mesh_tensor_1 = !mpmd.mesh_tensor<"mesh1", tensor<16x10x3xf32>>
    #topology = #mpmd.topology<<"mesh1" : <["x"=4, "y"=2]>>>
   func.func public @main(%arg1: !mesh_tensor_1, %arg2: !mesh_tensor_1) -> (!mesh_tensor_1)
    attributes {topology = #topology} {
      %0 = mpmd.fragment<mesh="mesh1", origin=["stage1"]> (%arg1, %arg2) (%arg3: tensor<16x10x3xf32>, %arg4: tensor<16x10x3xf32>) {
       %r = stablehlo.add %arg3, %arg4 : tensor<16x10x3xf32>
       mpmd.return %r : tensor<16x10x3xf32>
      } : (!mesh_tensor_1, !mesh_tensor_1) -> !mesh_tensor_1
      return %0 : !mesh_tensor_1
  }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  SDY_CHECK(module);
  FuncOp main_func = GetMainFunction(*module);
  SDY_CHECK(main_func);

  auto fragment = cast<FragmentOp>(*main_func.getOps().begin());
  Value result = fragment.getResult(0);

  EXPECT_EQ(fragment.getEdgeOwnerFromTarget(result), result);
  for (OpOperand& operand : fragment->getOpOperands()) {
    EXPECT_EQ(fragment.getEdgeOwnerFromSource(operand),
              fragment.getBody()->getArgument(operand.getOperandNumber()));
  }
}

}  // namespace
}  // namespace mlir::mpmd
