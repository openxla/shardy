/* Copyright 2024 The Shardy Authors.

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

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"  // IWYU pragma: keep
#include "shardy/integrations/c/attributes.h"
#include "shardy/integrations/c/dialect.h"

namespace mlir {
namespace sdy {

namespace {

namespace py = pybind11;

// Returns a vector containing elements with type T extracted from an attribute
// using the two provided callbacks.
template <typename T>
std::vector<T> propertyVector(
    MlirAttribute attr, llvm::function_ref<intptr_t(MlirAttribute)> sizeFn,
    llvm::function_ref<T(MlirAttribute, intptr_t)> getFn) {
  std::vector<T> result;
  intptr_t size = sizeFn(attr);
  result.reserve(size);
  for (intptr_t i = 0; i < size; ++i) {
    result.push_back(getFn(attr, i));
  }
  return result;
}

py::str toPyString(MlirStringRef mlirStringRef) {
  return py::str(mlirStringRef.data, mlirStringRef.length);
}

MlirStringRef toStringRef(const std::string& s) {
  return mlirStringRefCreate(s.c_str(), s.size());
}

MlirAttribute toMeshOrRefAttr(
    MlirContext ctx,
    const std::variant<std::string, MlirAttribute>& meshOrRef) {
  if (auto* meshName = std::get_if<std::string>(&meshOrRef)) {
    return mlirFlatSymbolRefAttrGet(ctx, toStringRef(*meshName));
  }
  return std::get<MlirAttribute>(meshOrRef);
}

PYBIND11_MODULE(_sdy, m) {
  m.doc() = "SDY main Python extension";

  //
  // Dialects.
  //

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__sdy__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

  //
  // Attributes.
  //

  mlir::python::adaptors::mlir_attribute_subclass(m, "MeshAxisAttr",
                                                  sdyAttributeIsAMeshAxisAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string& name, int64_t size,
             MlirContext ctx) {
            return cls(sdyMeshAxisAttrGet(ctx, toStringRef(name), size));
          },
          py::arg("cls"), py::arg("name"), py::arg("size"),
          py::arg("context") = py::none(),
          "Creates a MeshAxisAttr with the given axis name and size.")
      .def_property_readonly("name",
                             [](MlirAttribute self) {
                               return toPyString(sdyMeshAxisAttrGetName(self));
                             })
      .def_property_readonly("size", [](MlirAttribute self) {
        return sdyMeshAxisAttrGetSize(self);
      });

  mlir::python::adaptors::mlir_attribute_subclass(m, "MeshAttr",
                                                  sdyAttributeIsAMeshAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<MlirAttribute>& meshAxes,
             const std::vector<int64_t>& deviceIds,
             MlirContext ctx) {
            return cls(sdyMeshAttrGet(ctx, meshAxes.size(), meshAxes.data(),
                                      deviceIds.size(), deviceIds.data()));
          },
          py::arg("cls"), py::arg("mesh_axes"),
          py::arg("device_ids") = std::vector<int64_t>(),
          py::arg("context") = py::none(),
          "Creates a MeshAttr with the given mesh axes.")
      .def_property_readonly("device_ids", [](MlirAttribute self) {
        return propertyVector<int64_t>(self, sdyMeshAttrGetDeviceIdsSize,
                                             sdyMeshAttrGetDeviceIdsElem);
      })
      .def_property_readonly("axes", [](MlirAttribute self) {
        return propertyVector<MlirAttribute>(self, sdyMeshAttrGetAxesSize,
                                             sdyMeshAttrGetAxesElem);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "SubAxisInfoAttr", sdyAttributeIsASubAxisInfoAttr)
      .def_classmethod(
          "get",
          [](py::object cls, int64_t preSize, int64_t size, MlirContext ctx) {
            return cls(sdySubAxisInfoAttrGet(ctx, preSize, size));
          },
          py::arg("cls"), py::arg("pre_size"), py::arg("size"),
          py::arg("context") = py::none(),
          "Creates a SubAxisInfoAttr with the given pre-size and size.")
      .def_property_readonly(
          "pre_size",
          [](MlirAttribute self) { return sdySubAxisInfoAttrGetPreSize(self); })
      .def_property_readonly("size", [](MlirAttribute self) {
        return sdySubAxisInfoAttrGetSize(self);
      });

  mlir::python::adaptors::mlir_attribute_subclass(m, "AxisRefAttr",
                                                  sdyAttributeIsAnAxisRefAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string& name,
             std::optional<MlirAttribute> subAxisInfoAttr, MlirContext ctx) {
            return cls(sdyAxisRefAttrGet(ctx, toStringRef(name),
                                         subAxisInfoAttr.has_value()
                                             ? *subAxisInfoAttr
                                             : MlirAttribute()));
          },
          py::arg("cls"), py::arg("name"),
          py::arg("sub_axis_info") = py::none(),
          py::arg("context") = py::none(),
          "Creates an AxisRefAttr with the given name and SubAxisInfoAttr.")
      .def_property_readonly("name",
                             [](MlirAttribute self) {
                               return toPyString(sdyAxisRefAttrGetName(self));
                             })
      .def_property_readonly("sub_axis_info", [](MlirAttribute self) {
        MlirAttribute subAxisInfo = sdyAxisRefAttrGetSubAxisInfo(self);
        return subAxisInfo.ptr == nullptr ? std::nullopt
                                          : std::optional(subAxisInfo);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "DimensionShardingAttr", sdyAttributeIsADimensionShardingAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<MlirAttribute>& axes,
             bool isClosed, std::optional<int64_t> priority, MlirContext ctx) {
            return cls(sdyDimensionShardingAttrGet(
                ctx, axes.size(), axes.data(), isClosed,
                priority.has_value() ? *priority : -1));
          },
          py::arg("cls"), py::arg("axes"), py::arg("is_closed"),
          py::arg("priority") = py::none(), py::arg("context") = py::none(),
          "Creates a DimensionShardingAttr with the given axes, whether it's "
          "closed, and priority.")
      .def_property_readonly("axes",
                             [](MlirAttribute self) {
                               return propertyVector<MlirAttribute>(
                                   self, sdyDimensionShardingAttrGetAxesSize,
                                   sdyDimensionShardingAttrGetAxesElem);
                             })
      .def_property_readonly("is_closed",
                             [](MlirAttribute self) {
                               return sdyDimensionShardingAttrGetIsClosed(self);
                             })
      .def_property_readonly("priority", [](MlirAttribute self) {
        int64_t priority = sdyDimensionShardingAttrGetPriority(self);
        return priority == -1 ? std::nullopt : std::optional(priority);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "TensorShardingAttr", sdyAttributeIsATensorShardingAttr)
      .def_classmethod(
          "get",
          [](py::object cls,
             const std::variant<std::string, MlirAttribute>& meshOrRef,
             const std::vector<MlirAttribute>& dimensionShardings,
             const std::vector<MlirAttribute>& replicatedAxes,
             MlirContext ctx) {
            return cls(sdyTensorShardingAttrGet(
                ctx, toMeshOrRefAttr(ctx, meshOrRef), dimensionShardings.size(),
                dimensionShardings.data(), replicatedAxes.size(),
                replicatedAxes.data()));
          },
          py::arg("cls"), py::arg("mesh_or_ref"),
          py::arg("dimension_shardings"),
          py::arg("replicated_axes") = std::vector<MlirAttribute>(),
          py::arg("context") = py::none(),
          "Creates a TensorShardingAttr with either an inlined mesh or mesh "
          "name, dimension shardings, and replicated axes.")
      .def_property_readonly("mesh_or_ref",
                             [](MlirAttribute self) {
                               return sdyTensorShardingAttrGetMeshOrRef(self);
                             })
      .def_property_readonly("dimension_shardings",
                             [](MlirAttribute self) {
                               return propertyVector<MlirAttribute>(
                                   self,
                                   sdyTensorShardingAttrGetDimShardingsSize,
                                   sdyTensorShardingAttrGetDimShardingsElem);
                             })
      .def_property_readonly("replicated_axes", [](MlirAttribute self) {
        return propertyVector<MlirAttribute>(
            self, sdyTensorShardingAttrGetReplicatedAxesSize,
            sdyTensorShardingAttrGetReplicatedAxesElem);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "TensorShardingPerValueAttr",
      sdyAttributeIsATensorShardingPerValueAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<MlirAttribute>& shardings,
             MlirContext ctx) {
            return cls(sdyTensorShardingPerValueAttrGet(ctx, shardings.size(),
                                                        shardings.data()));
          },
          py::arg("cls"), py::arg("shardings"), py::arg("context") = py::none(),
          "Creates a TensorShardingPerValueAttr with the tensor shardings.")
      .def_property_readonly("shardings", [](MlirAttribute self) {
        return propertyVector<MlirAttribute>(
            self, sdyTensorShardingPerValueAttrGetShardingsSize,
            sdyTensorShardingPerValueAttrGetShardingsElem);
      });

  mlir::python::adaptors::mlir_attribute_subclass(m, "DimMappingAttr",
                                                  sdyAttributeIsADimMappingAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<int64_t>& factorIndices,
             MlirContext ctx) {
            return cls(sdyDimMappingAttrGet(ctx, factorIndices.size(),
                                            factorIndices.data()));
          },
          py::arg("cls"), py::arg("factor_indices"),
          py::arg("context") = py::none(),
          "Creates a DimMappingAttr with the factor indices.")
      .def_property_readonly("factor_indices", [](MlirAttribute self) {
        return propertyVector<intptr_t>(self,
                                        sdyDimMappingAttrGetFactorIndicesSize,
                                        sdyDimMappingAttrGetFactorIndicesElem);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "TensorMappingAttr", sdyAttributeIsATensorMappingAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<MlirAttribute>& mappings,
             MlirContext ctx) {
            return cls(
                sdyTensorMappingAttrGet(ctx, mappings.size(), mappings.data()));
          },
          py::arg("cls"), py::arg("dim_mappings"),
          py::arg("context") = py::none(),
          "Creates a TensorMappingAttr with the dim mappings.")
      .def_property_readonly("dim_mappings",
                             [](MlirAttribute self) {
                               return propertyVector<MlirAttribute>(
                                   self, sdyTensorMappingAttrGetDimMappingsSize,
                                   sdyTensorMappingAttrGetDimMappingsElem);
                             })
      .def_property_readonly("rank", [](MlirAttribute self) {
        return sdyTensorMappingAttrGetRank(self);
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "OpShardingRuleAttr", sdyAttributeIsAOpShardingRuleAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<int64_t>& factorSizes,
             const std::vector<MlirAttribute>& operandMappings,
             const std::vector<MlirAttribute>& resultMappings,
             const std::vector<int64_t>& reductionFactors,
             const std::vector<int64_t>& needReplicationFactors, bool isCustom,
             MlirContext ctx) {
            return cls(sdyOpShardingRuleAttrGet(
                ctx, factorSizes.size(), factorSizes.data(),
                operandMappings.size(), operandMappings.data(),
                resultMappings.size(), resultMappings.data(),
                reductionFactors.size(), reductionFactors.data(),
                needReplicationFactors.size(), needReplicationFactors.data(),
                isCustom));
          },
          py::arg("cls"), py::arg("factor_sizes"), py::arg("operand_mappings"),
          py::arg("result_mappings"),
          py::arg("reduction_factors") = std::vector<int64_t>(),
          py::arg("need_replication_factors") = std::vector<int64_t>(),
          py::arg("is_custom") = false, py::arg("context") = py::none(),
          "Creates a OpShardingRuleAttr with the factor sizes and mappings for "
          "operands and results.")
      .def_property_readonly("is_custom",
                             [](MlirAttribute self) {
                               return sdyOpShardingRuleAttrGetIsCustom(self);
                             })
      .def_property_readonly("factor_sizes",
                             [](MlirAttribute self) {
                               return propertyVector<intptr_t>(
                                   self,
                                   sdyOpShardingRuleAttrGetFactorSizesSize,
                                   sdyOpShardingRuleAttrGetFactorSizesElem);
                             })
      .def_property_readonly("operand_mappings",
                             [](MlirAttribute self) {
                               return propertyVector<MlirAttribute>(
                                   self,
                                   sdyOpShardingRuleAttrGetOperandMappingsSize,
                                   sdyOpShardingRuleAttrGetOperandMappingsElem);
                             })
      .def_property_readonly("result_mappings",
                             [](MlirAttribute self) {
                               return propertyVector<MlirAttribute>(
                                   self,
                                   sdyOpShardingRuleAttrGetResultMappingsSize,
                                   sdyOpShardingRuleAttrGetResultMappingsElem);
                             })
      .def_property_readonly(
          "reduction_factors",
          [](MlirAttribute self) {
            return propertyVector<intptr_t>(
                self, sdyOpShardingRuleAttrGetReductionFactorsSize,
                sdyOpShardingRuleAttrGetReductionFactorsElem);
          })
      .def_property_readonly(
          "need_replication_factors", [](MlirAttribute self) {
            return propertyVector<intptr_t>(
                self, sdyOpShardingRuleAttrGetNeedReplicationFactorsSize,
                sdyOpShardingRuleAttrGetNeedReplicationFactorsElem);
          });

  mlir::python::adaptors::mlir_attribute_subclass(m, "ManualAxesAttr",
                                                  sdyAttributeIsAManualAxesAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::vector<MlirAttribute>& meshAxes,
             MlirContext ctx) {
            return cls(sdyManualAxesAttrGet(ctx, meshAxes.size(),
                                            meshAxes.data()));
          },
          py::arg("cls"), py::arg("manual_axes"),
          py::arg("context") = py::none(),
          "Creates a ManualAxesAttr with the given manual axes.")
      .def("__getitem__", [](MlirAttribute &self, unsigned index) {
        if (index >= sdyManualAxesAttrGetAxesSize(self)) {
          throw py::index_error();
        }
        return toPyString(sdyManualAxesAttrGetAxesElem(self, index));
      })
      .def("__len__", [](MlirAttribute &self) {
        return sdyManualAxesAttrGetAxesSize(self);
      });
}

}  // namespace
}  // namespace sdy
}  // namespace mlir
