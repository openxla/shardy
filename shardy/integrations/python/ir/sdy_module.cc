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
#include <string>
#include <vector>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"  // IWYU pragma: keep
#include "shardy/integrations/c/attributes.h"
#include "shardy/integrations/c/dialect.h"

namespace mlir {
namespace sdy {

namespace {

namespace py = pybind11;

// Returns a vector containing MlirAttribute elements extracted from an
// attribute using the two provided callbacks.
std::vector<MlirAttribute> attributePropertyVector(
    MlirAttribute attr, llvm::function_ref<intptr_t(MlirAttribute)> sizeFn,
    llvm::function_ref<MlirAttribute(MlirAttribute, intptr_t)> getFn) {
  std::vector<MlirAttribute> result;
  intptr_t size = sizeFn(attr);
  result.reserve(size);
  for (intptr_t i = 0; i < size; ++i) {
    result.push_back(getFn(attr, i));
  }
  return result;
}

auto toPyString(MlirStringRef mlirStringRef) {
  return py::str(mlirStringRef.data, mlirStringRef.length);
}

auto toStringRef(const std::string& s) {
  return mlirStringRefCreate(s.c_str(), s.size());
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
             MlirContext ctx) {
            return cls(sdyMeshAttrGet(ctx, meshAxes.size(), meshAxes.data()));
          },
          py::arg("cls"), py::arg("meshAxes"), py::arg("context") = py::none(),
          "Creates a MeshAttr with the given mesh axes.")
      .def_property_readonly("axes", [](MlirAttribute self) {
        return attributePropertyVector(self, sdyMeshAttrGetAxesSize,
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
                               return attributePropertyVector(
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
          [](py::object cls, const std::string& meshName,
             const std::vector<MlirAttribute>& dimensionShardings,
             const std::vector<MlirAttribute>& replicatedAxes,
             MlirContext ctx) {
            return cls(sdyTensorShardingAttrGet(
                ctx, toStringRef(meshName), dimensionShardings.size(),
                dimensionShardings.data(), replicatedAxes.size(),
                replicatedAxes.data()));
          },
          py::arg("cls"), py::arg("mesh_name"), py::arg("dimension_shardings"),
          py::arg("replicated_axes") = std::vector<MlirAttribute>(),
          py::arg("context") = py::none(),
          "Creates a TensorShardingAttr with the mesh name, dimension "
          "shardings, and replicated axes.")
      .def_property_readonly(
          "mesh_name",
          [](MlirAttribute self) {
            return toPyString(sdyTensorShardingAttrGetMeshName(self));
          })
      .def_property_readonly("dimension_shardings",
                             [](MlirAttribute self) {
                               return attributePropertyVector(
                                   self,
                                   sdyTensorShardingAttrGetDimShardingsSize,
                                   sdyTensorShardingAttrGetDimShardingsElem);
                             })
      .def_property_readonly("replicated_axes", [](MlirAttribute self) {
        return attributePropertyVector(
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
        return attributePropertyVector(
            self, sdyTensorShardingPerValueAttrGetShardingsSize,
            sdyTensorShardingPerValueAttrGetShardingsElem);
      });
}

}  // namespace
}  // namespace sdy
}  // namespace mlir
