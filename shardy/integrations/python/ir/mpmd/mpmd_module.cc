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

#include <cstdint>
#include <string>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep
#include "nanobind/nanobind.h"
#include "shardy/integrations/c/mpmd/attributes.h"
#include "shardy/integrations/c/mpmd/dialect.h"

namespace mlir {
namespace sdy {

namespace {

namespace nb = nanobind;

nb::str toPyString(MlirStringRef mlirStringRef) {
  return nb::str(mlirStringRef.data, mlirStringRef.length);
}

MlirStringRef toStringRef(const std::string& s) {
  return mlirStringRefCreate(s.c_str(), s.size());
}

NB_MODULE(_sdyMpmd, m) {
  m.doc() = "MPMD main Python extension";

  //
  // Dialects.
  //

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__mpmd__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  //
  // Attributes.
  //

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "UserOriginAttr", mpmdAttributeIsAUserOriginAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::string& name, int64_t transposeCount,
             MlirContext ctx) {
            return cls(
                mpmdUserOriginAttrGet(ctx, toStringRef(name), transposeCount));
          },
          nb::arg("cls"), nb::arg("name"), nb::arg("transpose_count"),
          nb::arg("context").none() = nb::none(),
          "Creates a UserOriginAttr with the given user name and transpose "
          "count.")
      .def_property_readonly(
          "user_name",
          [](MlirAttribute self) {
            return toPyString(mpmdUserOriginAttrGetUserName(self));
          })
      .def_property_readonly("transpose_count", [](MlirAttribute self) {
        return mpmdUserOriginAttrGetTransposeCount(self);
      });
}

}  // namespace
}  // namespace sdy
}  // namespace mlir
