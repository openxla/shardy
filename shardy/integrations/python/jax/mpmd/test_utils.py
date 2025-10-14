# Copyright 2025 The MPMD Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test utils for MPMD. Do not use in production code!"""

import jax
from jax import sharding
import numpy as np


def with_sharding(tensor, partition_spec, mesh):
  return jax.lax.with_sharding_constraint(
      tensor,
      sharding.NamedSharding(mesh, sharding.PartitionSpec(*partition_spec)),
  )


def named_sharding(
    mesh: sharding.Mesh | sharding.AbstractMesh, *specs
) -> sharding.NamedSharding:
  return sharding.NamedSharding(mesh, sharding.PartitionSpec(*specs))


def get_two_mesh_topology():
  """Returns a 2-mesh topology with 4 devices per mesh."""
  devices = jax.devices()
  device_count = len(devices) // 2
  mesh1 = sharding.Mesh(np.array(devices[:device_count]), ('x',))
  mesh2 = sharding.Mesh(np.array(devices[device_count:]), ('x',))

  return {'mesh1': mesh1, 'mesh2': mesh2}


def get_four_mesh_topology():
  """Returns a 4-mesh topology with 2 devices per mesh."""
  devices = jax.devices()
  device_count = len(devices) // 4
  mesh1 = sharding.Mesh(np.array(devices[:device_count]), ('x',))
  mesh2 = sharding.Mesh(
      np.array(devices[device_count : device_count * 2]), ('x',)
  )
  mesh3 = sharding.Mesh(
      np.array(devices[device_count * 2 : device_count * 3]), ('x',)
  )
  mesh4 = sharding.Mesh(
      np.array(devices[device_count * 3 : device_count * 4]), ('x',)
  )

  return {'mesh1': mesh1, 'mesh2': mesh2, 'mesh3': mesh3, 'mesh4': mesh4}
