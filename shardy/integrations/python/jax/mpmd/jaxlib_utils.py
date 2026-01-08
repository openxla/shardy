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
"""Utilities for converting between Python types and jaxlib pybind types."""

from jaxlib import _sdy_mpmd as jaxlib_mpmd

from shardy.integrations.python.jax.mpmd import types


def _to_jaxlib_split_type(
    split_type: types.SplitFragmentType | None,
) -> jaxlib_mpmd.SplitFragmentType | None:
  """Convert native Python enum to pybinded enum."""
  if split_type is None:
    return None
  if split_type == types.SplitFragmentType.KEEP_TRANSFERRED:
    return jaxlib_mpmd.SplitFragmentType.KEEP_TRANSFERRED
  elif split_type == types.SplitFragmentType.DROP_TRANSFERRED:
    return jaxlib_mpmd.SplitFragmentType.DROP_TRANSFERRED
  else:
    raise ValueError(f'Unknown SplitFragmentType: {split_type}')


def _from_jaxlib_split_type(
    split_type: jaxlib_mpmd.SplitFragmentType | None,
) -> types.SplitFragmentType | None:
  """Convert pybinded enum to native Python enum."""
  if split_type is None:
    return None
  if split_type == jaxlib_mpmd.SplitFragmentType.KEEP_TRANSFERRED:
    return types.SplitFragmentType.KEEP_TRANSFERRED
  elif split_type == jaxlib_mpmd.SplitFragmentType.DROP_TRANSFERRED:
    return types.SplitFragmentType.DROP_TRANSFERRED
  else:
    raise ValueError(f'Unknown jaxlib_mpmd.SplitFragmentType: {split_type}')


def convert_fragment_info_to_pybind(
    fragment: types.FragmentInfo,
) -> jaxlib_mpmd.FragmentInfo:
  """Converts FragmentInfo to jaxlib_mpmd.FragmentInfo."""
  return jaxlib_mpmd.FragmentInfo(
      origins=[
          jaxlib_mpmd.FragmentOrigin(
              origin.computation_name, origin.transpose_count
          )
          for origin in fragment.origins
      ],
      stage_id=fragment.stage_id,
      call_counter=fragment.call_counter,
      split_type=_to_jaxlib_split_type(fragment.split_type),
      mesh_name=fragment.mesh_name,
  )


def convert_pybind_fragment_info_to_types(
    fragment: jaxlib_mpmd.FragmentInfo,
) -> types.FragmentInfo:
  """Converts jaxlib_mpmd.FragmentInfo to FragmentInfo."""
  return types.FragmentInfo(
      origins=tuple(
          types.FragmentOrigin(origin.computation_name, origin.transpose_count)
          for origin in fragment.origins
      ),
      stage_id=fragment.stage_id,
      call_counter=fragment.call_counter,
      split_type=_from_jaxlib_split_type(fragment.split_type),
      mesh_name=fragment.mesh_name,
  )


def convert_fragment_merge_rules_to_pybind(
    fragment_merge_rules: types.FragmentMergeRules,
) -> list[jaxlib_mpmd.FragmentMergeRule]:
  """Converts fragment merge rules to jaxlib_mpmd.FragmentMergeRules."""
  pybind_fragment_merge_rules = []
  for rule in fragment_merge_rules:
    fragments = [
        convert_fragment_info_to_pybind(fragment) for fragment in rule.sources
    ]
    pybind_fragment_merge_rules.append(
        jaxlib_mpmd.FragmentMergeRule(
            sources=fragments,
            target=convert_fragment_info_to_pybind(rule.target),
        )
    )
  return pybind_fragment_merge_rules


def convert_fragment_schedule_rules_to_pybind(
    fragment_schedule_rules: types.FragmentScheduleRules,
) -> list[jaxlib_mpmd.FragmentScheduleRule]:
  """Converts fragment schedule rules to jaxlib_mpmd.FragmentScheduleRules."""
  pybind_fragment_schedule_rules = []
  for rule in fragment_schedule_rules:
    fragments = [
        convert_fragment_info_to_pybind(fragment)
        for fragment in rule.ordered_fragments
    ]
    pybind_fragment_schedule_rules.append(
        jaxlib_mpmd.FragmentScheduleRule(ordered_fragments=fragments)
    )
  return pybind_fragment_schedule_rules
