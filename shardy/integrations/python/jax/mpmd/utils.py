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

"""Utility functions for MPMD ops."""

from collections.abc import Callable
import functools
from typing import Any, TypeVar

X = TypeVar('X')
Y = TypeVar('Y')
T = TypeVar('T')


def get_func_name(func: Callable[..., Any], prefix: str = 'mpmd_') -> str:
  """Attempts to determine a name for func."""
  if hasattr(func, '__name__'):
    return f'{prefix}{func.__name__}'
  elif isinstance(func, functools.partial) and hasattr(func.func, '__name__'):
    return f'{prefix}{func.func.__name__}'
  else:
    return f'{prefix}fn'
