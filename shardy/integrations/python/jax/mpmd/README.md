<!-- Copyright 2025 The MPMD Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================== -->

# Shardy MPMD

Shardy MPMD JAX integration for multi-program multi-data parallelism.

## Development Setup

To set up a development environment:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/openxla/shardy.git
    cd shardy/shardy/integrations/python/jax/mpmd
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv sdy_venv
    source sdy_venv/bin/activate
    ```

3.  **Install the package in editable mode with dev dependencies:**

    ```bash
    pip install -e '.[dev]'
    ```

4.  **Run unit tests:**

    ```bash
    pytest jit_test.py
    ```

## Usage

Here is a simple "Hello World" example demonstrating how to use Shardy MPMD to define a 2-stage pipeline.

See also https://github.com/openxla/shardy/blob/main/rfcs/2025-06-18-mpmd-rfc-read-only-colab.ipynb
(We'll convert this to a runnable colab soon...)

```python
import jax
import jax.numpy as jnp
import numpy as np
from shardy.integrations.python.jax import mpmd
from shardy.integrations.python.jax.mpmd import stages

def mpmd_hello_world():
  # 1. Define the MPMD function
  def mpmd_fn(w1, w2, x):
    # Stage 1: Matrix multiplication on mesh1
    stage1 = mpmd.named_computation(lambda a: a @ w1, name='stage1')(x)
    # Stage 2: Matrix multiplication on mesh2
    stage2 = mpmd.named_computation(lambda a: a @ w2, name='stage2')(stage1)
    return stage2

  # 2. Define the configuration (topology and mesh assignment)
  # In a real scenario, you would define your mesh topology here.
  # This example assumes a helper or existing topology.
  # For demonstration, we'll use a mock topology setup.
  # topology = ...
  # mpmd_config = mpmd.make_config(
  #     topology,
  #     {'stage1': 'mesh1', 'stage2': 'mesh2'}
  # )

  # NOTE: For this example to run, you need a valid topology.
  # See tests/jit_test.py for full setup details.

  # 3. JIT compile and lower
  # lowered: stages.MpmdLowered = mpmd.jit(mpmd_fn, mpmd_config).lower(
  #     np.ones((3, 5), dtype=jnp.float32),
  #     np.ones((5, 7), dtype=jnp.float32),
  #     np.ones((10, 3), dtype=jnp.float32),
  # )

  # 4. Compile (requires Pathways backend)
  # compiled = lowered.compile()
```

## Concepts

-   **`mpmd.named_computation`**: Wraps a function to be executed on a specific named mesh.
-   **`mpmd.make_config`**: Creates a configuration object binding stage names to mesh names in the topology.
-   **`mpmd.jit`**: JIT compiles the MPMD function, handling the partitioning and communication between stages.
