# Custom MuJoCo Environments

Contains a two examples for custom MuJoCo environments.

The examples can be used as a template for other custom MuJoCo environments. They contain:
- All necessary handling of the MuJoCo physics engine directly via its Python bindings to form a stand-alone environment class
- Implementation of a GLFW viewer for rendering (non-MJX version)

The version with the MJX suffix uses the new MuJoCo XLA backend that enables running the environment on a GPU (similar to Isaac Gym / Sim).
A modern NVIDIA GPU can easily handle 4000 of those environments in parallel.
This gives a significant speedup compared to using normal MuJoCo.  
The example environment jit-compiles the step and reset functions and uses a Gym wrapper to be compatible with the standard Gym interface.
A bigger speedup could be achieved by jitting the complete training loop.
Algorithm implementations that jit-compile the complete loop break the compatibility with all other environments but an example will be added in the future.  
Use this version together with a JAX-based algorithm as the combination of JAX (used in MJX) and PyTorch can cause problems.

More specifically, the example uses the Ant robot and defines as the task to track a given velocity command.

| Version | Observation space | Action space | Data interface |
| ----------- | ----------- | ----------- | ----------- |
| Ant | Flat value | Continuous | Numpy |
| Ant MJX | Flat value | Continuous | Numpy |