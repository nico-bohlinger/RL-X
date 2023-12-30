# Custom MuJoCo Environments

Contains an example for a custom MuJoCo environment.

This example can be used as a template for other custom MuJoCo environments. It contains:
- All necessary handling of the MuJoCo physics engine directly via its Python bindings to form a stand-alone environment class
- Implementation of a GLFW viewer for rendering

More specifically, the example uses the Ant robot and defines as the task to track a given velocity command.

| Version | Observation space | Action space | Data interface |
| ----------- | ----------- | ----------- | ----------- |
| Ant | Flat value | Continuous | Numpy |