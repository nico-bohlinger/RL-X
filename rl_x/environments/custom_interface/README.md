# Custom Environments

Contains a prototype for a custom environment interface with simple socket communication.

| Version | Observation space | Action space | Data interface |
| ----------- | ----------- | ----------- | ----------- |
| Prototype | Flat value | Continuous | Numpy |

## Communication
The underlying environment has to send an initial encoded message after the connection is established in the following JSON format:

```json
{
    "actionCount": 10,
    "observationCount": 300,
}
```

The environment will receive encoded actions that are defined in the following JSON format:

```json
{
    "action": [0.0, 0.0, ...]
}
```

The environment has to respond in the following JSON format:

```json
{
    "observation": [0.0, 0.0, ...],
    "reward": 0.0,
    "terminated": false,
    "truncated": false,
    "extraValueNames": ["extraValue1", "extraValue2", ...],
    "extraValues": [0.0, 0.0, ...]
}
```
The `extraValueNames` and `extraValues` fields are optional and can be used to provide additional information that will be logged.
