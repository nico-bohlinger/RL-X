# Custom Environments

Contains a prototype for a custom environment interface with simple socket communication.

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

## Asynchronous environments
With the ```--config.environment.synchronized=False``` flag, the environment does not have to wait for every environment to step.  
Instead it will only wait for ```--config.environment.async_threshold=0.8``` percent of the environments to step and fills the remaining slots with dummy (observation, reward, terminated, truncated, info) tuples.