from pathlib import Path
import mujoco
import numpy as np
from viewer import MujocoGlfwViewer


xml_path = (Path(__file__).resolve().parent / "data" / "ant.xml").as_posix()

model = mujoco.MjModel.from_xml_path(xml_path)

data = mujoco.MjData(model)

def user_warning_raise_exception(warning):
    if 'Pre-allocated constraint buffer is full' in warning:
        raise RuntimeError(warning + 'Increase njmax in mujoco XML')
    elif 'Pre-allocated contact buffer is full' in warning:
        raise RuntimeError(warning + 'Increase njconmax in mujoco XML')
    elif 'Unknown warning type' in warning:
        raise RuntimeError(warning + 'Check for NaN in simulation.')
    else:
        raise RuntimeError('Got MuJoCo Warning: ' + warning)

mujoco.set_mju_user_warning(user_warning_raise_exception)

dt = 0.01
nr_substeps = 1
nr_intermediate_steps = 1

viewer = MujocoGlfwViewer(model, dt * nr_substeps * nr_intermediate_steps)

for episode in range(100):
    mujoco.mj_resetData(model, data)
    for step in range(1000):
        action = np.random.randn(model.nu) * 30.0
        for _ in range(nr_intermediate_steps):
            data.ctrl = action
            mujoco.mj_step(model, data, nr_substeps)
        viewer.render(data, False)
    print('Episode', episode)