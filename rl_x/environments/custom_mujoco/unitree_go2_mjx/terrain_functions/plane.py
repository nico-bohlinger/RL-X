import mujoco
import jax.numpy as jnp


class PlaneTerrainGeneration:
    def __init__(self, env):
        self.env = env
        self.xml_file_name = "plane.xml"
        self.uses_hfield = False
        self.ground_height = 0.0


    def init_attributes_with_model(self):
        if not jnp.all(self.env.initial_mjx_model.geom_type[self.env.foot_geom_indices] == 2):
            raise ValueError("Foot geoms are not of type sphere. Necessary for collision detection and the usage of the geom size attribute.")
        
        self.foot_name_geom_index_dict = {foot_name: mujoco.mj_name2id(self.env.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, foot_name) for foot_name in self.env.feet_names}


    def init(self, internal_state):
        internal_state["center_height"] = self.ground_height
        internal_state["robot_height_over_ground"] = self.env.initial_mj_model.keyframe("home").qpos[2] - internal_state["center_height"]
    

    def check_foot_floor_contact(self, data, mjx_model, internal_state, foot_name):
        foot_id = self.foot_name_geom_index_dict[foot_name]
        foot_size = mjx_model.geom_size[foot_id, 0]  # Assuming the foot is a sphere
        foot_xpos = data.geom_xpos[foot_id]
        foot_distance_to_floor = foot_xpos[2] - self.ground_height
        return foot_distance_to_floor < foot_size


    def pre_step(self, data, internal_state):
        internal_state["robot_height_over_ground"] = data.qpos[2] - internal_state["center_height"]


    def post_step(self, data, mjx_model, internal_state, key):
        return data


    def sample(self, mjx_model, internal_state, info, key):
        return mjx_model
