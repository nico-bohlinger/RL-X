import os

# Silence tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

# Fix wandb connection issues on slow clusters - https://github.com/wandb/wandb/issues/3911#issuecomment-1409769887
os.environ["WANDB__SERVICE_WAIT"] = "600"

import sys
import importlib
from absl import app
from absl import flags
from absl import logging as absl_logging
import logging
import logging.handlers
from ml_collections import config_dict, config_flags

from rl_x.runner.runner_mode import RunnerMode
from rl_x.runner.default_config import get_config as get_runner_config
from rl_x.algorithms.algorithm_manager import get_algorithm_config, get_algorithm_model_class, get_algorithm_general_properties
from rl_x.environments.environment_manager import get_environment_config, get_environment_create_env, get_environment_general_properties
from rl_x.environments.data_interface_type import DataInterfaceType
from rl_x.environments.simulation_type import SimulationType
from rl_x.algorithms.deep_learning_framework_type import DeepLearningFrameworkType

DEFAULT_ALGORITHM = "ppo.pytorch"
DEFAULT_ENVIRONMENT = "gym.mujoco.humanoid_v4"
DEFAULT_RUNNER_MODE = "train"

# Silence jax logging
absl_logging.set_verbosity(absl_logging.ERROR)

rlx_logger = logging.getLogger("rl_x")


class Runner:
    def __init__(self, implementation_package_names=["rl_x"]):
        algorithm_name, environment_name, self._mode, jax_cache_dir = self.parse_arguments()

        # Early importing of environment to start Isaac if needed
        self.import_environment(environment_name, implementation_package_names)
        environment_general_properties = get_environment_general_properties(environment_name)
        self.environment_uses_isaac_lab = SimulationType.ISAAC_LAB == environment_general_properties.simulation_type

        if self.environment_uses_isaac_lab:
            import argparse
            from isaaclab.app import AppLauncher

            app_launcher_args = argparse.Namespace()

            def get_env_config_value(arg_name):
                arg_value = [arg for arg in sys.argv if arg.startswith(f"--environment.{arg_name}=")]
                if arg_value:
                    arg_value = arg_value[0].split("=")[1]
                else:
                    arg_value = getattr(get_environment_config(environment_name), arg_name, None)
                return arg_value
            
            app_launcher_args.disable_fabric = get_env_config_value("disable_fabric")
            app_launcher_args.num_envs = get_env_config_value("nr_envs")
            app_launcher_args.task = get_env_config_value("name")
            render = get_env_config_value("render")
            if isinstance(render, str):
                render = render.lower() == "true"
            app_launcher_args.headless = not render
            app_launcher_args.livestream = get_env_config_value("livestream")
            app_launcher_args.enable_cameras = get_env_config_value("enable_cameras")
            app_launcher_args.xr = get_env_config_value("xr")
            app_launcher_args.device = "cuda:0" if get_env_config_value("device") == "gpu" else get_env_config_value("device")
            app_launcher_args.cpu = get_env_config_value("cpu")
            app_launcher_args.verbose = get_env_config_value("verbose")
            app_launcher_args.info = get_env_config_value("info")
            app_launcher_args.experience = get_env_config_value("experience")
            app_launcher_args.rendering_mode = get_env_config_value("rendering_mode")
            app_launcher_args.kit_args = get_env_config_value("kit_args")
            app_launcher_args.anim_recording_enabled = get_env_config_value("anim_recording_enabled")
            app_launcher_args.anim_recording_start_time = get_env_config_value("anim_recording_start_time")
            app_launcher_args.anim_recording_stop_time = get_env_config_value("anim_recording_stop_time")

            app_launcher = AppLauncher(app_launcher_args)
            self.isaac_simulation_app = app_launcher.app

        # Compatibility check
        self.import_algorithm(algorithm_name, implementation_package_names)
        algorithm_general_properties = get_algorithm_general_properties(algorithm_name)
        if environment_general_properties.action_space_type not in algorithm_general_properties.action_space_types:
            raise ValueError(f"Incompatible action space type. Environment: {environment_general_properties.action_space_type}, Algorithm: {algorithm_general_properties.action_space_types}")
        if environment_general_properties.observation_space_type not in algorithm_general_properties.observation_space_types:
            raise ValueError(f"Incompatible observation space type. Environment: {environment_general_properties.observation_space_type}, Algorithm: {algorithm_general_properties.observation_space_types}")
        if environment_general_properties.data_interface_type not in algorithm_general_properties.data_interface_types:
            raise ValueError(f"Incompatible data interface type. Environment: {environment_general_properties.data_interface_type}, Algorithm: {algorithm_general_properties.data_interface_types}")

        # General Deep Learning framework settings
        algorithm_uses_torch = DeepLearningFrameworkType.TORCH == algorithm_general_properties.deep_learning_framework_type
        algorithm_uses_jax = DeepLearningFrameworkType.JAX == algorithm_general_properties.deep_learning_framework_type
        environment_uses_jax = SimulationType.JAX_BASED == environment_general_properties.simulation_type
        environment_uses_torch = SimulationType.ISAAC_LAB == environment_general_properties.simulation_type or SimulationType.MANISKILL == environment_general_properties.simulation_type

        import gymnasium as gym
        # Silences the box bound precision warning for cartpole
        gym.logger.set_level(40)

        if algorithm_uses_torch:  
            # Avoids warning when TensorFloat32 is available
            import torch
            torch.set_float32_matmul_precision("high")
            # Silence UserWarning https://github.com/pytorch/pytorch/issues/109842
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, message=".*is deprecated, please use.*")
            # Check device
            if algorithm_uses_torch and environment_uses_torch:
                alg_device = [arg for arg in sys.argv if arg.startswith("--algorithm.device=")]
                if alg_device:
                    alg_device = alg_device[0].split("=")[1]
                else:
                    alg_device = getattr(get_algorithm_config(algorithm_name), "device", None)
                env_device = [arg for arg in sys.argv if arg.startswith("--environment.device=")]
                if env_device:
                    env_device = env_device[0].split("=")[1]
                else:
                    env_device = getattr(get_environment_config(environment_name), "device", None)
                if alg_device and env_device and alg_device != env_device:
                    raise ValueError("Incompatible device types between algorithm and environment")
        elif algorithm_uses_jax or environment_uses_jax:
            # Guarantee enough memory for CUBLAS to initialize when using jax
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
            import jax
            # Spent most possible time to optimize execution time and memory usage
            jax.config.update("jax_exec_time_optimization_effort", 1.0)
            jax.config.update("jax_memory_fitting_effort", 1.0)
            # Enable jax cache
            jax.config.update("jax_compilation_cache_dir", jax_cache_dir)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
            jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
            # Set device
            alg_device = None
            if algorithm_uses_jax:
                alg_device = [arg for arg in sys.argv if arg.startswith("--algorithm.device=")]
                if alg_device:
                    alg_device = alg_device[0].split("=")[1]
                else:
                    alg_device = getattr(get_algorithm_config(algorithm_name), "device", None)
            env_device = None
            if environment_uses_jax:
                env_device = [arg for arg in sys.argv if arg.startswith("--environment.device=")]
                if env_device:
                    env_device = env_device[0].split("=")[1]
                else:
                    env_device = getattr(get_environment_config(environment_name), "device", None)
            if alg_device and env_device and alg_device != env_device:
                raise ValueError("Incompatible device types between algorithm and environment")
            device = alg_device or env_device
            if device == "cpu":
                jax.config.update("jax_platform_name", "cpu")
            try:
                jax.default_backend()
            except:
                pass

        self._model_class = get_algorithm_model_class(algorithm_name)
        self._create_env = get_environment_create_env(environment_name)
        
        runner_default_config = get_runner_config(self._mode)
        algorithm_default_config = get_algorithm_config(algorithm_name)
        environment_default_config = get_environment_config(environment_name)
        self._runner_config_flag = config_flags.DEFINE_config_dict("runner", runner_default_config)
        self._algorithm_config_flag = config_flags.DEFINE_config_dict("algorithm", algorithm_default_config)
        self._environment_config_flag = config_flags.DEFINE_config_dict("environment", environment_default_config)

        # Logging
        rlx_logger = logging.getLogger("rl_x")
        rlx_logger.setLevel(logging.INFO)
        rlx_logger.propagate = False
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setLevel(logging.INFO)
        consoleHandler.setFormatter(logging.Formatter("[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s","%m-%d %H:%M:%S"))
        memoryHandler = logging.handlers.MemoryHandler(100, logging.ERROR, consoleHandler)
        rlx_logger.addHandler(memoryHandler)
        def info(msg, flush=True, *args, **kwargs):
            if rlx_logger.isEnabledFor(logging.INFO):
                rlx_logger._log(logging.INFO, msg, args, stacklevel=2, **kwargs)
            if flush:
                rlx_logger.handlers[0].flush()
        rlx_logger.info = info
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            rlx_logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        sys.excepthook = handle_exception


    def parse_arguments(self):
        algorithm_name = [arg for arg in sys.argv if arg.startswith("--algorithm.name=")]
        environment_name = [arg for arg in sys.argv if arg.startswith("--environment.name=")]
        runner_mode = [arg for arg in sys.argv if arg.startswith("--runner.mode=")]
        jax_cache_dir = [arg for arg in sys.argv if arg.startswith("--runner.jax_cache_dir=")]

        if algorithm_name:
            algorithm_name = algorithm_name[0].split("=")[1]
            del sys.argv[sys.argv.index("--algorithm.name=" + algorithm_name)]
        else:
            algorithm_name = DEFAULT_ALGORITHM
        
        if environment_name:
            environment_name = environment_name[0].split("=")[1]
            del sys.argv[sys.argv.index("--environment.name=" + environment_name)]
        else:
            environment_name = DEFAULT_ENVIRONMENT
        
        if runner_mode:
            runner_mode = runner_mode[0].split("=")[1]
            del sys.argv[sys.argv.index("--runner.mode=" + runner_mode)]
        else:
            runner_mode = DEFAULT_RUNNER_MODE

        if jax_cache_dir:
            jax_cache_dir = jax_cache_dir[0].split("=")[1]
            del sys.argv[sys.argv.index("--runner.jax_cache_dir=" + jax_cache_dir)]
        else:
            jax_cache_dir = get_runner_config(None).jax_cache_dir

        return algorithm_name, environment_name, runner_mode, jax_cache_dir


    def import_environment(self, environment_name, implementation_package_names):
        for implementation_library_name in implementation_package_names:
            try:
                importlib.import_module(f"{implementation_library_name}.environments.{environment_name}")
                break
            except ModuleNotFoundError:
                pass


    def import_algorithm(self, algorithm_name, implementation_package_names):
        for implementation_library_name in implementation_package_names:
            try:
                importlib.import_module(f"{implementation_library_name}.algorithms.{algorithm_name}")
                break
            except ModuleNotFoundError:
                pass


    def run(self):
        if self._mode == RunnerMode.SHOW_CONFIG:
            main_func = self._show_config
        elif self._mode == RunnerMode.TRAIN:
            main_func = self._train
        elif self._mode == RunnerMode.TEST:
            main_func = self._test
        else:
            raise ValueError("Invalid mode")

        try:
            app.run(main_func)
        except KeyboardInterrupt:
            rlx_logger.warning("KeyboardInterrupt")

    
    def init_config(self):
        self._config = config_dict.ConfigDict()
        self._config.runner = self._runner_config_flag.value
        self._config.algorithm = self._algorithm_config_flag.value
        self._config.environment = self._environment_config_flag.value


    def _show_config(self, _):
        self.init_config()
        rlx_logger.info("\n" + str(self._config))


    def _train(self, _):
        self.init_config()

        if self._config.runner.track_wandb:
            import wandb
            wandb.init(
                entity=self._config.runner.wandb_entity,
                project=self._config.runner.project_name,
                group=self._config.runner.exp_name,
                name=self._config.runner.run_name,
                notes=self._config.runner.notes,
                sync_tensorboard=True,
                config=self._config.to_dict(),
                monitor_gym=True,
                save_code=True,
            )

        run_path = f"runs/{self._config.runner.project_name}/{self._config.runner.exp_name}/{self._config.runner.run_name}"
        run_path = os.path.abspath(run_path)
        writer = None
        if self._config.runner.track_tb:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(run_path)
            all_config_items = self._config.runner.items() + self._config.algorithm.items() + self._config.environment.items()
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in all_config_items])),
            )

        env = self._create_env(self._config)
        
        if self._config.runner.load_model:
            explicitly_set_algorithm_params = [param_name for param_name in self._algorithm_config_flag._flagvalues if param_name.startswith("algorithm.")]
            model = self._model_class.load(self._config, env, run_path, writer, explicitly_set_algorithm_params)
        else:
            model = self._model_class(self._config, env, run_path, writer)

        try:
            model.train()
        finally:
            env.close()
            if self._config.runner.track_tb:
                writer.close()
            if self._config.runner.track_wandb:
                wandb.finish()
            if self.environment_uses_isaac_lab:
                self.isaac_simulation_app.close()


    def _test(self, _):
        self.init_config()
        
        if self._config.runner.track_wandb:
            raise ValueError("Wandb is not supported in test mode")
        if self._config.runner.track_tb:
            raise ValueError("Tensorboard is not supported in test mode")
        if self._config.runner.save_model:
            raise ValueError("Saving model is not supported in test mode")
        
        run_path = f"runs/{self._config.runner.project_name}/{self._config.runner.exp_name}/{self._config.runner.run_name}"
        run_path = os.path.abspath(run_path)

        env = self._create_env(self._config)
        
        if self._config.runner.load_model:
            explicitly_set_algorithm_params = [param_name for param_name in self._algorithm_config_flag._flagvalues if param_name.startswith("algorithm.")]
            model = self._model_class.load(self._config, env, run_path, None, explicitly_set_algorithm_params)
        else:
            model = self._model_class(self._config, env, run_path, None)
        
        try:
            model.test(self._config.runner.nr_test_episodes)
        finally:
            env.close()
            if self.environment_uses_isaac_lab:
                self.isaac_simulation_app.close()
