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
import wandb
import gymnasium as gym

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

# Silences the box bound precision warning for cartpole
gym.logger.set_level(40)

rlx_logger = logging.getLogger("rl_x")


class Runner:
    def __init__(self, implementation_package_names=["rl_x"]):
        algorithm_name, environment_name, self._mode = self.parse_arguments(implementation_package_names)

        # Compatibility check
        algorithm_general_properties = get_algorithm_general_properties(algorithm_name)
        environment_general_properties = get_environment_general_properties(environment_name)
        if environment_general_properties.action_space_type not in algorithm_general_properties.action_space_types:
            raise ValueError("Incompatible action space type")
        if environment_general_properties.observation_space_type not in algorithm_general_properties.observation_space_types:
            raise ValueError("Incompatible observation space type")
        if environment_general_properties.data_interface_type not in algorithm_general_properties.data_interface_types:
            raise ValueError("Incompatible data interface type")

        # General Deep Learning framework settings
        algorithm_uses_torch = DeepLearningFrameworkType.TORCH == algorithm_general_properties.deep_learning_framework_type
        algorithm_uses_jax = DeepLearningFrameworkType.JAX == algorithm_general_properties.deep_learning_framework_type
        environment_uses_jax = SimulationType.JAX_BASED == environment_general_properties.simulation_type

        if algorithm_uses_torch:  
            # Avoids warning when TensorFloat32 is available
            import torch
            torch.set_float32_matmul_precision("high")
            # Silence UserWarning https://github.com/pytorch/pytorch/issues/109842
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, message=".*is deprecated, please use.*")
        elif algorithm_uses_jax or environment_uses_jax:
            # Guarantee enough memory for CUBLAS to initialize when using jax
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
            import jax
            # Spent most possible time to optimize execution time and memory usage
            jax.config.update("jax_exec_time_optimization_effort", 1.0)
            jax.config.update("jax_memory_fitting_effort", 1.0)
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


    def parse_arguments(self, implementation_package_names):
        algorithm_name = [arg for arg in sys.argv if arg.startswith("--algorithm.name=")]
        environment_name = [arg for arg in sys.argv if arg.startswith("--environment.name=")]
        runner_mode = [arg for arg in sys.argv if arg.startswith("--runner.mode=")]

        if algorithm_name:
            algorithm_name = algorithm_name[0].split("=")[1]
            del sys.argv[sys.argv.index("--algorithm.name=" + algorithm_name)]
        else:
            algorithm_name = DEFAULT_ALGORITHM
        
        for implementation_library_name in implementation_package_names:
            try:
                importlib.import_module(f"{implementation_library_name}.algorithms.{algorithm_name}")
                break
            except ModuleNotFoundError:
                pass
        
        if environment_name:
            environment_name = environment_name[0].split("=")[1]
            del sys.argv[sys.argv.index("--environment.name=" + environment_name)]
        else:
            environment_name = DEFAULT_ENVIRONMENT
        
        for implementation_library_name in implementation_package_names:
            try:
                importlib.import_module(f"{implementation_library_name}.environments.{environment_name}")
                break
            except ModuleNotFoundError:
                pass
        
        if runner_mode:
            runner_mode = runner_mode[0].split("=")[1]
            del sys.argv[sys.argv.index("--runner.mode=" + runner_mode)]
        else:
            runner_mode = DEFAULT_RUNNER_MODE

        return algorithm_name, environment_name, runner_mode


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
