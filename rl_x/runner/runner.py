import os
import sys
from absl import app
from absl import flags
from absl import logging as absl_logging
import logging
from ml_collections import config_dict, config_flags
import wandb
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

from rl_x.runner.runner_mode import RunnerMode
from rl_x.runner.default_config import get_config as get_runner_config
from rl_x.algorithms.algorithm_manager import get_algorithm_config, get_algorithm_model_class
from rl_x.environments.environment_manager import get_environment_config, get_environment_create_env


# Silence jax logging
absl_logging.set_verbosity(absl_logging.ERROR)

# Silences the box bound precision warning for cartpole
gym.logger.set_level(40)

# Silence tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Guarantee enough memory for CUBLAS to initialize when using jax
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

rlx_logger = logging.getLogger("rl_x")


class Runner:
    def __init__(self, algorithm_name, environment_name):
        self._model_class = get_algorithm_model_class(algorithm_name)
        self._create_env = get_environment_create_env(environment_name)
        
        runner_default_config = get_runner_config()
        algorithm_default_config = get_algorithm_config(algorithm_name)
        environment_default_config = get_environment_config(environment_name)
        combined_default_config = config_dict.ConfigDict()
        combined_default_config.runner = runner_default_config
        combined_default_config.algorithm = algorithm_default_config
        combined_default_config.environment = environment_default_config
        self._config_flag = config_flags.DEFINE_config_dict("config", combined_default_config)

        # Logging
        rlx_logger = logging.getLogger("rl_x")
        rlx_logger.setLevel(logging.INFO)
        rlx_logger.propagate = False
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.INFO)
        consoleHandler.setFormatter(logging.Formatter("[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s","%m-%d %H:%M:%S"))
        rlx_logger.addHandler(consoleHandler)
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            rlx_logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        sys.excepthook = handle_exception

    

    def run(self, mode: RunnerMode):
        if mode == RunnerMode.SHOW_CONFIG:
            main_func = self._show_config
        elif mode == RunnerMode.TRAIN:
            main_func = self._train
        elif mode == RunnerMode.TEST:
            main_func = self._test
        else:
            raise ValueError("Invalid mode")

        try:
            app.run(main_func)
        except KeyboardInterrupt:
            rlx_logger.warning("KeyboardInterrupt")


    def _show_config(self, _):
        self._config = self._config_flag.value
        rlx_logger.info("\n" + str(self._config))


    def _train(self, _):
        self._config = self._config_flag.value

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
        writer = None
        if self._config.runner.track_tb:
            writer = SummaryWriter(run_path)
            all_config_items = self._config.runner.items() + self._config.algorithm.items() + self._config.environment.items()
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in all_config_items])),
            )

        env = self._create_env(self._config)
        
        if self._config.runner.load_model:
            model = self._model_class.load(self._config, env, run_path, writer)
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
        if self._config.runner.track_wandb:
            raise ValueError("Wandb is not supported in test mode")
        if self._config.runner.track_tb:
            raise ValueError("Tensorboard is not supported in test mode")
        if self._config.runner.save_model:
            raise ValueError("Saving model is not supported in test mode")
        
        run_path = f"runs/{self._config.runner.project_name}/{self._config.runner.exp_name}/{self._config.runner.run_name}"

        env = self._create_env(self._config)
        
        if self._config.runner.load_model:
            model = self._model_class.load(self._config, env, run_path, None)
        else:
            model = self._model_class(self._config, env, run_path, None)
        
        try:
            model.test(self._config.runner.nr_test_episodes)
        finally:
            env.close()
