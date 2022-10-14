from absl import app
from absl import flags
from absl import logging as absl_logging
import logging
from ml_collections import config_dict, config_flags
import wandb
from torch.utils.tensorboard import SummaryWriter

from rl_x.algorithms.algorithm_manager import Algorithm
from rl_x.environments.environment_manager import Environment
from rl_x.algorithms.algorithm_manager import AlgorithmManager
from rl_x.environments.environment_manager import EnvironmentManager


# Change with newer jax version
# https://github.com/google/jax/issues/10070
# https://github.com/google/jax/pull/12769
absl_logging.set_verbosity(absl_logging.ERROR)

rlx_logger = logging.getLogger('rl_x')
rlx_logger.setLevel(logging.INFO)


class Runner:
    def __init__(self, algorithm: Algorithm, environment: Environment):
        self._model_class = AlgorithmManager.get_model_class(algorithm)
        self._create_env = EnvironmentManager.get_create_env(environment)
        
        algorithm_default_config = AlgorithmManager.get_default_config(algorithm, environment)
        environment_default_config = EnvironmentManager.get_default_config(algorithm, environment)
        combined_default_config = config_dict.ConfigDict()
        combined_default_config.algorithm = algorithm_default_config
        combined_default_config.environment = environment_default_config
        self._config_flag = config_flags.DEFINE_config_dict("config", combined_default_config)


    def run_and_show_config(self):
        def _show_config(_):
            self._config = self._config_flag.value
            rlx_logger.info("\n" + str(self._config))
        app.run(_show_config)
    

    def run_experiment(self):
        app.run(self._main)


    def _main(self, _):
        self._config = self._config_flag.value
        if self._config.algorithm.mode == "train":
            self._train()
        elif self._config.algorithm.mode == "test":
            self._test()


    def _train(self):
        if self._config.algorithm.wandb_track:
            wandb.init(
                project=self._config.algorithm.project_name,
                group=self._config.algorithm.exp_name,
                name=self._config.algorithm.run_name,
                sync_tensorboard=True,
                config=self._config.to_dict(),
                monitor_gym=True,
                save_code=True,
            )

        if self._config.algorithm.tb_track:
            writer = SummaryWriter(self._config.algorithm.run_path)
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in self._config.algorithm.items() + self._config.environment.items()])),
            )
        
        env = self._create_env(self._config)
        
        model = self._model_class(self._config, env)

        try:
            model.train()
        finally:
            env.close()
            if self._config.algorithm.tb_track:
                writer.close()
            if self._config.algorithm.wandb_track:
                wandb.finish()


    def _test(self):
        pass
