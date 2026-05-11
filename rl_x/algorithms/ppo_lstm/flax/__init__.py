from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.ppo_lstm.flax.ppo_lstm import PPO_LSTM
from rl_x.algorithms.ppo_lstm.flax.default_config import get_config
from rl_x.algorithms.ppo_lstm.flax.general_properties import GeneralProperties


PPO_LSTM_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(PPO_LSTM_FLAX, get_config, PPO_LSTM, GeneralProperties)
