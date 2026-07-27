from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.ppo_lstm.pytorch.ppo_lstm import PPO_LSTM
from rl_x.algorithms.ppo_lstm.pytorch.default_config import get_config
from rl_x.algorithms.ppo_lstm.pytorch.general_properties import GeneralProperties


PPO_LSTM_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(PPO_LSTM_PYTORCH, get_config, PPO_LSTM, GeneralProperties)
