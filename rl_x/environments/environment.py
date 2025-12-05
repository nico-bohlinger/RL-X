class Environment:
    def __init__(self, name, get_default_config, create_train_and_eval_env, general_properties):
        self.name = name
        self.get_default_config = get_default_config
        self.create_train_and_eval_env = create_train_and_eval_env
        self.general_properties = general_properties
