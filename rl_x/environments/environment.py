class Environment:
    def __init__(self, name, get_default_config, create_env):
        self.name = name
        self.get_default_config = get_default_config
        self.create_env = create_env
