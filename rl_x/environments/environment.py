class Environment:
    def __init__(self, name, get_default_config, create_env, general_properties):
        self.name = name
        self.get_default_config = get_default_config
        self.create_env = create_env
        self.general_properties = general_properties
