class Algorithm:
    def __init__(self, name, get_default_config, get_model_class):
        self.name = name
        self.get_default_config = get_default_config
        self.get_model_class = get_model_class
