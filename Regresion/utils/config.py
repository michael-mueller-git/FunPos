import yaml

def read_yaml_config(config_file: str) -> dict:
    """ Parse a yaml config file

    Args:
        config_file (str): path to config file to parse

    Returns:
        dict: the configuration dictionary
    """
    with open(config_file) as f:
        return yaml.load(f, Loader = yaml.FullLoader)


CONFIG = read_yaml_config('config.yaml')
