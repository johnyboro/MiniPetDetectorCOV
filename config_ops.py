import yaml
from copy import deepcopy


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_dict(data, prefix=""):
    flattened = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def set_nested_value(data, dotted_key, value):
    keys = dotted_key.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def apply_overrides(base_config, overrides):
    config = deepcopy(base_config)
    for key, value in overrides.items():
        if "." in key:
            set_nested_value(config, key, value)
        elif (
            key in config and isinstance(config[key], dict) and isinstance(value, dict)
        ):
            config[key].update(value)
        else:
            config[key] = value
    return config

