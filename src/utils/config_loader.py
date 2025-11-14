import copy
import tomllib


def _convert_empty_strings_to_none(config):
    """
    Recursively convert empty strings to None in the config dictionary.
    This handles TOML's limitation of not supporting null values.

    Args:
        config: Configuration dictionary

    Returns:
        dict: Config with empty strings converted to None
    """
    if isinstance(config, dict):
        return {k: _convert_empty_strings_to_none(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_convert_empty_strings_to_none(item) for item in config]
    elif config == "":
        return None
    else:
        return config


def get_config(preset=None, config_path="configs/config.toml"):
    """
    Load configuration from a TOML file and apply a preset.

    Args:
        preset (str, optional): The name of the preset to apply. Defaults to None.
        config_path (str, optional): The path to the configuration file. Defaults to "config.toml".

    Returns:
        dict: The final configuration dictionary.
    """
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    if preset:
        presets = config.get("presets", {})
        preset_config = presets.get(preset)
        if not preset_config:
            raise ValueError(f"Preset '{preset}' not found in the configuration file.")

        # Deep copy the base config to avoid modifying it
        config = copy.deepcopy(config)

        # Merge preset values into the corresponding sections
        for key, value in preset_config.items():
            for section, settings in config.items():
                if isinstance(settings, dict) and key in settings:
                    config[section][key] = value

    # Convert empty strings to None (TOML doesn't support null)
    config = _convert_empty_strings_to_none(config)

    return config
