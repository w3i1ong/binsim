
def check_and_set_default(config_name, kwargs: dict, default_value: dict = None, required_kwargs: set = None,
                          logger=None):
    if default_value is not None:
        for key, value in default_value.items():
            if key not in kwargs:
                if logger is not None:
                    logger.warning(f"The key {key} in {config_name} is not specified, use default value {value}.")
                kwargs[key] = value
    if required_kwargs is not None:
        for key in required_kwargs:
            assert key in kwargs, f"The key {key} in {config_name} is required, but not specified."
    return kwargs
