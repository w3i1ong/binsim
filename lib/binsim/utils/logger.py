import logging

def init_logger(logger_name,
                level=logging.INFO,
                console=True,
                console_level = logging.INFO,
                file=None,
                file_level = logging.INFO):
    if isinstance(level, str): level = level.upper()
    if isinstance(console_level, str): console_level = console_level.upper()
    if isinstance(file_level, str): file_level = file_level.upper()


    formatter = logging.Formatter('%(asctime)s, [%(levelname)s]::%(name)s: %(message)s')
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    for handler in logger.handlers:
        logger.removeHandler(handler)
        handler.close()

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(console_level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if file is not None:
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level)
        logger.addHandler(file_handler)
    return logger