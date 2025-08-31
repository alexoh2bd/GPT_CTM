import logging
import os

def setup_logger(log_dir="logs", log_file="train.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("TransformerTrainer")
    logger.setLevel(logging.INFO)  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Avoid duplicated handlers
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        console_handler.setFormatter(console_format)

        # File handler
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(file_format)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
