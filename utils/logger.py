from loguru import logger
import sys
import time
import warnings

def setup_logger(args, path):
    warnings.filterwarnings("ignore", module='pandas')
    logfilename = time.asctime()
    logger.remove()
    format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | " \
             "<level>{level: <8}</level> | " \
             "<level>{message}</level>"

    logger.add(sys.stdout, format=format, colorize=True)
    logger.add(f"./results/{path}/{args.model}/{logfilename}.log", format=format)
