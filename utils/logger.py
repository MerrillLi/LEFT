from loguru import logger
import sys
import warnings


def get_log_filename(args):

    if args.model == 'CPals':
        return f"{args.dataset}_{args.density}_{args.model}_{args.rank}_{args.lmda}"
    elif args.model == 'CPsgd':
        return f"{args.dataset}_{args.density}_{args.model}_{args.rank}_{args.lr}"
    elif args.model == 'NCP':
        return f"{args.dataset}_{args.density}_{args.model}_{args.rank}"
    elif args.model == 'NTC':
        return f"{args.dataset}_{args.density}_{args.model}_{args.rank}_{args.channels}"
    elif args.model == 'NTM':
        return f"{args.dataset}_{args.density}_{args.model}_{args.rank}"
    elif args.model == 'NTF':
        return f"{args.dataset}_{args.density}_{args.model}_{args.rank}_{args.window}"
    elif args.model == 'LTP':
        return f"{args.dataset}_{args.density}_{args.model}_{args.rank}_{args.window}"
    elif args.model == 'CoSTCo':
        return f"{args.dataset}_{args.density}_{args.model}_{args.rank}_{args.channels}"


def setup_logger(args, path):
    warnings.filterwarnings("ignore", module='pandas')
    logger.remove()
    format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | " \
             "<level>{level: <8}</level> | " \
             "<level>{message}</level>"

    logger.add(sys.stdout, format=format, colorize=True)

    logfilename = get_log_filename(args)
    logger.add(f"./results/{path}/{args.model}/{logfilename}.log", format=format)
