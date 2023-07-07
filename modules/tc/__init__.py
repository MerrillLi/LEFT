from modules.tc.ltp import LTP
from modules.tc.ntf import NTF
from modules.tc.ntc import NTC
from modules.tc.ncp import NCP
from modules.tc.costco import CoSTCo
from modules.tc.meta_tc import MetaTC

def get_model(args):
    if args.model == 'LTP':
        model = LTP(args)
    elif args.model == 'NTF':
        model = NTF(args)
    elif args.model == 'NTC':
        model = NTC(args)
    elif args.model == 'NCP':
        model = NCP(args)
    elif args.model == 'CoSTCo':
        model = CoSTCo(args)
    else:
        raise NotImplementedError
    return model.to(args.device)
