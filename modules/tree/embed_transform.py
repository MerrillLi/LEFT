from torch.nn import *

class EmbedTransform(Module):

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.net = Sequential(Linear(in_feats, out_feats),
                                ReLU(),
                                Dropout(p=0.2),
                                Linear(out_feats, out_feats))


    def forward(self, x):
        return self.net(x)
