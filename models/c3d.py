import torch.nn as nn
import torch
import math

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    maxpool_count = 0
    for v in cfg:
        if v == 'M':
            maxpool_count += 1
            if maxpool_count==1:
                layers += [nn.MaxPool3d(kernel_size=(1, 7, 7), stride=(1, 7, 7))]
                # layers += [nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))]
            elif maxpool_count==5:
                layers += [nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=(0,1,1))]
            else:
                layers += [nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=(3,3,3), padding=(1,1,1))
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

class C3D(nn.Module):
    """
    The C3D network as described in [1].
        References
        ----------
       [1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks."
       Proceedings of the IEEE international conference on computer vision. 2015.
    """

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def __init__(self):
        super(C3D, self).__init__()
        self.features = make_layers(cfg['A'], batch_norm=False)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x