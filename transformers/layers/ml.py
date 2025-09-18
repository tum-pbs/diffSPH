import torch
import torch
import inspect
import re
import torch.nn as nn
from diffSPH.sphOperations.shared import scatter_sum

def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ScatterSumLayer(nn.Module):
    def __init__(self):
        super(ScatterSumLayer, self).__init__()
    def forward(self, input, index, dim, dim_size):
        return scatter_sum(input, index, dim=dim, dim_size=dim_size)
    

# outputDecoder = buildMLPwDict({
#     'inputFeatures': latentSpaceSize,
#     'output': gt.shape[1],
#     'activation': 'celu',
#     'layout': [hiddenMLPSize] * outputDecoderLayers,
#     'norm': False,
#     'bias': False,
# }
# ).to(device)

# numberOfParameters = sum(p.numel() for p in outputDecoder.parameters())
# print(f'Number of parameters in output decoder: {numberOfParameters}')

# for i, layer in enumerate(outputDecoder):
#     print(f'Layer {i}: {layer}')