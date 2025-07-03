import torch.nn as nn
from architecture.folding import FoldingEncoder, FoldingDecoder
from architecture.gae import GraphEncoder, GraphDecoder
from architecture.tgae import TransformerPCEncoder, TransformerPCDecoder


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
    model (nn.Module): The PyTorch model.

    Returns:
    int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_features=512

fenc = FoldingEncoder(num_features=num_features).train()
fdec = FoldingDecoder(in_channel=num_features, pe_type='sphere').train()
genc = GraphEncoder(num_features=num_features,film_flag=False).train()
gdec = GraphDecoder(num_features=num_features).train()
tenc = TransformerPCEncoder(num_features=num_features, film_flag=False)
tdec = TransformerPCDecoder(num_features=num_features, num_layers = 2)



fenc_c = count_trainable_parameters(fenc)
fdec_c = count_trainable_parameters(fdec)
genc_c = count_trainable_parameters(genc)
gdec_c = count_trainable_parameters(gdec)
tenc_c = count_trainable_parameters(tenc)
tdec_c = count_trainable_parameters(tdec)


print(f"{fenc_c}; Folding ENC")
print(f"{fdec_c}; Folding DEC")
print(f"{genc_c}; GAE ENC")
print(f"{gdec_c}; GAE DEC")
print(f"{tenc_c}; TGAE ENC")
print(f"{tdec_c}; TGAE DEC")
