import json
from collections import OrderedDict

import torch

from electricity_zinb_patches import ImprovedZINBElectricityMeterEncoder

sd = torch.load('checkpoints_paper_s43/best_ema.pt',
                map_location='cpu', weights_only=True)

print('--- version-hint keys ---')
for k in sd:
    if any(s in k for s in ['glm', 'baseline_blend', 'tariff_season', 'stable']):
        print(' ', k, tuple(sd[k].shape))

print('--- input embedding first layer ---')
for k in sd:
    if 'rate_emb.mlp.0.weight' in k:
        print(' ', k, tuple(sd[k].shape))

print('--- total params ---')
print(' ', sum(v.numel() for v in sd.values()))

with open("static_cardinalities_ramz.json") as f:
    cards = json.load(f, object_pairs_hook=OrderedDict)

M = ImprovedZINBElectricityMeterEncoder(d_model=192, n_heads=6, n_layers=5,
        static_cardinalities=cards, dropout=0.05,
        )
print(sum(p.numel() for p in M.parameters() if p.requires_grad))
for n,mod in M.named_children():
    print(n, sum(p.numel() for p in mod.parameters() if p.requires_grad))