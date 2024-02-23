import torch

from ..modules.scaleu import ScaleU


def get_scaleu_patch(scaleu_nets):
    def scaleu_patch(h, hsp, transformer_options):
        _, idx = transformer_options['block']
        sk = scaleu_nets[idx](h, hsp)
        return sk

    return scaleu_patch


def prepare_scaleu_nets(scaleu_ckpt) -> torch.nn.Module:
    scaleu_nets = []
    for i in range(12):
        ckpt = scaleu_ckpt[f'{i}']
        scaleu = ScaleU(True, len(ckpt['scaleu_b']), len(ckpt['scaleu_s']))
        scaleu.load_state_dict(ckpt)
        scaleu_nets.append(scaleu)
    return scaleu_nets
