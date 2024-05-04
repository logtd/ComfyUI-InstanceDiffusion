import torch

#from ..modules.attention import GatedSelfAttentionDense
from comfy.gligen import GatedSelfAttentionDense as GSAD

class GatedSelfAttentionDense(GSAD):
    def forward(self, x, instance_options={}):
        objs = instance_options['objs']
        x = super().forward(x, objs)
        return x.to(torch.float16)

def prepare_fusers(fusers_ckpt) -> list[torch.nn.Module]:
    fusers_list = []
    for key in fusers_ckpt['input_blocks']:
        fusers_ckpt['input_blocks'][key]['params']['query_dim'] = fusers_ckpt['input_blocks'][key]['params']['n_heads'] * \
            fusers_ckpt['input_blocks'][key]['params']['d_head']
        fuser = GatedSelfAttentionDense(
            **fusers_ckpt['input_blocks'][key]['params'])
        fuser.load_state_dict(fusers_ckpt['input_blocks'][key]['state'])
        fusers_list.append(fuser)

    fusers_ckpt['middle_block']['1']['params']['query_dim'] = fusers_ckpt['middle_block']['1']['params']['n_heads'] * \
        fusers_ckpt['middle_block']['1']['params']['d_head']
    fuser = GatedSelfAttentionDense(
        **fusers_ckpt['middle_block']['1']['params'])
    fuser.load_state_dict(fusers_ckpt['middle_block']['1']['state'])
    fusers_list.append(fuser)

    for key in fusers_ckpt['output_blocks']:
        fusers_ckpt['output_blocks'][key]['params']['query_dim'] = fusers_ckpt['output_blocks'][key]['params']['n_heads'] * \
            fusers_ckpt['output_blocks'][key]['params']['d_head']
        fuser = GatedSelfAttentionDense(
            **fusers_ckpt['output_blocks'][key]['params'])
        fuser.load_state_dict(fusers_ckpt['output_blocks'][key]['state'])
        fusers_list.append(fuser)

    return fusers_list
