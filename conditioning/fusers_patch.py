import torch

from .embeddings import prepare_embeddings


block_map = {
    'input': {
        1: 0,
        2: 1,
        4: 2,
        5: 3,
        7: 4,
        8: 5
    },
    'middle': {
        0: 6,
    },
    'output': {
        3: 7,
        4: 8,
        5: 9,
        6: 10,
        7: 11,
        8: 12,
        9: 13,
        10: 14,
        11: 15
    }
}


class FusersPatch(torch.nn.Module):
    def __init__(self, conds, fusers_list, positionnet, latent_shape, idxs, device):
        super(FusersPatch, self).__init__()
        self.conds = conds
        self.fusers_list = fusers_list
        self.positionnet = positionnet
        self.latent_shape = latent_shape
        self.idxs = idxs
        self.device = device

    def _get_position_objs(self, idxs):
        embeddings = prepare_embeddings(
            self.conds, self.latent_shape, idxs, True)
        for key in embeddings:
            embeddings[key] = embeddings[key].to(self.device)
        objs, drop_box_mask = self.positionnet(embeddings)
        return {'objs': objs, 'drop_box_mask': drop_box_mask}

    def _get_idxs(self, x, extra_options):
        if extra_options is not None:
            if 'ad_params' in extra_options:
                return extra_options['ad_params']['sub_idxs']
            elif 'sub_idxs' in extra_options:
                return extra_options['sub_idxs']

        return list(range(x.shape[0]))

    @torch.no_grad()
    def forward(self, x, extra_options):
        block, idx = extra_options['block']
        fuser_idx = block_map[block][idx]
        fuser = self.fusers_list[fuser_idx]
        attn_total = []
        idxs = self._get_idxs(x, extra_options)

        attn_total = fuser(x, self._get_position_objs(idxs))
        return attn_total.to(torch.float16)
