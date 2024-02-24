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
    def __init__(self, conds, fusers_list, fusers_batch_size, positionnet, latent_shape, idxs, device):
        super(FusersPatch, self).__init__()
        self.conds = conds
        self.fusers_list = fusers_list
        self.fusers_batch_size = fusers_batch_size
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

    def _get_idxs(self, extra_options):
        idxs = None
        if extra_options is not None:
            if 'ad_params' in extra_options:
                idxs = extra_options['ad_params']['sub_idxs']
            elif 'sub_idxs' in extra_options:
                idxs = extra_options['sub_idxs']
        
        return idxs

    @torch.no_grad()
    def forward(self, x, extra_options):
        print('--------', x.shape)
        block, idx = extra_options['block']
        fuser_idx = block_map[block][idx]
        fuser = self.fusers_list[fuser_idx]
        attn_total = []
        idxs = self._get_idxs(extra_options)
        print('idxs', idxs)
        for i in range(0, len(x), self.fusers_batch_size):
            batch_idxs = idxs[i:i+self.fusers_batch_size]
            x_idxs = list(range(i, i+self.fusers_batch_size))
            # print('batch_idxs', batch_idxs, x_idxs)
            # print('x[x_idxs].shape', x[x_idxs].shape)
            # print('doing attn')
            attn = fuser(x[x_idxs], self._get_position_objs(batch_idxs))
            # print('attn.shape', attn.shape)
            attn_total.append(attn)
        attn_total = torch.cat(attn_total)
        print(attn_total.shape)
        return attn_total.to(torch.float16)
