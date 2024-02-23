import torch


class FusersPatch(torch.nn.Module):
  def __init__(self, fusers_list, objs, drop_box_mask, device, fusers_batch_size=5):
    super(FusersPatch, self).__init__()
    self.fusers_list = fusers_list
    self.transformer_options = {'objs': objs, 'drop_box_mask': drop_box_mask}
    self.device = device
    self.fusers_batch_size = fusers_batch_size
    # call_idx traces where in the UNet this is being called from
    # a hack around lack of Comfy support
    self.call_idx = 0

  @torch.no_grad()
  def forward(self, x, extra_options):
    fuser = self.fusers_list[self.call_idx %
                             len(self.fusers_list)]
    attn = fuser(x, self.transformer_options)
    self.call_idx += 1
    return attn.to(torch.float16)
