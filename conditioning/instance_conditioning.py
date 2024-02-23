import torch

import comfy.model_management

from .embeddings import prepare_embeddings


class InstanceConditioning:
  """
  This class masquerades as both a ControlNet and Gligen in order to trigger setup
  """

  def __init__(self, fusers, positionnet, fusers_batch_size):
    self.fusers_list = fusers['model_list']
    self.positionnet = positionnet['model']
    self.fusers_batch_size = fusers_batch_size
    self.conds = []

    # ControlNet hacks
    self.load_device = comfy.model_management.get_torch_device()

    # Gligen hacks
    self.model = self

    # AnimateDiff hacks
    self.sub_idxs = None
    self.full_latent_length = None
    self.context_length = None

  def get_fusers_patch(self, objs, drop_box_mask):
    return

  def set_position(self, latent_shape, _, device):
    # Called in samplers by gligen cond to return middle attention patch
    batch_size = latent_shape[0]
    idxs = self.sub_idxs if self.sub_idxs is not None else list(
      range(batch_size))

    # TODO conds
    embeddings = prepare_embeddings(self.conds, latent_shape, idxs, True)
    objs, drop_box_mask = self.positionnet.to(device)(embeddings)
    fusers_patch = self.get_fusers_patch(objs, drop_box_mask, device)
    return fusers_patch

  def add_conds(self, conds):
    self.conds.extend(conds)

  def get_models(self) -> list[torch.nn.Module]:
    # Used to get models for loading/offloading
    # TODO need to make this work for ControlNet and Gligen
    return [(None, model) for model in [*self.fusers_list, self.positionnet]]

  def inference_memory_requirements(self, dtype) -> int:
    # Used to calculate memory requirements by ControlNet
    return 0

  def is_clone(self, other):
    return other == self

  # def __getattribute__(self, __name: str) -> Any:
  #   if __name == 'gligen':
  #     return 'something'  # TODO
  #   return super().__getattribute__(__name)
