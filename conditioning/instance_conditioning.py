import torch

import comfy.model_management

from .fusers_patch import FusersPatch
from .embeddings import prepare_embeddings


class InstanceConditioning:
    """
    This class masquerades as Gligen in order to trigger setup
    """

    def __init__(self, fusers, positionnet, fusers_batch_size):
        self.fusers_list = fusers['model_list']
        self.positionnet = positionnet['model']
        self.fusers_batch_size = fusers_batch_size
        self.conds = []
        self.current_device = comfy.model_management.intermediate_device()

        # Gligen hacks
        self.model = self
        self.load_device = comfy.model_management.get_torch_device()
        self.offload_device = comfy.model_management.intermediate_device()

    def get_fusers_patch(self, latent_shape, idxs, device):
        return FusersPatch(self.conds, self.fusers_list, self.fusers_batch_size, self.positionnet, latent_shape, idxs, device)

    def set_position(self, latent_shape, _, device):
        # Called in samplers by gligen cond to return middle attention patch
        batch_size = latent_shape[0]
        idxs = list(range(batch_size))
        fusers_patch = self.get_fusers_patch(latent_shape, idxs, device)
        return fusers_patch

    def add_conds(self, conds):
        self.conds.extend(conds)

    def get_models(self) -> list[torch.nn.Module]:
        # Used to get models for loading/offloading
        return [(None, model) for model in [*self.fusers_list, self.positionnet]]

    def inference_memory_requirements(self, dtype) -> int:
        # Used to calculate memory requirements by ControlNet
        return 0

    def is_clone(self, other):
        return other == self

    def model_size(self):
        return 0

    def model_patches_to(self, device_or_dtype):
        if device_or_dtype == torch.float16 or device_or_dtype == torch.float32:
            return
        if device_or_dtype is None:
            return
        self.positionnet = self.positionnet.to(device_or_dtype)
        for i, fuser in enumerate(self.fusers_list):
            self.fusers_list[i] = fuser.to(device_or_dtype)

    def model_dtype(self):
        # Fusers requires float32 so we will ignore this
        return torch.float32

    def patch_model(self, device_to):
        # TODO what is this for?
        return
