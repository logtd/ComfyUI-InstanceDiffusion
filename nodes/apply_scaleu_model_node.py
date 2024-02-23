import comfy.model_management

from .. import constants as constants
from ..model_helpers.prepare_scaleu import get_scaleu_patch


class ApplyScaleUModelNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "scaleu": ("SCALEU",),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "instance"

    def apply(self, model, scaleu):
        # Validate patches dict is setup correctly
        transformer_options = model.model_options['transformer_options']
        if 'patches' not in transformer_options:
            transformer_options['patches'] = {}

        if 'output_block_patch' not in transformer_options['patches']:
            transformer_options['patches']['output_block_patch'] = []

        # Add scaleu patch to model patches
        scaleu_nets = scaleu['model_list']
        # TODO make this load in KSampler
        for i, scaleu in enumerate(scaleu_nets):
            scaleu_nets[i] = scaleu.to(
                comfy.model_management.get_torch_device())
        transformer_options['patches']['output_block_patch'].append(
            get_scaleu_patch(scaleu_nets))
        return (model,)
