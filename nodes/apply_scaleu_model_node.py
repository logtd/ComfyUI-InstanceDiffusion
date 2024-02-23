from .. import constants as constants
from ..models.prepare_scaleu import get_scaleu_patch


class ApplyScaleUModel:
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
    if 'patches' not in model or model['patches'] is None:
      model['patches'] = {}

    if 'output_block_patch' not in model['patches']:
      model['patches']['output_block_patch'] = []

    # Add scaleu patch to model patches
    scaleu_nets = scaleu['model_list']
    model['patches']['output_block_patch'].append(
      get_scaleu_patch(scaleu_nets))
    return (model,)
