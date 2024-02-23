from .. import constants as constants
from ..utils.model_utils import get_model_dir, load_checkpoint
from ..models.prepare_scaleu import prepare_scaleu_nets


class LoadInstanceScaleUNode:
  @classmethod
  def INPUT_TYPES(s):
    return {"required": {
        "model_path": (get_model_dir(constants.INSTANCE_SCALEU_DIR)),
    }}

  RETURN_TYPES = ("SCALEU",)
  FUNCTION = "load_model"

  CATEGORY = "instance/loaders"

  def load_model(self, model_path: str):
    checkpoint = load_checkpoint(model_path)
    scaleu_list = prepare_scaleu_nets(checkpoint)
    scaleu = {
      'model_list': scaleu_list
    }
    return (scaleu,)
