from .. import constants as constants
from ..utils.model_utils import get_model_dir, load_checkpoint
from ..models.prepare_positionnet import prepare_positionnet, get_positionnet_default_params


class LoadInstancePositionNetNode:
  @classmethod
  def INPUT_TYPES(s):
    return {"required": {
        "model_path": (get_model_dir(constants.INSTANCE_POSITIONNET_DIR)),
    }}

  RETURN_TYPES = ("POSITIONNET",)
  FUNCTION = "load_model"

  CATEGORY = "instance/loaders"

  def load_model(self, model_path: str):
    checkpoint = load_checkpoint(model_path)
    params = get_positionnet_default_params()
    model = prepare_positionnet(checkpoint, params)
    positionnet = {
      'model': model,
    }
    return (positionnet,)
