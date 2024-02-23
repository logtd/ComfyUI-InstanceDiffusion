from .. import constants as constants
from ..utils.model_utils import get_model_dir, load_checkpoint
from ..models.prepare_fusers import prepare_fusers


class LoadInstanceFusersNode:
  @classmethod
  def INPUT_TYPES(s):
    return {"required": {
        "model_path": (get_model_dir(constants.INSTANCE_FUSERS_DIR)),
    }}

  RETURN_TYPES = ("FUSERS",)
  FUNCTION = "load_model"

  CATEGORY = "instance/loaders"

  def load_model(self, model_path: str):
    checkpoint = load_checkpoint(model_path)
    fusers_list = prepare_fusers(checkpoint)
    fusers = {
      'model_list': fusers_list
    }
    return (fusers,)
