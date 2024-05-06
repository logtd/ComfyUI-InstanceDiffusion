from .. import constants as constants
from ..utils.model_utils import get_model_list, load_checkpoint
from ..model_helpers.prepare_fusers import prepare_fusers


class LoadInstanceFusersNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_filename": (get_model_list(constants.INSTANCE_FUSERS_DIR),),
            "fusers_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("FUSERS",)
    FUNCTION = "load_model"

    CATEGORY = "instance/loaders"

    def load_model(self, model_filename: str, fusers_scale: float):
        checkpoint = load_checkpoint(
            constants.INSTANCE_FUSERS_DIR, model_filename)
        fusers_list = prepare_fusers(checkpoint, fusers_scale)
        fusers = {
            'model_list': fusers_list
        }
        return (fusers,)
