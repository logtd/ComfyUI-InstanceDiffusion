from .. import constants as constants
from ..utils.model_utils import get_model_list, load_checkpoint
from ..model_helpers.prepare_scaleu import prepare_scaleu_nets


class LoadInstanceScaleUNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_filename": (get_model_list(constants.INSTANCE_SCALEU_DIR),),
        }}

    RETURN_TYPES = ("SCALEU",)
    FUNCTION = "load_model"

    CATEGORY = "instance/loaders"

    def load_model(self, model_filename: str):
        checkpoint = load_checkpoint(
            constants.INSTANCE_SCALEU_DIR, model_filename)
        scaleu_list = prepare_scaleu_nets(checkpoint)
        scaleu = {
            'model_list': scaleu_list
        }
        return (scaleu,)
