from .. import constants as constants
from ..utils.model_utils import get_model_list, load_checkpoint
from ..model_helpers.prepare_positionnet import prepare_positionnet, get_positionnet_default_params


class LoadInstancePositionNetNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_filename": (get_model_list(constants.INSTANCE_POSITIONNET_DIR),),
        }}

    RETURN_TYPES = ("POSITIONNET",)
    FUNCTION = "load_model"

    CATEGORY = "instance/loaders"

    def load_model(self, model_filename: str):
        checkpoint = load_checkpoint(
            constants.INSTANCE_POSITIONNET_DIR, model_filename)
        params = get_positionnet_default_params()
        model = prepare_positionnet(checkpoint, params)
        positionnet = {
            'model': model,
        }
        return (positionnet,)
