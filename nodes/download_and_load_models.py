import os
import folder_paths
import comfy.utils
from .. import constants as constants
from ..model_helpers.prepare_positionnet import prepare_positionnet, get_positionnet_default_params
from ..model_helpers.prepare_scaleu import prepare_scaleu_nets
from ..model_helpers.prepare_fusers import prepare_fusers
from huggingface_hub import snapshot_download

INSTANCE_FUSERS_DIR = "fuser_models"

INSTANCE_SCALEU_DIR = "scaleu_models"

class DownloadInstanceDiffusionModels:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "use_segs": ("BOOLEAN", {"default": True}),
            "fusers_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("POSITIONNET", "FUSERS", "SCALEU", )
    FUNCTION = "load_model"

    CATEGORY = "instance/loaders"

    def load_model(self, use_segs: bool, fusers_scale: float):
        repo_id = "logtd/instance_diffusion"
        instance_models_folder = os.path.join(folder_paths.models_dir, constants.INSTANCE_MODELS_DIR)

        models_to_download = [
            ("position_net", constants.INSTANCE_POSITIONNET_DIR, "position_net.ckpt"),
            ("fusers", constants.INSTANCE_FUSERS_DIR, "fusers.ckpt"),
            ("scaleu", constants.INSTANCE_SCALEU_DIR, "scaleu.ckpt")
        ]

        for model_name, model_folder, model_file in models_to_download:
            model_folder_path = os.path.join(instance_models_folder, model_folder)
            model_file_path = os.path.join(model_folder_path, model_file)

            if not os.path.exists(model_file_path):
                print(f"Selected model: {model_file_path} not found, downloading...")
                allow_patterns = [f"*{model_name}*"]
                snapshot_download(repo_id=repo_id, 
                                  allow_patterns=allow_patterns, 
                                  local_dir=model_folder_path, 
                                  local_dir_use_symlinks=False
                                  )
                
        positionnet_file = os.path.join(instance_models_folder, constants.INSTANCE_POSITIONNET_DIR, "position_net.ckpt")
        fusers_file = os.path.join(instance_models_folder, constants.INSTANCE_FUSERS_DIR, "fusers.ckpt")
        scaleu_file = os.path.join(instance_models_folder, constants.INSTANCE_SCALEU_DIR, "scaleu.ckpt")

        pos_checkpoint = comfy.utils.load_torch_file(positionnet_file, safe_load=True)
        params = get_positionnet_default_params()
        params["use_segs"] = use_segs
        model = prepare_positionnet(pos_checkpoint, params)
        positionnet = {
            'model': model,
        }

        fusers_checkpoint = comfy.utils.load_torch_file(fusers_file, safe_load=True)
        fusers_list = prepare_fusers(fusers_checkpoint, fusers_scale)
        fusers = {
            'model_list': fusers_list
        }
        scaleu_checkpoint = comfy.utils.load_torch_file(scaleu_file, safe_load=True)
        scaleu_list = prepare_scaleu_nets(scaleu_checkpoint)
        scaleu = {
            'model_list': scaleu_list
        }
        return (positionnet, fusers, scaleu)
    
