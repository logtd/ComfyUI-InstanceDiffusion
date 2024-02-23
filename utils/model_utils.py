import os
import folder_paths

import comfy.utils

from .. import constants


def get_model_dir(model_dir):
  root_path = folder_paths.folder_names_and_paths['custom_nodes'][0][0]
  path = os.path.join(root_path, 'ComfyUI-InstanceDiffusion', constants.INSTANCE_MODELS_DIR,
                      model_dir)
  return path


def get_model_list(model_dir) -> list[str]:
  path = get_model_dir(model_dir)
  return os.listdir(path)


def load_checkpoint(model_dir, filename):
  checkpoint_path = os.path.join(get_model_dir(model_dir), filename)
  checkpoint = comfy.utils.load_torch_file(checkpoint_path, safe_load=True)
  return checkpoint
