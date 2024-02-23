import os
import folder_paths
import torch

import comfy.utils

from . import constants as constants


def get_model_dir(model_dir) -> list[str]:
  path = os.path.join(constants.INSTANCE_MODELS_DIR,
                      model_dir)
  return folder_paths.get_filename_list(path)


def load_checkpoint(checkpoint_path):
  checkpoint = comfy.utils.load_torch_file(checkpoint_path, safe_load=True)
  return checkpoint
