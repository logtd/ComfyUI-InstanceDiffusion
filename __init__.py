from .nodes.apply_scaleu_model_node import ApplyScaleUModelNode
from .nodes.load_instance_scaleu_node import LoadInstanceScaleUNode
from .nodes.load_instance_fusers_node import LoadInstanceFusersNode
from .nodes.load_instance_positionnet_node import LoadInstancePositionNetNode
from .nodes.instance_diffusion_tracking_prompt_node import InstanceDiffusionTrackingPromptNode
from .nodes.download_and_load_models import DownloadInstanceDiffusionModels


NODE_CLASS_MAPPINGS = {
    "ApplyScaleUModelNode": ApplyScaleUModelNode,
    "LoadInstanceScaleUNode": LoadInstanceScaleUNode,
    "LoadInstancePositionNetModel": LoadInstancePositionNetNode,
    "LoadInstanceFusersNode": LoadInstanceFusersNode,
    "InstanceDiffusionTrackingPrompt": InstanceDiffusionTrackingPromptNode,
    "DownloadInstanceDiffusionModels": DownloadInstanceDiffusionModels
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyScaleUModelNode": "Apply Instance Diffusion ScaleU",
    "LoadInstancePositionNetModel": "Load Instance PositionNet Model",
    "LoadInstanceScaleUModel": "Load Instance ScaleU Model",
    "LoadInstanceFusersNode": "Load Instance Fusers Model",
    "InstanceDiffusionTrackingPrompt": "Instance Diffusion Tracking Prompt",
    "DownloadInstanceDiffusionModels": "(Down)Load Instance Diffusion Models"
}
