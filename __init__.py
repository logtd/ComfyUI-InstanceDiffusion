from .nodes.instance_diffusion_tracking_prompt_node import InstanceDiffusionTrackingPromptNode

NODE_CLASS_MAPPINGS = {
    "LoadInstanceDiffusion": None,
    "LoadInstancePositionNetModel": None,
    "LoadInstanceScaleUModel": None,
    "LoadInstanceAttentionModel": None,
    "InstanceDiffusionTrackingPrompt": InstanceDiffusionTrackingPromptNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadInstanceDiffusion": "Load Instance Diffusion",
    "LoadInstancePositionNetModel": "Load Instance PositionNet Model",
    "LoadInstanceScaleUModel": "Load Instance ScaleU Model",
    "LoadInstanceAttentionModel": "Load Instance Attention Model",
    "InstanceDiffusionTrackingPrompt": "Instance Diffusion Tracking Prompt"
}
