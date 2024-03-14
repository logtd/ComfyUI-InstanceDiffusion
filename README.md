# ComfyUI-InstanceDiffusion
ComfyUI nodes to use InstanceDiffusion.

Original research repo: https://github.com/frank-xwang/InstanceDiffusion

## Table of Contents
- [Installation](#installation)
  - [How to Install](#how-to-install)
  - [How to Configure Models](#how-to-configure-models)
- [Accompanying Node Repos](#accompanying-node-repos)
- [Examples](#examples)
- [Acknowledgements](#acknowledgements)

## Installation

### How to Install
Clone or download this repo into your `ComfyUI/custom_nodes/` directory.
There are no Python package requirements outside of the standard ComfyUI requirements at this time.

### How to Configure Models
These models were trained by [frank-xwang](https://github.com/frank-xwang) baked inside of StableDiffusion 1.5. These are spliced out into individual models to be used with other SD1.5 checkpoints.
Download each of these checkpoints and place them into the Installation Directory within `ComfyUI/models/instance_models/` directory.

| Model Name | URL | Installation Directory |
|------------|-----|------------------------|
| fusers.ckpt     | [huggingface](https://huggingface.co/spaces/logtd/instancediffusion/blob/main/fusers.ckpt) | `instance_models/fuser_models/`      |
| positionnet.ckpt     | [huggingface](https://huggingface.co/spaces/logtd/instancediffusion/blob/main/position_net.ckpt) | `instance_models/positionnet_models/`      |
| scaleu.ckpt     | [huggingface](https://huggingface.co/spaces/logtd/instancediffusion/blob/main/scaleu.ckpt) | `instance_models/scaleu_models/`      |


## Accompanying Node Repos
* [Tracking Nodes for videos](https://github.com/logtd/ComfyUI-TrackingNodes)
* [AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
* [Video Helper Suite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

## Examples
Example workflows can be found in the `example_workflows/` directory.

https://github.com/logtd/ComfyUI-InstanceDiffusion/assets/160989552/ee42891a-cc38-421c-98bf-03a1be11d315

https://github.com/logtd/ComfyUI-InstanceDiffusion/assets/160989552/40038526-5850-4cb6-9658-c38c7e4b20f9

https://github.com/logtd/ComfyUI-InstanceDiffusion/assets/160989552/eae3520c-9a3d-4cde-b32f-1af9231ad2d4

https://github.com/logtd/ComfyUI-InstanceDiffusion/assets/160989552/85b7d9df-7f7e-43c7-b2fa-b14fd5ec5e6d

## Acknowledgements
* [frank-xwang](https://github.com/frank-xwang) for creating the original repo, training models, etc.
* [Kosinkadink](https://github.com/Kosinkadink) for creating AnimateDiff-Evolved and providing support on integration
