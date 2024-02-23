import torch


class GroundingNetInput:
    def __init__(self):
        self.set = False
        self.return_att_masks = False
        self.image_size = 64
        self.return_att_masks32 = False

    def prepare(self, batch, image_size=64, device=None, dtype=None, return_att_masks=False):
        """
        batch should be the output from dataset.
        Please define here how to process the batch and prepare the 
        input only for the ground tokenizer. 
        """
        output = {}
        self.set = True
        self.return_att_masks = return_att_masks

        boxes = batch['boxes']
        masks = batch['masks']
        positive_embeddings = batch["prompts"]

        if self.return_att_masks:
            assert 'att_masks' in batch
            att_masks = batch['att_masks']

        scribbles = batch['scribbles']
        polygons = batch['polygons']
        self.dim_scribbles = scribbles.shape[-1]
        self.dim_polygons = polygons.shape[-1]
        # NOTE: New Seg
        segs = batch["segments"]
        self.dim_segs = segs.shape[-1]
        points = batch["points"]

        self.batch, self.max_box, self.in_dim = positive_embeddings.shape
        self.device = positive_embeddings.device
        self.dtype = positive_embeddings.dtype

        output = {
            "boxes": boxes,
            "masks": masks,
            "prompts": positive_embeddings,
        }
        output["scribbles"] = scribbles
        output["polygons"] = polygons
        output["segments"] = segs
        output["points"] = points

        if self.return_att_masks:
            output['att_masks'] = att_masks
        return output

    def get_null_input(self, batch=None, latent_width=64, latent_height=64, device=None, dtype=None):
        """
        Guidance for training (drop) or inference, 
        please define the null input for the grounding tokenizer 
        """

        assert self.set, "not set yet, cannot call this funcion"
        batch = self.batch if batch is None else batch
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype

        boxes = torch.zeros(batch, self.max_box, 4,).type(dtype).to(device)
        masks = torch.zeros(batch, self.max_box).type(dtype).to(device)
        # NOTE: New Seg
        segs = torch.zeros(batch, self.max_box, self.dim_segs,
                           self.dim_segs).type(dtype).to(device)

        scribbles = torch.zeros(batch, self.max_box,
                                self.dim_scribbles).type(dtype).to(device)
        polygons = torch.zeros(batch, self.max_box,
                               self.dim_polygons).type(dtype).to(device)
        points = torch.zeros(batch, self.max_box, 2).type(dtype).to(device)

        positive_embeddings = torch.zeros(
            batch, self.max_box, self.in_dim).type(dtype).to(device)

        output = {
            "boxes": boxes,
            "masks": masks,
            "prompts": positive_embeddings,
        }
        output["scribbles"] = scribbles
        output["polygons"] = polygons
        output["segments"] = segs
        output["points"] = points

        if self.return_att_masks:
            att_masks = torch.zeros(batch, self.max_box, latent_width, latent_height).type(
                dtype).to(device)  # TODO Order width/height
            output['att_masks'] = att_masks
        return output
