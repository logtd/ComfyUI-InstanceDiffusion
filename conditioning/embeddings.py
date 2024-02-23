import torch
import numpy as np
from skimage.transform import resize
from ..utils.decode_item import binary_mask_to_polygon, sample_uniform_sparse_points


N_SCRIBBLE_POINTS = 20
N_POLYGON_POINTS = 256
N_MAX_OBJECTS = 30


def get_point_from_box(bbox):
  x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
  return [(x0 + x1) / 2.0, (y0 + y1) / 2.0]


def get_empty_binary_mask(img_width, img_height):
  return np.zeros((img_width, img_height, 1))


def sample_random_points_from_mask(mask, k=N_SCRIBBLE_POINTS):
  mask = mask[:, :, 0]
  # Find the coordinates of non-zero pixels in the binary mask
  nonzero_coords = np.transpose(np.nonzero(mask))

  # Randomly sample 'k' points
  # return all zeros if there is no non-zero pixel
  if len(nonzero_coords) == 0:
    xy_points = [0 for _ in range(k * 2)]
    return xy_points

  # randomly sample with replacement if there are not enough non-zero pixels
  if len(nonzero_coords) < k and len(nonzero_coords) > 0:
    random_indices = np.random.choice(len(nonzero_coords), k, replace=True)
  # randomly sample withiout replacement if there are enough non-zero pixels
  else:
    random_indices = np.random.choice(
        len(nonzero_coords), k, replace=False)
  sampled_points = nonzero_coords[random_indices]

  # order the points by their distance to (0, 0)
  # center = np.array([mask.shape[0] // 2, mask.shape[1] // 2])
  center = np.array([0, 0])
  sampled_points = sorted(sampled_points, key=lambda x: np.linalg.norm(
      np.array(x) - center))  # np.linalg.norm

  # concatenate x and y coordinates and return them as a list
  # [x1,y1,x2,y2,...,x_k,y_k]
  xy_points = []
  for x in sampled_points:
    xy_points.append(float(x[1]))
    xy_points.append(float(x[0]))
  return xy_points


def convert_points(points, img_width, img_height):
  # convert polygons/scribbless' coordinates to the relative values (0, 1)
  for i in range(len(points)):
    if i % 2 == 0:
      points[i] = min(points[i] / img_width, 1.0)
    else:
      points[i] = min(points[i] / img_height, 1.0)
  return points


def sample_sparse_points_from_mask(mask, k=256):
  n_points = k
  n_polygons = n_points // 2  # half points should be sampled from the polygons
  mask = mask[:, :, 0]
  # sample sparse points from the polygons (boundary)
  polygons = binary_mask_to_polygon(mask, tolerance=0.0)
  # concatenate polygons to a single list
  polygons_single = []
  for polygon in polygons:
    polygons_single += polygon
  if len(polygons_single) != 0:
    # uniformly sample points from the polygon
    polygons_single = np.array(polygons_single).reshape(-1, 2)
    indexes = np.linspace(0, polygons_single.shape[0] - 1, n_polygons)
    indexes = list([int(i) for i in indexes])

    polygons_single = polygons_single[indexes]
    sampled_polygons = [(x[0], x[1]) for x in polygons_single]
  else:
    return [0 for _ in range(256 * 2)]

  # sample sparse points from the mask
  n_inside_points = n_points - len(sampled_polygons)
  inside_points = sample_uniform_sparse_points(mask, n_inside_points)

  # combine inside_points and sampled_polygons
  xy_points = inside_points + sampled_polygons

  # order the points by their distance to (0, 0)
  center = np.array([0, 0])
  xy_points = sorted(xy_points, key=lambda x: np.linalg.norm(
      np.array(x) - center))  # np.linalg.norm

  # return the sampled points
  sampled_points = []
  for x in xy_points:
    sampled_points.append(x[0])
    sampled_points.append(x[1])
  return sampled_points


# [x0, y0, x1, y1]
def get_grounding_input_from_coords(coords, img_width, img_height):
  x0, y0, x1, y1, coord_width, coord_height = coords
  location = [x0 / coord_width, y0 / coord_height,
              x1 / coord_width, y1 / coord_height]

  point = get_point_from_box(location)
  binary_mask = get_empty_binary_mask(img_width, img_height)

  scribble = sample_random_points_from_mask(binary_mask, k=N_SCRIBBLE_POINTS)
  scribble = convert_points(scribble, img_width, img_height)

  polygon = sample_sparse_points_from_mask(binary_mask, k=N_POLYGON_POINTS)
  polygon = convert_points(polygon, img_width, img_height)

  segment = resize(binary_mask.astype(np.float32),
                   (img_width, img_height)).squeeze()
  # segment = np.stack(segment).astype(np.float32).squeeze() if len(segment) > 0 else segment

  return dict(
      polygon=polygon,
      scribble=scribble,
      segment=segment,
      box=location,
      point=point,
  )


def create_zero_input_tensors(n_frames, img_width, img_height):
  masks = torch.zeros(n_frames, N_MAX_OBJECTS)
  text_masks = torch.zeros(n_frames, N_MAX_OBJECTS)
  text_embeddings = torch.zeros(n_frames, N_MAX_OBJECTS, 768)
  box_embeddings = torch.zeros(n_frames, N_MAX_OBJECTS, 4)
  polygon_embeddings = torch.zeros(
      n_frames, N_MAX_OBJECTS, N_POLYGON_POINTS * 2)
  scribble_embeddings = torch.zeros(
      n_frames, N_MAX_OBJECTS, N_SCRIBBLE_POINTS * 2)
  segment_embeddings = torch.zeros(
      n_frames, N_MAX_OBJECTS, img_width, img_height)  # TODO: width height order
  point_embeddings = torch.zeros(n_frames, N_MAX_OBJECTS, 2)

  return dict(
      masks=masks,
      text_masks=text_masks,
      prompts=text_embeddings,
      boxes=box_embeddings,
      polygons=polygon_embeddings,
      scribbles=scribble_embeddings,
      segments=segment_embeddings,
      points=point_embeddings
  )


def get_attn_mask(img_size=64):
  return torch.zeros(N_MAX_OBJECTS, img_size, img_size)


def prepare_embeddings(conds, latent_shape, idxs, use_masked_att=False):
  batch_size, _, latent_height, latent_width = latent_shape
  embeddings = create_zero_input_tensors(
    batch_size, latent_width, latent_height)
  if use_masked_att:
    embeddings['att_masks'] = torch.zeros(
        idx, N_MAX_OBJECTS, latent_width, latent_height)

  for idx in idxs:
    for cond_idx, cond in enumerate(conds):
      if cond['positions'][idx] is None:
        continue

      grounding = get_grounding_input_from_coords(
          cond['positions'][idx], latent_width, latent_height)
      embeddings['masks'][idx][cond_idx] = 1
      embeddings['text_masks'][idx][cond_idx] = 1
      embeddings['prompts'][idx][cond_idx] = cond['cond_pooled']
      embeddings['boxes'][idx][cond_idx] = torch.tensor(
          grounding['box'])
      embeddings['polygons'][idx][cond_idx] = torch.tensor(
          grounding['polygon'])
      embeddings['scribbles'][idx][cond_idx] = torch.tensor(
          grounding['scribble'])
      embeddings['segments'][idx][cond_idx] = torch.tensor(
          grounding['segment'])
      embeddings['points'][idx][cond_idx] = torch.tensor(
          grounding['point'])

      if use_masked_att:
        box = grounding['box']
        x1, y1, x2, y2 = int(np.round(box[0] * latent_width)), int(np.round(box[1] * latent_height)), int(
            np.round(box[2] * latent_width)), int(np.round(box[3] * latent_height))
        embeddings['att_masks'][idx][cond_idx][x1:x2, y1:y2] = 1

  return embeddings


def get_model_inputs():
  return
