# Directly taken from InstanceDiffusion repo
import torch
import random
import base64
import numpy as np
from io import BytesIO
from collections import Counter
from PIL import Image, ImageDraw
import base64
from skimage import measure


# import nltk
# from nltk.corpus import stopwords

def decode_base64_to_pillow(image_b64):
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert('RGB')


def decode_tensor_from_string(arr_str, use_tensor=True):
    arr = np.frombuffer(base64.b64decode(arr_str), dtype='float32')
    if use_tensor:
        arr = torch.from_numpy(arr)
    return arr


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

# convert binay mask to polygon format


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, tolerance)
    polygons = []
    # print(contours)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def sample_random_points_from_mask(mask, k):
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

# convert numpy array of bool mask to float mask


def binary_mask_to_int(binary_mask):
    return binary_mask.astype(np.int32)

# uniformly sample points from the mask


def sample_sparse_points(binary_mask, k, return_2d=False):
    # Find the coordinates of non-zero pixels in the binary mask
    nonzero_coords = np.array(np.nonzero(binary_mask))
    if len(nonzero_coords) == 0:
        xy_points = [0 for _ in range(k * 2)]
        return xy_points

    # Calculate the total number of non-zero pixels
    num_nonzero_pixels = len(nonzero_coords)

    xy_points = []
    if k >= num_nonzero_pixels:
        for x in nonzero_coords:
            xy_points.append(float(x[1]))
            xy_points.append(float(x[0]))
        for _ in range(k - num_nonzero_pixels):
            xy_points.append(nonzero_coords[-1][1])
            xy_points.append(nonzero_coords[-1][0])
        return nonzero_coords

    # Calculate the number of points to sample in each dimension
    num_points_per_dim = int(np.sqrt(k))

    # Calculate the step size to ensure equal spacing
    step_size = max(1, num_nonzero_pixels // (num_points_per_dim ** 2))

    # Sample points with equal spacing
    sampled_points = nonzero_coords[::step_size][:k]
    if return_2d:
        sampled_points = [(x[1], x[0]) for x in sampled_points]
    else:
        for x in sampled_points:
            xy_points.append(float(x[1]))
            xy_points.append(float(x[0]))
        return xy_points


def sample_uniform_sparse_points(binary_mask, k):
    # binary_mask = binary_mask[:,:,0]
    # Step 1: Get the indices of '1' values in the binary mask
    foreground_indices = np.argwhere(binary_mask == 1)

    if len(foreground_indices) == 0:
        return []

    selected_points = []
    if len(foreground_indices) < k:
        # randomly sample with replacement if there are not enough non-zero pixels
        for i in range(k):
            random_point = random.choice(foreground_indices)
            selected_points.append((random_point[1], random_point[0]))
    else:
        # rank the points by their distance to the mean of the foreground_indices
        center = np.mean(foreground_indices, axis=0)
        # print(center)
        foreground_indices = sorted(
            foreground_indices, key=lambda x: np.linalg.norm(x - center))  # np.linalg.norm
        # Calculate the number of points to select from each segment
        points_per_segment = len(foreground_indices) // k

        # Step 2: Randomly select one point from each segment
        # print(k)
        for i in range(k):
            segment_points = foreground_indices[i *
                                                points_per_segment: (i + 1) * points_per_segment]
            # choose the middle point in each segment
            random_point = segment_points[len(segment_points) // 2]
            # random_point = random.choice(segment_points)
            selected_points.append((random_point[1], random_point[0]))

    return selected_points


def sample_sparse_points_from_mask(mask, k):
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
        return None

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


def get_polygons_from_mask(mask, tolerance=0, n_polygon_points=256):
    mask = binary_mask_to_int(mask)
    return_polygons = True
    if return_polygons:
        # convert float mask to polygons
        polygons = binary_mask_to_polygon(mask[:, :, 0], tolerance=tolerance)

        # return all zeros if there is no polygon
        if len(polygons) == 0:
            polygons = [0 for _ in range(n_polygon_points * 2)]
            return polygons

        # concatenate polygons to a single list
        polygon = []
        for p in polygons:
            polygon += p

        # uniformly sample points the polygon
        polygon = np.array(polygon).reshape(-1, 2)
        indexes = np.linspace(0, polygon.shape[0] - 1, n_polygon_points)
        indexes = [int(i) for i in indexes]
        polygon = polygon[indexes].reshape(-1)

        return polygon
    else:
        sampled_points = sample_sparse_points(mask, n_polygon_points)
        return sampled_points


def decode_item(item):
    # convert string to dict
    if "image" in item and isinstance(item['image'], Image.Image):
        return item

    item['image'] = decode_base64_to_pillow(item['image'])
    segs = []
    for anno in item['annos']:
        anno['image_embedding_before'] = decode_tensor_from_string(
            anno['image_embedding_before'])
        anno['text_embedding_before'] = decode_tensor_from_string(
            anno['text_embedding_before'])
        anno['image_embedding_after'] = decode_tensor_from_string(
            anno['image_embedding_after'])
        anno['text_embedding_after'] = decode_tensor_from_string(
            anno['text_embedding_after'])
        if "blip_clip_embeddings" in anno:
            anno['blip_clip_embeddings'] = decode_tensor_from_string(
                anno['blip_clip_embeddings'])
        if 'mask' in anno:
            # sample k random points from the mask
            n_scribble_points = 20
            rle = anno['mask']
            binary_mask = decodeToBinaryMask(rle)
            segs.append(binary_mask)
            if "scribbles" in anno:
                anno['scribbles'] = anno["scribbles"]
            else:
                anno['scribbles'] = sample_random_points_from_mask(
                    binary_mask, n_scribble_points)
            # convert mask to polygon
            n_polygon_points = 256
            polygons = sample_sparse_points_from_mask(
                binary_mask, k=n_polygon_points)
            if polygons != None:
                anno['polygons'] = polygons
            else:
                anno['polygons'] = [0 for _ in range(n_polygon_points * 2)]
    if len(segs) > 0:
        item['segs'] = np.stack(segs).astype(np.float32).squeeze()
    return item


def check_unique(images, fields):
    for field in fields:
        temp_list = []
        for img_info in images:
            temp_list.append(img_info[field])
        assert len(set(temp_list)) == len(temp_list), field


def clean_data(data):
    for data_info in data:
        data_info.pop("original_img_id", None)
        data_info.pop("original_id", None)
        # sentence id for each image (multiple sentences for one image)
        data_info.pop("sentence_id", None)
        data_info.pop("dataset_name", None)
        data_info.pop("data_source", None)
        data_info["data_id"] = data_info.pop("id")


def clean_annotations(annotations):
    for anno_info in annotations:
        anno_info.pop("iscrowd", None)
        anno_info.pop("category_id", None)
        anno_info.pop("area", None)
        anno_info["data_id"] = anno_info.pop("image_id")


def draw_box(img, boxes):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle([box[0], box[1], box[2], box[3]],
                       outline="red", width=2)  # x0 y0 x1 y1
    return img


def xyhw2xyxy(box):
    x0, y0, w, h = box
    return [x0, y0, x0 + w, y0 + h]


def make_a_sentence_count_nums(obj_names):
    # count the number of duplicated strings in the list
    # ["dog", "dog", "cat"]
    obj_names = dict(Counter(obj_names))
    # {'dog': 2, 'cat': 1}
    caption = ""
    for item in obj_names:
        caption += str(obj_names[item]) + " " + item + ", "
    return caption[:-2]


def make_a_sentence(obj_names, clean=False):

    if clean:
        obj_names = [name[:-6] if ("-other" in name)
                     else name for name in obj_names]

    caption = ""
    tokens_positive = []
    for obj_name in obj_names:
        start_len = len(caption)
        caption += obj_name
        end_len = len(caption)
        caption += ", "
        tokens_positive.append(
            # in real caption, positive tokens can be disjoint, thus using list of list
            [[start_len, end_len]]
        )
    caption = caption[:-2]  # remove last ", "

    return caption  # , tokens_positive


def mask_for_random_drop_text_or_image_feature(masks, random_drop_embedding):
    """
    input masks tell how many valid grounding tokens for this image
    e.g., 1,1,1,1,0,0,0,0,0,0...

    If random_drop_embedding=both.  we will random drop either image or
    text feature for each token, 
    but we always make sure there is at least one feature used. 
    In other words, the following masks are not valid 
    (because for the second obj, no feature at all):
    image: 1,0,1,1,0,0,0,0,0
    text:  1,0,0,0,0,0,0,0,0

    if random_drop_embedding=image. we will random drop image feature 
    and always keep the text one.  

    """
    N = masks.shape[0]

    if random_drop_embedding == 'both':
        temp_mask = torch.ones(2, N)
        for i in range(N):
            if random.uniform(0, 1) < 0.5:  # else keep both features
                # randomly choose to drop image or text feature
                idx = random.sample([0, 1], 1)[0]
                temp_mask[idx, i] = 0
        image_masks = temp_mask[0] * masks
        text_masks = temp_mask[1] * masks

    if random_drop_embedding == 'image':
        image_masks = masks * (torch.rand(N) > 0.5) * 1
        text_masks = masks

    return image_masks, text_masks


def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x @ torch.transpose(projection_matrix, 0, 1)


def inv_project(y, projection_matrix):
    """
    y (Batch*768) should be the CLIP feature (after projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim).  
    this function will return the CLIP penultimate feature. 

    Note: to make sure getting the correct penultimate feature, the input y should not be normalized. 
    If it is normalized, then the result will be scaled by CLIP feature norm, which is unknown.   
    """
    return y @ torch.transpose(torch.linalg.inv(projection_matrix), 0, 1)
