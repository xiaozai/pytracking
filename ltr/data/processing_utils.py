import torch
import math
import cv2 as cv
import random
import torch.nn.functional as F
from .bounding_box_utils import rect_to_rel, rel_to_rect
import scipy.stats

import math
import numpy as np

def numpy_to_torch(a: np.ndarray):
    '''
    Song copy from pytracking.features.preprocessing
    '''
    return torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0)


def torch_to_numpy(a: torch.Tensor):
    return a.squeeze(0).permute(1,2,0).numpy()

def hard_masking(rgb, prob_map, prob_threshold=0.2):
    '''
    Song : performing the Hard Masking on RGB images with depth-based probability map

    Input :
        - rgb, [h, w, 3]
        - prob_map, [h, w], float
        - prob_threshold, float, in (0, 1)

    Return :
        - rgb after masking
    '''
    depth_mask = prob_map > prob_threshold
    depth_mask = depth_mask.astype(int)

    rgb[depth_mask==0] = 0

    return rgb

def gaussian_prob(x, mu, std):
    '''
    Song : the depth value closer to the mu (center depth value) obtains the largest weights
    '''
    var = float(std)**2
    denom = (2*math.pi*var)**.5
    num = np.exp(-(x-float(mu))**2/(2*var))
    return num

def depth_prob_map_from_histogram(depth_img, box, num_bins=8):
    '''
    Song : to select the candidate depth and the corresponding depth-based probability map

    inputs :
        - depth_img, a single depth image , usually is the search region
        - box, (x,y,w,h), GT box,int
        - num_bins , int, the number of the bins in the histogram

    return :
        - candidate_depth, the estimated depth value for the target
        - prob_map, the depth-based probability map (Gaussian-like)
    '''

    # 1) crop the depth area
    box_depths = depth_img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    center_depth = depth_img[int((box[1]+box[3])/2.0), int((box[0]+box[2])/2.0)]

    dp_values = np.array(box_depths, dtype=np.float32).flatten()
    dp_values.sort()

    max_depth_img = np.max(depth_img.flatten())
    if len(dp_values) > 0:
        max_depth_box = dp_values[-1]
    else:
        '''
        Sometimes errors occur here, needs to be figured out why
        '''
        prob_map = np.ones_like(depth_img, dtype=np.float32)
        candidate_depth = 0
        return prob_map, candidate_depth

    # 2) perform the historgram to count the frequency of the depth values
    hist, bin_edges = np.histogram(dp_values, bins=num_bins)
    hist = list(hist)

    # 3) select the candidate depth
    candidate_idx = hist.index(np.max(hist))
    bin_lower = bin_edges[candidate_idx]
    bin_upper = bin_edges[candidate_idx+1]

    if bin_upper == max_depth_img:
        '''
        We assume that the largest depth belongs to the background

        if not the largest depth in the search region ,
            we choose the top 1 as the candidate depth for target
            and use the mean value of bin
        else
            choose the 2nd as the candidate
        '''
        sorted_hist = hist
        sorted_hist.sort()

        candiate_idx = hist.index(sorted_hist[-2]) # 2nd largest
        bin_lower = bin_edges[candidate_idx]
        bin_upper = bin_edges[candidate_idx+1]

    depth_in_bin = dp_values[dp_values>=bin_lower]
    depth_in_bin = depth_in_bin[depth_in_bin<=bin_upper]
    candidate_depth = np.mean(depth_in_bin) + 0.1 # to avoid the zeros

    # if bin_upper == max_depth_box:
    #     '''
    #     Sometimes the largest depth in the box is also the background
    #     check if the candidate depth is same as the center one
    #
    #     usually we assume that the target should locates in the center of the box
    #
    #     hard-programming ????
    #     '''
    #     if abs(bin_upper - center_depth) > 1000:
    #         candidate_depth = center_depth

    # 4) obtain the depth-based Gaussian probability map
    sigma = (bin_upper - bin_lower + 1) / 2.0
    prob_map = gaussian_prob(depth_img, candidate_depth, sigma)

    return prob_map, candidate_depth

def proposals_depth_prob_map_from_histogram(depth_img, boxes, num_bins=8):
    '''
    Song : multiple proposal boxes
           use the element-wise maximum operation on the prob_maps for each box ???
           this one seems not that good

           we create the prob map for each proposaled box, and then use the maximum opertaion
    '''

    prob_map = np.zeros_like(depth_img, dtype=np.float32)

    for box in boxes:
        temp_prob_map, _ = depth_prob_map_from_histogram(depth_img, box, num_bins=num_bins)
        prob_map = np.maximum(prob_map, temp_prob_map)

    return prob_map

def plot_prob_map(s, data, depth_mask, prob_map):
    ''' Example Plot
        s : train or test
        data : 'train_images' and 'test_images'
     '''
    print(s)
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
    rgb = data[s + '_images'][0].numpy() # (3, 288, 288)
    rgb = np.transpose(rgb, (1, 2, 0))    # (288, 288, 3) normalized rgb image
    # new_image_blue, new_image_green, new_image_red = rgb
    # rgb = np.dstack([new_image_red, new_image_green, new_image_blue])
    ax1.imshow(rgb)
    rect = data[s + '_anno'][0].numpy()
    rect1 = patches.Rectangle((rect[0],rect[1]),rect[2],rect[3],linewidth=1,edgecolor='r',facecolor='none')
    ax1.add_patch(rect1)
    dp_colormap = data[s + '_depths'][0]
    dp_colormap = cv.normalize(dp_colormap, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    dp_colormap = cv.applyColorMap(dp_colormap, cv.COLORMAP_JET)

    ax2.imshow(dp_colormap)
    rect2 = patches.Rectangle((rect[0],rect[1]),rect[2],rect[3],linewidth=1,edgecolor='r',facecolor='none')
    ax2.add_patch(rect2)


    rgb_copy = data[s + '_images_copy'][0]
    ax3.imshow(rgb_copy)
    rect3 = patches.Rectangle((rect[0],rect[1]),rect[2],rect[3],linewidth=1,edgecolor='r',facecolor='none')
    ax3.add_patch(rect3)

    ax4.imshow(prob_map)
    rect4 = patches.Rectangle((rect[0],rect[1]),rect[2],rect[3],linewidth=1,edgecolor='r',facecolor='none')
    ax4.add_patch(rect4)


    masked_rgb_copy = rgb_copy
    masked_rgb_copy[depth_mask==0] = 0
    ax5.imshow(masked_rgb_copy)
    rect5 = patches.Rectangle((rect[0],rect[1]),rect[2],rect[3],linewidth=1,edgecolor='r',facecolor='none')
    ax5.add_patch(rect5)

    masked_rgb = rgb
    masked_rgb[depth_mask==0] = 0
    ax6.imshow(masked_rgb)
    rect6 = patches.Rectangle((rect[0],rect[1]),rect[2],rect[3],linewidth=1,edgecolor='r',facecolor='none')
    ax6.add_patch(rect6)

    plt.show()
    # plt.pause(5)
    plt.close()

def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    x, y, w, h = target_bb.tolist()

    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_REPLICATE)
    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))

        if mask is None:
            return im_crop_padded, resize_factor
        mask_crop_padded = \
        F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
        return im_crop_padded, resize_factor, mask_crop_padded

    else:
        if mask is None:
            return im_crop_padded, 1.0
        return im_crop_padded, 1.0, mask_crop_padded

def sample_target_depth_mask(im, dp, target_bb, search_area_factor, output_sz=None, mask=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        dp - cv image, H*W
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop , masking with depths
        float - the factor by which the crop has been resized to make the crop size equal output_size


    Song : what is the difference between depth and mask ????
    """
    x, y, w, h = target_bb.tolist()

    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    dp = torch.from_numpy(np.array(dp, dtype=np.float32))
    # dp = np.array(dp, dtype=np.float32) # from uint16 to ??
    dp_crop = dp[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_REPLICATE)

    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    dp_crop_padded = F.pad(dp_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)
    # dp_crop_padded = cv.copyMakeBorder(dp_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_REPLICATE)

    '''
        convert dp to depth-based Gaussian Probability map
        then multiply with rgb crops

        Method #1, choose the center pixel as the default the depth value => easily to get the background
        Method #2, choose the most frequent depth values => histogram ??

        for histogam , bins = 3 ? one for BG, one for target, one for possible distractor

        further , we should use the attention here for find the possible target depth values
    '''

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        dp_crop_padded = F.interpolate(dp_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]

        if mask is None:
            return im_crop_padded, dp_crop_padded.numpy(), resize_factor
        mask_crop_padded = \
        F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]

        return im_crop_padded, dp_crop_padded.numpy(), resize_factor, mask_crop_padded

    else:
        if mask is None:
            return im_crop_padded, dp_crop_padded.numpy(), 1.0
        return im_crop_padded, dp_crop_padded.numpy(), 1.0, mask_crop_padded


def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    return box_out


def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

    if masks is None:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz)
                                for f, a in zip(frames, box_extract)]
        frames_crop, resize_factors = zip(*crops_resize_factors)
        masks_crop = None
    else:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz, m)
                                for f, a, m in zip(frames, box_extract, masks)]
        frames_crop, resize_factors, masks_crop = zip(*crops_resize_factors)

    crop_sz = torch.Tensor([output_sz, output_sz])

    # find the bb location in the crop
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]

    return frames_crop, box_crop, masks_crop

def jittered_center_crop_depth_mask(frames, depths, box_extract, box_gt, search_area_factor, output_sz, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

    if masks is None:
        crops_resize_factors = [sample_target_depth_mask(f, d, a, search_area_factor, output_sz)
                                for f, d, a in zip(frames, depths, box_extract)]
        frames_crop, depths_prob, resize_factors = zip(*crops_resize_factors)
        masks_crop = None
    else:
        crops_resize_factors = [sample_target_depth_mask(f, d, a, search_area_factor, output_sz, m)
                                for f, d, a, m in zip(frames, depths, box_extract, masks)]
        frames_crop, depths_prob, resize_factors, masks_crop = zip(*crops_resize_factors)

    crop_sz = torch.Tensor([output_sz, output_sz])

    # find the bb location in the crop
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]

    return frames_crop, depths_prob, box_crop, masks_crop


def sample_target_adaptive(im, target_bb, search_area_factor, output_sz, mode: str = 'replicate',
                           max_scale_change=None, mask=None):
    """ Extracts a crop centered at target_bb box, of area search_area_factor^2. If the crop area contains regions
    outside the image, it is shifted so that the it is inside the image. Further, if the crop area exceeds the image
    size, a smaller crop which fits the image is returned instead.

    args:
        im - Input numpy image to crop.
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.
        mode - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
               If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
               If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis of the image.
        max_scale_change - Maximum allowed scale change when performing the crop (only applicable for 'inside' and 'inside_major')
        mask - Optional mask to apply the same crop.

    returns:
        numpy image - Extracted crop.
        torch.Tensor - A bounding box denoting the cropped region in the image.
        numpy mask - Cropped mask returned only if mask is not None.
    """

    if max_scale_change is None:
        max_scale_change = float('inf')
    if isinstance(output_sz, (float, int)):
        output_sz = (output_sz, output_sz)
    output_sz = torch.Tensor(output_sz)

    im_h = im.shape[0]
    im_w = im.shape[1]

    bbx, bby, bbw, bbh = target_bb.tolist()

    # Crop image
    crop_sz_x, crop_sz_y = (output_sz * (
                target_bb[2:].prod() / output_sz.prod()).sqrt() * search_area_factor).ceil().long().tolist()

    # Get new sample size if forced inside the image
    if mode == 'inside' or mode == 'inside_major':
        # Calculate rescaling factor if outside the image
        rescale_factor = [crop_sz_x / im_w, crop_sz_y / im_h]
        if mode == 'inside':
            rescale_factor = max(rescale_factor)
        elif mode == 'inside_major':
            rescale_factor = min(rescale_factor)
        rescale_factor = min(max(1, rescale_factor), max_scale_change)

        crop_sz_x = math.floor(crop_sz_x / rescale_factor)
        crop_sz_y = math.floor(crop_sz_y / rescale_factor)

    if crop_sz_x < 1 or crop_sz_y < 1:
        raise Exception('Too small bounding box.')

    x1 = round(bbx + 0.5 * bbw - crop_sz_x * 0.5)
    x2 = x1 + crop_sz_x

    y1 = round(bby + 0.5 * bbh - crop_sz_y * 0.5)
    y2 = y1 + crop_sz_y

    # Move box inside image
    shift_x = max(0, -x1) + min(0, im_w - x2)
    x1 += shift_x
    x2 += shift_x

    shift_y = max(0, -y1) + min(0, im_h - y2)
    y1 += shift_y
    y2 += shift_y

    out_x = (max(0, -x1) + max(0, x2 - im_w)) // 2
    out_y = (max(0, -y1) + max(0, y2 - im_h)) // 2
    shift_x = (-x1 - out_x) * (out_x > 0)
    shift_y = (-y1 - out_y) * (out_y > 0)

    x1 += shift_x
    x2 += shift_x
    y1 += shift_y
    y2 += shift_y

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_REPLICATE)

    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    # Resize image
    im_out = cv.resize(im_crop_padded, tuple(output_sz.long().tolist()))

    if mask is not None:
        mask_out = \
        F.interpolate(mask_crop_padded[None, None], tuple(output_sz.flip(0).long().tolist()), mode='nearest')[0, 0]

    crop_box = torch.Tensor([x1, y1, x2 - x1, y2 - y1])

    if mask is None:
        return im_out, crop_box
    else:
        return im_out, crop_box, mask_out


def crop_and_resize(im, box, crop_bb, output_sz, mask=None):
    if isinstance(output_sz, (float, int)):
        output_sz = (output_sz, output_sz)

    im_h = im.shape[0]
    im_w = im.shape[1]

    if crop_bb[2] < 1 or crop_bb[3] < 1:
        raise Exception('Too small bounding box.')

    x1 = crop_bb[0]
    x2 = crop_bb[0] + crop_bb[2]

    y1 = crop_bb[1]
    y2 = crop_bb[1] + crop_bb[3]

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_REPLICATE)

    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    # Resize image
    im_out = cv.resize(im_crop_padded, output_sz)

    if mask is not None:
        mask_out = F.interpolate(mask_crop_padded[None, None], (output_sz[1], output_sz[0]), mode='nearest')[0, 0]

    rescale_factor = output_sz[0] / crop_bb[2]

    # Hack
    if box is not None:
        box_crop = box.clone()
        box_crop[0] -= crop_bb[0]
        box_crop[1] -= crop_bb[1]

        box_crop *= rescale_factor
    else:
        box_crop = None

    if mask is None:
        return im_out, box_crop
    else:
        return im_out, box_crop, mask_out


def transform_box_to_crop(box: torch.Tensor, crop_box: torch.Tensor, crop_sz: torch.Tensor) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box - the box for which the co-ordinates are to be transformed
        crop_box - bounding box defining the crop in the original image
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """

    box_out = box.clone()
    box_out[:2] -= crop_box[:2]

    scale_factor = crop_sz / crop_box[2:]

    box_out[:2] *= scale_factor
    box_out[2:] *= scale_factor
    return box_out


def target_image_crop(frames, box_extract, box_gt, search_area_factor, output_sz, mode: str = 'replicate',
                      max_scale_change=None, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. If the crop area contains regions outside the image, it is shifted / shrunk so that it
    completely fits inside the image. The extracted crops are then resized to output_sz. Further, the co-ordinates of
    the box box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized
        mode - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
               If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
               If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis of the image.
        max_scale_change - Maximum allowed scale change when performing the crop (only applicable for 'inside' and 'inside_major')
        masks - Optional masks to apply the same crop.

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

    if isinstance(output_sz, (float, int)):
        output_sz = (output_sz, output_sz)

    if masks is None:
        frame_crops_boxes = [sample_target_adaptive(f, a, search_area_factor, output_sz, mode, max_scale_change)
                             for f, a in zip(frames, box_extract)]

        frames_crop, crop_boxes = zip(*frame_crops_boxes)
    else:
        frame_crops_boxes_masks = [
            sample_target_adaptive(f, a, search_area_factor, output_sz, mode, max_scale_change, mask=m)
            for f, a, m in zip(frames, box_extract, masks)]

        frames_crop, crop_boxes, masks_crop = zip(*frame_crops_boxes_masks)

    crop_sz = torch.Tensor(output_sz)

    # find the bb location in the crop
    box_crop = [transform_box_to_crop(bb_gt, crop_bb, crop_sz)
                for bb_gt, crop_bb in zip(box_gt, crop_boxes)]

    if masks is None:
        return frames_crop, box_crop
    else:
        return frames_crop, box_crop, masks_crop

def template_image_crop(frames, box_extract, box_gt, search_area_factor, output_sz, mode: str = 'replicate',
                      max_scale_change=None, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. If the crop area contains regions outside the image, it is shifted / shrunk so that it
    completely fits inside the image. The extracted crops are then resized to output_sz. Further, the co-ordinates of
    the box box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized
        mode - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
               If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
               If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis of the image.
        max_scale_change - Maximum allowed scale change when performing the crop (only applicable for 'inside' and 'inside_major')
        masks - Optional masks to apply the same crop.

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

    if isinstance(output_sz, (float, int)):
        output_sz = (output_sz, output_sz)

    if masks is None:
        frame_crops_boxes = [sample_target_adaptive(f, a, search_area_factor, output_sz, mode, max_scale_change)
                             for f, a in zip(frames, box_extract)]

        frames_crop, crop_boxes = zip(*frame_crops_boxes)
    else:
        frame_crops_boxes_masks = [
            sample_target_adaptive(f, a, search_area_factor, output_sz, mode, max_scale_change, mask=m)
            for f, a, m in zip(frames, box_extract, masks)]

        frames_crop, crop_boxes, masks_crop = zip(*frame_crops_boxes_masks)

    crop_sz = torch.Tensor(output_sz)

    # find the bb location in the crop
    box_crop = [transform_box_to_crop(bb_gt, crop_bb, crop_sz)
                for bb_gt, crop_bb in zip(box_gt, crop_boxes)]

    if masks is None:
        return frames_crop, box_crop
    else:
        return frames_crop, box_crop, masks_crop

def iou(reference, proposals):
    """Compute the IoU between a reference box with multiple proposal boxes.

    args:
        reference - Tensor of shape (1, 4).
        proposals - Tensor of shape (num_proposals, 4)

    returns:
        torch.Tensor - Tensor of shape (num_proposals,) containing IoU of reference box with each proposal box.
    """

    # Intersection box
    tl = torch.max(reference[:, :2], proposals[:, :2])
    br = torch.min(reference[:, :2] + reference[:, 2:], proposals[:, :2] + proposals[:, 2:])
    sz = (br - tl).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = reference[:, 2:].prod(dim=1) + proposals[:, 2:].prod(dim=1) - intersection

    return intersection / union


def rand_uniform(a, b, shape=1):
    """ sample numbers uniformly between a and b.
    args:
        a - lower bound
        b - upper bound
        shape - shape of the output tensor

    returns:
        torch.Tensor - tensor of shape=shape
    """
    return (b - a) * torch.rand(shape) + a


def perturb_box(box, min_iou=0.5, sigma_factor=0.1):
    """ Perturb the input box by adding gaussian noise to the co-ordinates

     args:
        box - input box
        min_iou - minimum IoU overlap between input box and the perturbed box
        sigma_factor - amount of perturbation, relative to the box size. Can be either a single element, or a list of
                        sigma_factors, in which case one of them will be uniformly sampled. Further, each of the
                        sigma_factor element can be either a float, or a tensor
                        of shape (4,) specifying the sigma_factor per co-ordinate

    returns:
        torch.Tensor - the perturbed box
    """

    if isinstance(sigma_factor, list):
        # If list, sample one sigma_factor as current sigma factor
        c_sigma_factor = random.choice(sigma_factor)
    else:
        c_sigma_factor = sigma_factor

    if not isinstance(c_sigma_factor, torch.Tensor):
        c_sigma_factor = c_sigma_factor * torch.ones(4)

    perturb_factor = torch.sqrt(box[2] * box[3]) * c_sigma_factor

    # multiple tries to ensure that the perturbed box has iou > min_iou with the input box
    for i_ in range(100):
        c_x = box[0] + 0.5 * box[2]
        c_y = box[1] + 0.5 * box[3]
        c_x_per = random.gauss(c_x, perturb_factor[0])
        c_y_per = random.gauss(c_y, perturb_factor[1])

        w_per = random.gauss(box[2], perturb_factor[2])
        h_per = random.gauss(box[3], perturb_factor[3])

        if w_per <= 1:
            w_per = box[2] * rand_uniform(0.15, 0.5)

        if h_per <= 1:
            h_per = box[3] * rand_uniform(0.15, 0.5)

        box_per = torch.Tensor([c_x_per - 0.5 * w_per, c_y_per - 0.5 * h_per, w_per, h_per]).round()

        if box_per[2] <= 1:
            box_per[2] = box[2] * rand_uniform(0.15, 0.5)

        if box_per[3] <= 1:
            box_per[3] = box[3] * rand_uniform(0.15, 0.5)

        box_iou = iou(box.view(1, 4), box_per.view(1, 4))

        # if there is sufficient overlap, return
        if box_iou > min_iou:
            return box_per, box_iou

        # else reduce the perturb factor
        perturb_factor *= 0.9

    return box_per, box_iou


def gauss_1d(sz, sigma, center, end_pad=0, density=False):
    k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1)
    gauss = torch.exp(-1.0 / (2 * sigma ** 2) * (k - center.reshape(-1, 1)) ** 2)
    if density:
        gauss /= math.sqrt(2 * math.pi) * sigma
    return gauss


def gauss_2d(sz, sigma, center, end_pad=(0, 0), density=False):
    if isinstance(sigma, (float, int)):
        sigma = (sigma, sigma)
    return gauss_1d(sz[0].item(), sigma[0], center[:, 0], end_pad[0], density).reshape(center.shape[0], 1, -1) * \
           gauss_1d(sz[1].item(), sigma[1], center[:, 1], end_pad[1], density).reshape(center.shape[0], -1, 1)


def gaussian_label_function(target_bb, sigma_factor, kernel_sz, feat_sz, image_sz, end_pad_if_even=True, density=False,
                            uni_bias=0):
    """Construct Gaussian label function."""

    if isinstance(kernel_sz, (float, int)):
        kernel_sz = (kernel_sz, kernel_sz)
    if isinstance(feat_sz, (float, int)):
        feat_sz = (feat_sz, feat_sz)
    if isinstance(image_sz, (float, int)):
        image_sz = (image_sz, image_sz)

    image_sz = torch.Tensor(image_sz)
    feat_sz = torch.Tensor(feat_sz)

    target_center = target_bb[:, 0:2] + 0.5 * target_bb[:, 2:4]
    target_center_norm = (target_center - image_sz / 2) / image_sz

    center = feat_sz * target_center_norm + 0.5 * \
             torch.Tensor([(kernel_sz[0] + 1) % 2, (kernel_sz[1] + 1) % 2])

    sigma = sigma_factor * feat_sz.prod().sqrt().item()

    if end_pad_if_even:
        end_pad = (int(kernel_sz[0] % 2 == 0), int(kernel_sz[1] % 2 == 0))
    else:
        end_pad = (0, 0)

    gauss_label = gauss_2d(feat_sz, sigma, center, end_pad, density=density)
    if density:
        sz = (feat_sz + torch.Tensor(end_pad)).prod()
        label = (1.0 - uni_bias) * gauss_label + uni_bias / sz
    else:
        label = gauss_label + uni_bias
    return label


def gauss_density_centered(x, std):
    """Evaluate the probability density of a Gaussian centered at zero.
    args:
        x - Samples.
        std - List of standard deviations
    """
    return torch.exp(-0.5 * (x / std) ** 2) / (math.sqrt(2 * math.pi) * std)


def gmm_density_centered(x, std):
    """Evaluate the probability density of a GMM centered at zero.
    args:
        x - Samples. Assumes dim=-1 is the component dimension and dim=-2 is feature dimension. Rest are sample dimension.
        std - Tensor of standard deviations
    """
    if x.dim() == std.dim() - 1:
        x = x.unsqueeze(-1)
    elif not (x.dim() == std.dim() and x.shape[-1] == 1):
        raise ValueError('Last dimension must be the gmm stds.')
    return gauss_density_centered(x, std).prod(-2).mean(-1)


def sample_gmm_centered(std, num_samples=1):
    """Sample from a GMM distribution centered at zero:
    args:
        std - Tensor of standard deviations
        num_samples - number of samples
    """
    num_components = std.shape[-1]
    num_dims = std.numel() // num_components

    std = std.view(1, num_dims, num_components)

    # Sample component ids
    k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
    std_samp = std[0, :, k].t()

    # Sample
    x_centered = std_samp * torch.randn(num_samples, num_dims)
    prob_dens = gmm_density_centered(x_centered, std)

    return x_centered, prob_dens


def sample_gmm(mean, std, num_samples=1):
    """Sample from a GMM distribution:
    args:
        mean - a single mean vector
        std - Tensor of standard deviations
        num_samples - number of samples
    """
    num_dims = mean.numel()
    num_components = std.shape[-1]

    mean = mean.view(1, num_dims)
    std = std.view(1, -1, num_components)

    # Sample component ids
    k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
    std_samp = std[0, :, k].t()

    # Sample
    x_centered = std_samp * torch.randn(num_samples, num_dims)
    x = x_centered + mean
    prob_dens = gmm_density_centered(x_centered, std)

    return x, prob_dens


def sample_box_gmm(mean_box, proposal_sigma, gt_sigma=None, num_samples=1, add_mean_box=False):
    """Sample boxes from a Gaussian mixture model.
    args:
        mean_box - Center (or mean) bounding box
        proposal_sigma - List of standard deviations for each Gaussian
        gt_sigma - Standard deviation of the ground truth distribution
        num_samples - Number of sampled boxes
        add_mean_box - Also add mean box as first element

    returns:
        proposals, proposal density and ground truth density for all samples
    """
    center_std = torch.Tensor([s[0] for s in proposal_sigma])
    sz_std = torch.Tensor([s[1] for s in proposal_sigma])
    std = torch.stack([center_std, center_std, sz_std, sz_std])

    mean_box = mean_box.view(1, 4)
    sz_norm = mean_box[:, 2:].clone()

    # Sample boxes
    proposals_rel_centered, proposal_density = sample_gmm_centered(std, num_samples)

    # Add mean and map back
    mean_box_rel = rect_to_rel(mean_box, sz_norm)
    proposals_rel = proposals_rel_centered + mean_box_rel
    proposals = rel_to_rect(proposals_rel, sz_norm)

    if gt_sigma is None or gt_sigma[0] == 0 and gt_sigma[1] == 0:
        gt_density = torch.zeros_like(proposal_density)
    else:
        std_gt = torch.Tensor([gt_sigma[0], gt_sigma[0], gt_sigma[1], gt_sigma[1]]).view(1, 4)
        gt_density = gauss_density_centered(proposals_rel_centered, std_gt).prod(-1)

    if add_mean_box:
        proposals = torch.cat((mean_box, proposals))
        proposal_density = torch.cat((torch.Tensor([-1]), proposal_density))
        gt_density = torch.cat((torch.Tensor([1]), gt_density))

    return proposals, proposal_density, gt_density
