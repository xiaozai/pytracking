# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Song edited the DETR model
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops

from .res_backbone import build_backbone
from .transformer import build_transformer_track

class DETR(nn.Module):
    """
        Song : This is the DETR module that performs object tracking

        The differences are :
            1) To fix the query_embed, expecting a tensor [batchsize, N=1, 256],
               the deep feature map from Template branch on ResNet50, [batchsize, C, H, W]
               the QueryProj is used
            2) To remove the num_classes, num_queries, because they are 1
            3) To replace the class_embed to conf_embed ??? to predict the confidence?
    """
    def __init__(self, backbone, transformer, postprocessors):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
        """
        super().__init__()

        self.transformer = transformer
        hidden_dim = transformer.d_model                                        # 512
        self.conf_embed = MLP(hidden_dim, hidden_dim, 1, 3)                     # Song added
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_proj = QueryProj(backbone.num_channels, hidden_dim)          # Song added, project Template features into query
        self.backbone = backbone
        self.postprocessors = postprocessors

    def forward(self, search_imgs, template_imgs, template_bb=None, target_sizes=None):
        """ The forward expects a NestedTensor, which consists of:
               - template_imgs : batched images, of shape [batch_size x 3 x H x W], which are the template branch
               - template_bb   : batched bounding boxes, of shape [batch_size x 4]

               - search_imgs : batched test images , of shape [batch_size x 3 x H x W], which are the test branch


            It returns a dict with the following elements:

               - "pred_conf": the confidence values for the predicted bboxes

               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
        """

        features, pos = self.backbone(search_imgs)
        src, mask = features[-1].decompose()
        assert mask is not None

        '''
        Song : should I crop the template_imgs with the template_bb ??? or do it on the feature maps?????
                if the backbone is only convs layers, we can use the cropped template_imgs
        '''
        template_features, _ = self.backbone(template_imgs)
        template_src, template_mask = template_features[-1].decompose()
        assert template_mask is not None

        '''
        THe only difference is to fix the query_embedding with the template feature

        HS : it outputs the stact of 6 tensor, but keep the first one?, e.g. [dc_layers=6, num_classes=100, batch, 256]
        '''
        hs = self.transformer(self.input_proj(src), mask, self.query_proj(template_src), pos[-1])[0]

        outputs_conf = self.conf_embed(hs)                                      # Song added
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {'pred_conf': outputs_conf[-1], 'pred_boxes': outputs_coord[-1]}

        out = postprocessors(out, target_sizes)
        return out

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api
        Song : ? for tracking , we need to output the box of xywh and confidence
    """
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_conf, out_bbox = outputs['pred_conf'], outputs['pred_boxes']

        assert len(out_conf) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        scores = out_conf

        boxes = box_ops.box_cxcywh_to_xywh(out_bbox)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s,  'boxes': b} for s, b in zip(scores, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class QueryProj(nn.Module):
    """
    Song : project the template features into object query features
    [batch, C=512, H, W] -> [batch, N=1, output_dim=256]
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()

        pool_size = 5
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1) # C=512 -> 256
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.fc = nn.Linear(pool_size*pool_size*output_dim, output_dim)

    def forward(self, x):
        x = self.conv(x)              # [batch, 512, H, W] - > [batch, 256, H, W]
        x = self.pool(x)              # [batch, 256, H, W] - > [batch, 256, 5, 5]
        x = x.flatten(1)              # [batch, 256, 5, 5] - > [batch, 256*5*5]
        x = F.relu(self.fc(x))        # [batch, 256*5*5] -> [batch, 256]
        return torch.unsqueeze(x, 1)  # [batch, 1, 256]

@model_constructor
def transformer_track(args):

    backbone = build_backbone(args)
    transformer = build_transformer(args)
    postprocessors = PostProcess()

    model = DETR(backbone, transformer, postprocessors)

    return model
