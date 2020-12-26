import torch
import torch.nn.functional as F
from torch import nn

import ltr.data.box_ops as box_ops

class SetCriterion(nn.Module):
    """
    Song edited

    This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, losses=['boxes_iou_conf']):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()

        self.losses = losses

    def loss_boxes_iou_conf(self, outputs, targets):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           # # targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           # # The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
           Song : The target boxes are in format (x, y, w, h)
        """
        assert 'pred_boxes' in outputs

        src_boxes = outputs['pred_boxes']
        # target_boxes = torch.cat([t['boxes'] for t in zip(targets)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, targets, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum()

        # target_confidences = torch.diag(box_ops.generalized_box_iou(
        #     box_ops.box_cxcywh_to_xyxy(src_boxes),
        #     box_ops.box_cxcywh_to_xyxy(target_boxes)))
        target_confidences = torch.diag(box_ops.generalized_box_iou(src_boxes, targets))

        loss_giou = 1 - target_confidences
        losses['loss_giou'] = loss_giou.sum()

        # Song : use the giou as the confidence
        src_confidences = outputs['pred_conf']
        loss_confidence = F.mse_loss(src_confidences, target_confidences)
        losses['loss_conf'] = loss_confidence.sum()

        return losses

    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {
            'boxes_iou_conf': self.loss_boxes_iou_conf,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        return losses