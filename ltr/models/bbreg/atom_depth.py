import torch.nn as nn
import ltr.models.backbone as backbones
import ltr.models.bbreg as bbmodels
from ltr import model_constructor
import cv2

class ATOMnet_Depth(nn.Module):
    """ ATOM network module"""
    def __init__(self, feature_extractor, bb_regressor, bb_regressor_layer, extractor_grad=True):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(ATOMnet_Depth, self).__init__()

        self.feature_extractor = feature_extractor
        self.bb_regressor = bb_regressor
        self.bb_regressor_layer = bb_regressor_layer
        # self.depth_converter = depth_converter

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, train_depths, test_imgs, test_depths, train_bb, test_proposals):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions.
        test_imgs is thus of the form [sequence, batch, feature, row, col]

        # Song :
            train_imgs     : [sequence=1, batch=64, 3, 288, 288]
            train_depths   : [sequence=1, batch=64, 1, 288, 288] => [1, 64, K, 288, 288]
            test_imgs      : [sequence=1, batch=64, 3, 288, 288]
            test_depths    : [sequence=1, batch=64, 1, 288, 288] => [1, 64, K*16, 288, 288] depth probability
            train_ bb      : [sequence=1, batch=64, 4], is it the coordinators in train_imgs ???
            test_proposals : [sequence=1, batch64, num_proposal=16, 4]
        """

        num_sequences = train_imgs.shape[-4]                                    # Song : 64 ??? it should be the batch_size
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1  # 1, ??? it should be the sequence size
        num_test_images = test_imgs.shape[0] if test_imgs.dim() == 5 else 1     # 1

        # Extract backbone features                                             # reshape : [sequence*batch, 3, 288, 288]
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        train_feat_iou = [feat for feat in train_feat.values()] # layer3 : [64, 128, 36, 36], layer4 : [64, 256, 18, 18]
        test_feat_iou = [feat for feat in test_feat.values()]   # layer3 : [64, 128, 36, 36], layer4 : [64, 256, 18, 18]

        # Song : convert depth image to object probablity based on the K-clusters of depths
        # train_depths_prob = self.depth_converter(train_depths, train_bb, K=3, label='train')
        # test_depths_prob = self.depth_converter(test_depths, test_proposals, K=3, label='test')

        # Obtain iou prediction
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou,
                                     train_depths, test_depths,                            # [K, 288, 288], [K*proposal, 288, 288]
                                     train_bb.reshape(num_train_images, num_sequences, 4),           # [1, 64, 4]
                                     test_proposals.reshape(num_train_images, num_sequences, -1, 4)) # [1, 64, 16, 4]
        return iou_pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)


@model_constructor
def atom_resnet18(iou_input_dim=(256,256), iou_inter_dim=(256,256), backbone_pretrained=True):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    # Bounding box regressor
    # iou_predictor = bbmodels.AtomIoUNet(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)
    iou_predictor = bbmodels.AtomIoUNet_Depth(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    net = ATOMnet_Depth(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                  extractor_grad=False)

    return net


@model_constructor
def atom_resnet50(iou_input_dim=(256,256), iou_inter_dim=(256,256), backbone_pretrained=True):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # Bounding box regressor
    # iou_predictor = bbmodels.AtomIoUNet(input_dim=(4*128,4*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)
    iou_predictor = bbmodels.AtomIoUNet_Depth(input_dim=(4*128,4*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    net = ATOMnet_Depth(feature_extractor=backbone_net, bb_regressor=iou_predictor, bb_regressor_layer=['layer2', 'layer3'],
                  extractor_grad=False)

    return net
