import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, time
from itertools import chain
from base import BaseModel
from utils.losses import *
from models.encoder import Encoder
from models.modeling.deeplab import DeepLab as DeepLab_v3p
from models.modeling.deeplab_SubCls import Deeplab_SubCls as Deeplab_SubCls
#_______________________________________________________________________________________#
class Distanceminimi_Layer_learned(nn.Module):
    def __init__(self, in_features=0, out_features=0, dist='lin'):
        super(Distanceminimi_Layer_learned, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dist=dist
        self.omega = nn.Parameter(torch.Tensor(1, out_features, in_features, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.normal_(self.omega, mean=0, std=1)#/self.out_features)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = torch_nn_func.cosine_similarity(x, self.omega, dim=2, eps=1e-30)

        return out, self.omega
    
    
# class silog_loss(nn.Module):
#     def __init__(self, variance_focus):
#         super(silog_loss, self).__init__()
#         self.variance_focus = variance_focus

#     def forward(self, depth_est, depth_gt, mask):
#         d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
#         return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


class entropy_loss(nn.Module):
    def __init__(self):
        super(entropy_loss, self).__init__()

    def forward(self, embedding):
        embedding = nn.Softmax(dim=1)(embedding)
        minus_entropy = embedding * torch.log(embedding)
        minus_entropy = torch.sum(minus_entropy, dim=1)
        return minus_entropy.mean()


class uncertainty_loss(nn.Module):
    def __init__(self, args):
        super(uncertainty_loss, self).__init__()
        self.max_depth = args.max_depth

    def forward(self, uncer, final_depth, depth_gt, mask):
        abs_error = abs(final_depth.detach() - depth_gt)/self.max_depth
        abs_error[abs_error>1] = 1
        abs_error = abs_error[mask].detach()
        loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([5.0]).cuda(), reduction='mean')(uncer[mask], abs_error)
        return loss


class dissimilar_loss(nn.Module):
    def __init__(self):
        super(dissimilar_loss, self).__init__()

    def forward(self, protos):
        loss = -1 * torch.mean(torch.cdist(protos, protos))
        return loss
#_______________________________________________________________________________________#

class USRN(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, ignore_index=None, testing=False, pretrained=True):
        super(USRN, self).__init__()
        assert int(conf['supervised']) + int(conf['semi']) == 1, 'one mode only'
        if conf['supervised']:
            self.mode = 'supervised'
        elif conf['semi']:
            self.mode = 'semi'
        else:
            raise ValueError('No such mode choice {}'.format(self.mode))
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.sup_loss_w = conf['supervised_w']
        self.sup_loss = sup_loss
        self.downsample = conf['downsample']
        self.backbone = conf['backbone']
        self.layers = conf['layers']
        self.out_dim = conf['out_dim']
        assert self.layers in [50, 101]

        self.loss_weight_subcls = conf['loss_weight_subcls']
        self.loss_weight_unsup = conf['loss_weight_unsup']
        ### VOC Dataset
        if conf['n_labeled_examples'] == 662:
            self.split_list = [132, 2, 1, 1, 1, 2, 3, 4, 7, 2, 1, 2, 6, 2, 2, 15, 1, 1, 2, 2, 1]
        elif conf['n_labeled_examples'] == 331:
            self.split_list = [121, 2, 1, 1, 1, 1, 3, 3, 6, 3, 1, 2, 6, 2, 2, 15, 1, 1, 2, 2, 1]
        elif conf['n_labeled_examples'] == 165:
            self.split_list = [136, 2, 2, 1, 1, 1, 2, 4, 8, 3, 1, 2, 7, 2, 2, 18, 1, 1, 1, 3, 3]
        ### Cityscapes Dataset
        elif conf['n_labeled_examples'] == 372:
            self.split_list = [42, 7, 26, 1, 2, 2, 1, 1, 19, 2, 5, 2, 1, 8, 1, 1, 1, 1, 1]
        elif conf['n_labeled_examples'] == 186:
            self.split_list = [45, 7, 28, 1, 2, 2, 1, 1, 20, 2, 5, 2, 1, 8, 1, 1, 1, 1, 1]
        elif conf['n_labeled_examples'] == 93:
            self.split_list = [38, 6, 22, 1, 2, 2, 1, 1, 17, 2, 5, 1, 1, 7, 1, 1, 1, 1, 1]
        self.num_classes_subcls = sum(self.split_list)

        if self.backbone == 'deeplab_v3+':
            self.encoder = Deeplab_SubCls(backbone='resnet{}'.format(self.layers))
            #{ ... }
            self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
            for m in self.classifier.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.SyncBatchNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            self.classifier_SubCls = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(256, self.num_classes_subcls, kernel_size=1, stride=1))
            for m in self.classifier_SubCls.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.SyncBatchNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.backbone == 'psp':
            self.encoder = Encoder(pretrained=pretrained)
            self.classifier = nn.Conv2d(self.out_dim, num_classes, kernel_size=1, stride=1)
        else:
            raise ValueError("No such backbone {}".format(self.backbone))
        if self.mode == 'semi':
            self.epoch_start_unsup = conf['epoch_start_unsup']
            self.step_save = conf['step_save']
            self.pos_thresh_value = conf['pos_thresh_value']
            self.stride = conf['stride']

    def forward(self, x_l=None, target_l=None, target_l_subcls=None, x_ul=None, target_ul=None,
                curr_iter=None, epoch=None, gpu=None, gt_l=None, ul1=None, br1=None, ul2=None, br2=None, flip=None):
        if not self.training:
            # Inference mode for original-class segmentation
            enc, _ = self.encoder(x_l)
            enc = self.classifier(enc)
            return F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)

        if self.mode == 'supervised':
            # Supervised training mode for original-class and sub-class segmentation
            feat, feat_SubCls = self.encoder(x_l)

            # Forward pass through the classifier for original-class segmentation
            enc = self.classifier(feat)
            output_l = F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            
            # Calculate the supervised loss for original-class segmentation
            loss_sup = self.sup_loss(output_l, target_l, ignore_index=self.ignore_index,
                                    temperature=1.0) * self.sup_loss_w
            curr_losses = {'Ls': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup

            # Forward pass through the classifier for sub-class segmentation
            enc_SubCls = self.classifier_SubCls(feat_SubCls)
            output_l_SubCls = F.interpolate(enc_SubCls, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            
            # Calculate the supervised loss for sub-class segmentation
            loss_sup_SubCls = self.sup_loss(output_l_SubCls, target_l_subcls, ignore_index=self.ignore_index, temperature=1.0) * self.sup_loss_w
            curr_losses['Ls_sub'] = loss_sup_SubCls
            total_loss = total_loss + loss_sup_SubCls * self.loss_weight_subcls

            return total_loss, curr_losses, outputs

        elif self.mode == 'semi':
        # Semi-supervised training mode for original-class and sub-class segmentation

        # Supervised with labeled data
            feat, feat_SubCls = self.encoder(x_l)

            # Forward pass through the classifier for original-class segmentation
            enc = self.classifier(feat)
            output_l = F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            
            # Calculate the supervised loss for original-class segmentation of labeled data
            loss_sup = self.sup_loss(output_l, target_l, ignore_index=self.ignore_index,
                                    temperature=1.0) * self.sup_loss_w
            curr_losses = {'Ls': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup

            # Forward pass through the classifier for sub-class segmentation of labeled data
            enc_SubCls = self.classifier_SubCls(feat_SubCls)
            output_l_SubCls = F.interpolate(enc_SubCls, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            
            # Calculate the supervised loss for sub-class segmentation
            loss_sup_SubCls = self.sup_loss(output_l_SubCls, target_l_subcls, ignore_index=self.ignore_index, temperature=1.0) * self.sup_loss_w
            curr_losses['Ls_sub'] = loss_sup_SubCls
            total_loss = total_loss + loss_sup_SubCls * self.loss_weight_subcls

            if epoch < self.epoch_start_unsup:
                # Return the losses and outputs if still in the supervised training phase
                return total_loss, curr_losses, outputs

        # Semi-supervised training phase with unlabeled data
            x_w = x_ul[:, 0, :, :, :]  # Weak Augmentation
            x_s = x_ul[:, 1, :, :, :]  # Strong Augmentation

            # 1. classifier (data to logits) for orginal-class and sub-class (weak and strong) 

            # Forward pass through the classifier for orginal-class segmentation (weak and strong)
            feat_s, feat_SubCls_s = self.encoder(x_s)
            if self.downsample: # perform average pooling to reduce the feature map size
                feat_s = F.avg_pool2d(feat_s, kernel_size=2, stride=2)
            logits_s = self.classifier(feat_s)
            
            feat_w, feat_SubCls_w = self.encoder(x_w)
            if self.downsample:
                feat_w = F.avg_pool2d(feat_w, kernel_size=2, stride=2)
            logits_w = self.classifier(feat_w)

            # Forward pass through the classifier for sub-class segmentation (weak and strong)
            if self.downsample:
                feat_SubCls_s = F.avg_pool2d(feat_SubCls_s, kernel_size=2, stride=2)
            logits_SubCls_s = self.classifier_SubCls(feat_SubCls_s)
            if self.downsample:
                feat_SubCls_w = F.avg_pool2d(feat_SubCls_w, kernel_size=2, stride=2)
            logits_SubCls_w = self.classifier_SubCls(feat_SubCls_w)

            # 2. sub-class

            # 2.1 label: pseudo-label (logits to probs to label) for sub-class (weak)
            # Perform softmax on the sub-class segmentation logits to obtain probabilities
            seg_w_SubCls = F.softmax(logits_SubCls_w, 1)
            # Use the maximum probability as pseudo-logits for unlabeled sub-class segmentation
            pseudo_logits_SubCls_w = seg_w_SubCls.max(1)[0].detach()
            # Use the corresponding class label of the maximum probability as pseudo-label for unlabeled sub-class segmentation
            pseudo_label_SubCls_w = seg_w_SubCls.max(1)[1].detach()

            # 2.2 mask: basic mask (weak)
            # Create a positive mask for unlabeled sub-class segmentation using a threshold
            pos_mask_SubCls = pseudo_logits_SubCls_w > self.pos_thresh_value
            
            # 2.3 self-traning loss for sub-class (compare strong's logits and weak's labels for sub-class)
            # Calculate the unsupervised loss for sub-class segmentation with the positive mask
            loss_unsup_SubCls = (F.cross_entropy(logits_SubCls_s, pseudo_label_SubCls_w, reduction='none') * pos_mask_SubCls).mean()
            curr_losses['Lu_sub'] = loss_unsup_SubCls
            total_loss = total_loss + loss_unsup_SubCls * self.loss_weight_unsup * self.loss_weight_subcls

            # 3. orginal-class

            # 3.1 label: convert sub-class labels to original-class labels (weak)
            # Convert the sub-class labels to parent-class labels for regularization
            SubCls_reg_label = self.SubCls_to_ParentCls(pseudo_label_SubCls_w)
            # Perform softmax on the weakly augmented segmentation logits to obtain probabilities
            seg_w = F.softmax(logits_w, 1)
            # Convert the sub-class labels to one-hot representation
            SubCls_reg_label_one_hot = F.one_hot(SubCls_reg_label, num_classes=self.num_classes).permute(0, 3, 1, 2)

            # 3.2 mask: entropy-based mask for orginal-class (weak)
            # Compute the entropy with prob
            seg_w_ent = torch.sum(self.prob_2_entropy(seg_w.detach()), 1)
            seg_w_SubCls_ent = torch.sum(self.prob_2_entropy(seg_w_SubCls.detach()), 1)
            # Create a mask for the weakly augmented sub-class segmentation based on entropy
            SubCls_reg_label_one_hot_ent_reg = SubCls_reg_label_one_hot.clone()
            SubCls_reg_label_one_hot_ent_reg[(seg_w_SubCls_ent > seg_w_ent).unsqueeze(1).repeat(1,seg_w.shape[1], 1, 1)] = 1
            # Mask the weakly augmented sub-class segmentation with the entropy-based mask
            seg_w_reg = seg_w * SubCls_reg_label_one_hot_ent_reg
            # Use the maximum probability as pseudo-logits for weakly augmented sub-class segmentation
            pseudo_logits_w_reg = seg_w_reg.max(1)[0].detach()
            # Use the corresponding class label of maximum probability as pseudo-label for weakly augmented sub-class segmentation
            pseudo_label_w_reg = seg_w_reg.max(1)[1].detach()
            # Create a positive mask for weakly augmented sub-class segmentation using a threshold
            pos_mask_reg = pseudo_logits_w_reg > self.pos_thresh_value

            # 3.3 loss: entropy-based self-training loss (compare strong's logits and weak's labels)
            #                         (named as regularization loss in the code) 
            # Calculate the unsupervised loss for weakly augmented sub-class segmentation with the positive mask
            loss_unsup_reg = (F.cross_entropy(logits_s, pseudo_label_w_reg, reduction='none') * pos_mask_reg).mean()
            curr_losses['Lu_reg'] = loss_unsup_reg
            total_loss = total_loss + loss_unsup_reg * self.loss_weight_unsup

            return total_loss, curr_losses, outputs

        else:
            raise ValueError("No such mode {}".format(self.mode))

    def prob_2_entropy(self, prob):
        n, c, h, w = prob.size()
        return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


    def SubCls_to_ParentCls(self, label_SubCls):
        label_SubCls_to_ParentCls = label_SubCls.clone()
        subclasses = np.cumsum(np.asarray(self.split_list))
        subclasses = np.insert(subclasses, 0, 0)
        parentclasses = np.uint8(np.linspace(1,len(self.split_list),len(self.split_list))-1)
        for subcls_lower, subcls_upper, parcls in zip(np.flip(subclasses[:-1]), np.flip(subclasses[1:]), np.flip(parentclasses)):
            label_SubCls_to_ParentCls[(label_SubCls>=subcls_lower)*(label_SubCls<subcls_upper)] = parcls
        return label_SubCls_to_ParentCls.cuda().long()

    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        with torch.no_grad():
            tensors_gather = [torch.ones_like(tensor)
                            for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

            output = torch.cat(tensors_gather, dim=0)
        return output

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        return chain(self.encoder.get_module_params(), self.classifier.parameters(),
                    self.classifier_SubCls.parameters())

