import math, time
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from base import BaseModel
from utils.losses import *
from models.encoder import Encoder
from models.modeling.deeplab import DeepLab as DeepLab_v3p
from models.modeling.deeplab_SubCls import Deeplab_SubCls as Deeplab_SubCls
import numpy as np


# from models import loss

class Test(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, ignore_index=None, testing=False, pretrained=True):

        super(Test, self).__init__()
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

        if self.backbone == 'deeplab_v3+':
            self.encoder = DeepLab_v3p(backbone='resnet{}'.format(self.layers))
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

    def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None, curr_iter=None, epoch=None, gpu=None,
                gt_l=None, ul1=None, br1=None, \
                ul2=None, br2=None, flip=None):
        if not self.training:
            with torch.no_grad():
                feat = self.encoder(x_l)
                enc = self.classifier(feat)
                output_l = F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            return output_l
        else:
            raise ValueError("No such mode {}".format(self.mode))

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
        return chain(self.encoder.get_module_params(), self.classifier.parameters())


class Save_Features(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, ignore_index=None, testing=False, pretrained=True):

        super(Save_Features, self).__init__()
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

        if self.backbone == 'deeplab_v3+':
            self.encoder = DeepLab_v3p(backbone='resnet{}'.format(self.layers))
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

    def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None, curr_iter=None, epoch=None, gpu=None,
                gt_l=None, ul1=None, br1=None, \
                ul2=None, br2=None, flip=None):
        if not self.training:
            with torch.no_grad():
                feat = self.encoder(x_l)
                enc = self.classifier(feat)
                output_l = F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            # return output_l
            return feat, output_l
        else:
            raise ValueError("No such mode {}".format(self.mode))

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
        return chain(self.encoder.get_module_params(), self.classifier.parameters())


class Baseline(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, ignore_index=None, testing=False, pretrained=True):

        super(Baseline, self).__init__()
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

        if self.backbone == 'deeplab_v3+':
            self.encoder = DeepLab_v3p(backbone='resnet{}'.format(self.layers))
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
        elif self.backbone == 'psp':
            self.encoder = Encoder(pretrained=pretrained)
            self.classifier = nn.Conv2d(self.out_dim, num_classes, kernel_size=1, stride=1)
        else:
            raise ValueError("No such backbone {}".format(self.backbone))

    def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None, curr_iter=None, epoch=None, gpu=None,
                gt_l=None, ul1=None, br1=None, \
                ul2=None, br2=None, flip=None):
        if not self.training:
            enc = self.encoder(x_l)
            enc = self.classifier(enc)
            return F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)

        if self.mode == 'supervised':
            feat = self.encoder(x_l)
            enc = self.classifier(feat)
            output_l = F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)

            loss_sup = self.sup_loss(output_l, target_l, ignore_index=self.ignore_index,
                                     temperature=1.0) * self.sup_loss_w

            curr_losses = {'Ls': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup

            return total_loss, curr_losses, outputs
        else:
            raise ValueError("No such mode {}".format(self.mode))

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
        return chain(self.encoder.get_module_params(), self.classifier.parameters())


class USRN(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, ignore_index=None, testing=False, pretrained=True,
                 num_features=256):
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
        # {
        self.loss_weight_uncer_relevent = conf['loss_weight_uncer_relevent']
        self.loss_weight_dis_and_entro = conf['loss_weight_dis_and_entro']
        self.nb_prototype = conf['nb_prototype']
        # }
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
        self.num_classes_subcls = sum(self.split_list) # for voc331, num_classes_subcls=177

        if self.backbone == 'deeplab_v3+':
            self.encoder = Deeplab_SubCls(backbone='resnet{}'.format(self.layers))
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
           
            self.classifier_SubCls = nn.Sequential(nn.Dropout(0.1),
                                                   nn.Conv2d(256, self.num_classes_subcls, kernel_size=1, stride=1))
            #{
            self.classifier_SubCls_dm = nn.Sequential(nn.Dropout(0.1),
                                                   nn.Conv2d(self.nb_prototype, self.num_classes_subcls, kernel_size=1, stride=1))
            for m in self.classifier_SubCls_dm.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.SyncBatchNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            #}
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

        self.total_loss = 0
        self.curr_losses = {}

        # {
        self.DMlayer = Distanceminimi_Layer_learned(in_features=(num_features), out_features=self.nb_prototype,
                                                    dist='cos')
        self.DMBN = nn.BatchNorm2d(self.nb_prototype)
        self.get_uncer = nn.Conv2d(self.nb_prototype, self.num_classes_subcls, 1)
        # }

    def forward(self, x_l=None, target_l=None, target_l_subcls=None, x_ul=None, target_ul=None,
                curr_iter=None, epoch=None, gpu=None, gt_l=None, ul1=None, br1=None, ul2=None, br2=None, flip=None):

        if not self.training:
            enc, _ = self.encoder(x_l)
            enc = self.classifier(enc)
            return F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)

        if self.mode == 'supervised':
            outputs = self.comp_features_and_supervised_loss(x_l, target_l, target_l_subcls)

        elif self.mode == 'semi':
            outputs = self.comp_features_and_supervised_loss(x_l, target_l, target_l_subcls)

            if epoch < self.epoch_start_unsup:
                return self.total_loss, self.curr_losses, outputs

            x_w = x_ul[:, 0, :, :, :]  # Weak Aug; x_ul: [batch_size, 2, 3, H, W]
            x_s = x_ul[:, 1, :, :, :]  # Strong Aug

            logits_w, logits_SubCls_w = self.get_logits(x_w)
            # {
            # logits_s, logits_SubCls_s = self.get_logits(x_s)
            logits_s, logits_SubCls_s, uncer_sub_s, omega_sub, subembedding_ = self.get_logits_and_uncer(x_s)
            # }
            seg_w_SubCls = F.softmax(logits_SubCls_w, 1)
            pseudo_logits_SubCls_w = seg_w_SubCls.max(1)[
                0].detach()  # the maximum probability value along the class dimension of seg_w_SubCls for each pixel. # "pseudo-logits" seem to be calculated by taking the maximum probability values from the softmax output, "pseudo" is used here to indicate that these values are not true logits but are derived from probabilities.
            pseudo_label_SubCls_w = seg_w_SubCls.max(1)[
                1].detach()  # the class label with the maximum probability value along the class dimension of seg_w_SubCls for each pixel. It represents the predicted class label for each pixel.

            self.comp_features_and_unsupervised_sub_loss(logits_SubCls_s, pseudo_logits_SubCls_w, pseudo_label_SubCls_w)
            self.comp_features_and_unsupervised_reg_loss(logits_w, logits_s, seg_w_SubCls, pseudo_label_SubCls_w)

            self.curr_losses['L_task'] = self.total_loss

            # {
            self.curr_losses['L_uncertainty'] = self.uncertainty_loss(uncer_sub_s, logits_SubCls_s, logits_SubCls_w)
            self.curr_losses['L_dissimilar'] = self.dissimilar_loss(omega_sub)
            self.curr_losses['L_entropy'] = self.entropy_loss(subembedding_)
            self.total_loss = self.total_loss +  self.loss_weight_uncer_relevent * (self.curr_losses['L_uncertainty'] + self.loss_weight_dis_and_entro * (self.curr_losses['L_dissimilar'] + self.curr_losses['L_entropy']))
            # }
            #  a*(uncer + b*(en+dis))
            # task + a*uncer + b*(en+dis)
            return self.total_loss, self.curr_losses, outputs

        else:
            raise ValueError("No such mode {}".format(self.mode))

    def comp_features_and_supervised_loss(self, x_l, target_l, target_l_subcls):
        feat, feat_SubCls = self.encoder(x_l)
        enc = self.classifier(feat)
        output_l = F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)
        loss_sup = self.sup_loss(output_l, target_l, ignore_index=self.ignore_index,
                                 temperature=1.0) * self.sup_loss_w
        self.curr_losses = {'Ls': loss_sup}
        outputs = {'sup_pred': output_l}
        self.total_loss = loss_sup

        enc_SubCls = self.classifier_SubCls(feat_SubCls)
        output_l_SubCls = F.interpolate(enc_SubCls, size=x_l.size()[2:], mode='bilinear', align_corners=True)
        loss_sup_SubCls = self.sup_loss(output_l_SubCls, target_l_subcls, ignore_index=self.ignore_index,
                                        temperature=1.0) * self.sup_loss_w
        self.curr_losses['Ls_sub'] = loss_sup_SubCls
        self.total_loss = self.total_loss + loss_sup_SubCls * self.loss_weight_subcls

        return outputs

    def comp_features_and_unsupervised_sub_loss(self, logits_SubCls_s, pseudo_logits_SubCls_w, pseudo_label_SubCls_w):
        pos_mask_SubCls = pseudo_logits_SubCls_w > self.pos_thresh_value
        loss_unsup_SubCls = (
                    F.cross_entropy(logits_SubCls_s, pseudo_label_SubCls_w, reduction='none') * pos_mask_SubCls).mean()
        self.curr_losses['Lu_sub'] = loss_unsup_SubCls
        self.total_loss = self.total_loss + loss_unsup_SubCls * self.loss_weight_unsup * self.loss_weight_subcls
        return

    def comp_features_and_unsupervised_reg_loss(self, logits_w, logits_s, seg_w_SubCls, pseudo_label_SubCls_w):
        SubCls_reg_label = self.SubCls_to_ParentCls(pseudo_label_SubCls_w)
        seg_w = F.softmax(logits_w, 1)
        SubCls_reg_label_one_hot = F.one_hot(SubCls_reg_label, num_classes=self.num_classes).permute(0, 3, 1, 2)
        seg_w_ent = torch.sum(self.prob_2_entropy(seg_w.detach()), 1)
        seg_w_SubCls_ent = torch.sum(self.prob_2_entropy(seg_w_SubCls.detach()), 1)
        SubCls_reg_label_one_hot_ent_reg = SubCls_reg_label_one_hot.clone()
        SubCls_reg_label_one_hot_ent_reg[
            (seg_w_SubCls_ent > seg_w_ent).unsqueeze(1).repeat(1, seg_w.shape[1], 1, 1)] = 1
        seg_w_reg = seg_w * SubCls_reg_label_one_hot_ent_reg
        pseudo_logits_w_reg = seg_w_reg.max(1)[0].detach()
        pseudo_label_w_reg = seg_w_reg.max(1)[1].detach()
        pos_mask_reg = pseudo_logits_w_reg > self.pos_thresh_value
        loss_unsup_reg = (F.cross_entropy(logits_s, pseudo_label_w_reg, reduction='none') * pos_mask_reg).mean()
        self.curr_losses['Lu_reg'] = loss_unsup_reg
        self.total_loss = self.total_loss + loss_unsup_reg * self.loss_weight_unsup
        return

    def get_logits(self, x):
        feat, feat_SubCls = self.encoder(x)
        if self.downsample:
            feat = F.avg_pool2d(feat, kernel_size=2, stride=2)
            feat_SubCls = F.avg_pool2d(feat_SubCls, kernel_size=2, stride=2)
        logits = self.classifier(feat)
        logits_SubCls = self.classifier_SubCls(feat_SubCls)
        return logits, logits_SubCls

    def get_logits_and_uncer(self, x):
        feat, feat_SubCls = self.encoder(x)
        if self.downsample:
            feat = F.avg_pool2d(feat, kernel_size=2, stride=2)
            feat_SubCls = F.avg_pool2d(feat_SubCls, kernel_size=2, stride=2)
        # { original class
        # embedding_, omega = self.DMlayer(feat)
        # embedding = torch.exp(-embedding_)
        # out = self.DMBN(embedding)
        # final_uncer = self.get_uncer(out)
        # }
        # { sub class
        subembedding_, omega_sub = self.DMlayer(feat_SubCls)
        subembedding = torch.exp(-subembedding_)
        out_feat_sub = self.DMBN(subembedding)
        uncer_sub = self.get_uncer(out_feat_sub)
        logits = self.classifier(feat)
        logits_SubCls = self.classifier_SubCls_dm(out_feat_sub)
        return logits, logits_SubCls, uncer_sub, omega_sub.squeeze(), subembedding_
        # }
        # logits = self.classifier(feat)
        # logits_SubCls = self.classifier_SubCls(feat_SubCls)
        # return logits, logits_SubCls

    # {
    def uncertainty_loss(self, uncer, outputs, pseudo_gt):
        abs_error = abs(outputs.detach() - pseudo_gt)
        #abs_error[abs_error > 1] = 1
        abs_error = abs_error.detach()
        #{
        # loss = nn.CrossEntropyLoss(reduction='mean')(uncer, abs_error)
        loss = nn.CrossEntropyLoss(reduction='mean')(uncer, abs_error) / self.num_classes
        # loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).cuda(), reduction='mean')(uncer, abs_error)
        # }
        return loss

    def entropy_loss(self, embedding):
        embedding = nn.Softmax(dim=1)(embedding)
        minus_entropy = embedding * torch.log(embedding)
        minus_entropy = torch.sum(minus_entropy, dim=1)
        return minus_entropy.mean()

    def dissimilar_loss(self, protos):
        loss = -1 * torch.mean(torch.cdist(protos, protos))
        return loss

    # }

    def prob_2_entropy(self, prob):
        n, c, h, w = prob.size()
        return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

    def SubCls_to_ParentCls(self, label_SubCls):
        label_SubCls_to_ParentCls = label_SubCls.clone()
        subclasses = np.cumsum(np.asarray(self.split_list))
        subclasses = np.insert(subclasses, 0, 0)
        parentclasses = np.uint8(np.linspace(1, len(self.split_list), len(self.split_list)) - 1)
        for subcls_lower, subcls_upper, parcls in zip(np.flip(subclasses[:-1]), np.flip(subclasses[1:]),
                                                      np.flip(parentclasses)):
            label_SubCls_to_ParentCls[(label_SubCls >= subcls_lower) * (label_SubCls < subcls_upper)] = parcls
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


# {
class Distanceminimi_Layer_learned(nn.Module):
    def __init__(self, in_features=0, out_features=0, dist='lin'):
        super(Distanceminimi_Layer_learned, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dist = dist
        self.omega = nn.Parameter(torch.Tensor(1, out_features, in_features, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.omega, mean=0, std=1)  # /self.out_features)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.cosine_similarity(x, self.omega, dim=2, eps=1e-30)

        return out, self.omega
# }