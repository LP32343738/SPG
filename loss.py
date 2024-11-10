#author: Niwhskal
#https://github.com/Niwhskal


import torch
import cfg
from torch.distributions import MultivariateNormal as MVN
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

def build_discriminator_loss(x_true, x_fake):
    loss_real = adv_loss(x_true, 1)
    loss_fake = adv_loss(x_fake, 0)
    d_loss = loss_real + loss_fake
    # d_loss = -torch.mean(torch.log(torch.clamp(x_true, cfg.epsilon, 1.0)) + torch.log(torch.clamp(1.0 - x_fake, cfg.epsilon, 1.0)))
    return d_loss


def build_recognition_loss(preds, text):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    target = text[:, :]  # without [GO] Symbol
    rec_loss = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

    
    return rec_loss

def build_classifier_loss(logits, target):
    loss_c = torch.nn.CrossEntropyLoss()
    text_angle,char_angle,char_dist,char_aspt = logits
    loss_value_1 = loss_c(text_angle,target[0])
    loss_value_2 = loss_c(char_angle,target[1])
    loss_value_3 = loss_c(char_dist,target[2])
    loss_value_4 = loss_c(char_aspt,target[3])
    return loss_value_1,loss_value_2,loss_value_3,loss_value_4


def build_dice_loss(x_t, x_o):
       
    iflat = x_o.view(-1)
    tflat = x_t.view(-1)
    intersection = (iflat*tflat).sum()
    
    return 1. - torch.mean((2. * intersection + cfg.epsilon)/(iflat.sum() +tflat.sum()+ cfg.epsilon))

def build_l1_loss(x_t, x_o):
        
    return torch.mean(torch.abs(x_t - x_o))

def build_l1_loss_with_mask(x_t, x_o, mask):

    diff2 = (x_t* mask - x_o* mask) ** 2 
    result = torch.sum(diff2) / torch.sum(mask)
    return result
def build_perceptual_loss(x):        
    l = []
    for i, f in enumerate(x):
        l.append(build_l1_loss(f[0], f[1]))
    l = torch.stack(l, dim = 0)
    l = l.sum()
    return l

def bmc_loss_md(target, pred, noise_var):
    """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, d].
      target: A float tensor of size [batch, d].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    I = torch.eye(pred.shape[-1])
    logits = MVN(pred.unsqueeze(1), noise_var*I).log_prob(target.unsqueeze(0))  # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 
    
    return loss

def build_gram_matrix(x):

    x_shape = x.shape
    c, h, w = x_shape[1], x_shape[2], x_shape[3]
    matrix = x.view((-1, c, h * w))
    matrix1 = torch.transpose(matrix, 1, 2)
    gram = torch.matmul(matrix, matrix1) / (h * w * c)
    return gram

def build_style_loss(x):
        
    l = []
    for i, f in enumerate(x):
        f_shape = f[0].shape[0] * f[0].shape[1] *f[0].shape[2]
        f_norm = 1. / f_shape
        gram_true = build_gram_matrix(f[0])
        gram_pred = build_gram_matrix(f[1])
        l.append(f_norm * (build_l1_loss(gram_true, gram_pred)))
    l = torch.stack(l, dim = 0)
    l = l.sum()
    return l

def build_vgg_loss(x):
        
    splited = []
    for i, f in enumerate(x):
        splited.append(torch.chunk(f, 2))
    l_per = build_perceptual_loss(splited)
    l_style = build_style_loss(splited)
    return l_per, l_style

class FocalLoss(torch.nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=0.25, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, target, logit):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = 2
        # input = logit
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
            # input = input.view(input.size(0), input.size(1), -1)
            # input = input.permute(0, 2, 1).contiguous()
            # input = input.view(-1, input.size(-1))
            # logit = torch.squeeze(logit, 1)
            # logit = logit.view(-1, 1)
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        # 
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')
        
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
def sigmoid_focal_loss(
    targets: torch.Tensor,
    inputs: torch.Tensor,    
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
# class FocalLoss_sk(torch.nn.Module):
#     """
#     copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
#     This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
#     'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
#         Focal_Loss= -1*alpha*(1-pt)*log(pt)
#     :param num_class:
#     :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
#     :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
#                     focus on hard misclassified example
#     :param smooth: (float,double) smooth value when cross entropy
#     :param balance_index: (int) balance class index, should be specific when alpha is float
#     :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
#     """

#     def __init__(self, apply_nonlin=None, alpha=0.25, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
#         super(FocalLoss_sk, self).__init__()
#         self.apply_nonlin = apply_nonlin
#         self.alpha = alpha
#         self.gamma = gamma
#         self.balance_index = balance_index
#         self.smooth = smooth
#         self.size_average = size_average

#         if self.smooth is not None:
#             if self.smooth < 0 or self.smooth > 1.0:
#                 raise ValueError('smooth value should be in [0,1]')

#     def forward(self, target, logit):
#         if self.apply_nonlin is not None:
#             logit = self.apply_nonlin(logit)
#         num_class = 2

#         if logit.dim() > 2:
#             # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
#             logit = logit.view(logit.size(0), logit.size(1), -1)
#             logit = logit.permute(0, 2, 1).contiguous()
#             logit = logit.view(-1, logit.size(-1))
#         target = torch.squeeze(target, 1)
#         target = target.view(-1, 1)
#         # print(logit.shape, target.shape)
#         # 
#         alpha = self.alpha

#         if alpha is None:
#             alpha = torch.ones(num_class, 1)
#         elif isinstance(alpha, (list, np.ndarray)):
#             assert len(alpha) == num_class
#             alpha = torch.FloatTensor(alpha).view(num_class, 1)
#             alpha = alpha / alpha.sum()
#         elif isinstance(alpha, float):
#             alpha = torch.ones(num_class, 1)
#             alpha = alpha * (1 - self.alpha)
#             alpha[self.balance_index] = self.alpha

#         else:
#             raise TypeError('Not support alpha type')
        
#         if alpha.device != logit.device:
#             alpha = alpha.to(logit.device)

#         idx = target.cpu().long()

#         one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
#         one_hot_key = one_hot_key.scatter_(1, idx, 1)
#         if one_hot_key.device != logit.device:
#             one_hot_key = one_hot_key.to(logit.device)

#         if self.smooth:
#             one_hot_key = torch.clamp(
#                 one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
#         pt = (one_hot_key * logit).sum(1) + self.smooth
#         logpt = pt.log()

#         gamma = self.gamma

#         alpha = alpha[idx]
#         alpha = torch.squeeze(alpha)
#         loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

#         if self.size_average:
#             loss = loss.mean()
#         else:
#             loss = loss.sum()
#         return loss

# class FocalLoss_sk(torch.nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss_sk, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self,target,input):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)
#         target= (target>0.5).long()

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()
def build_focal_loss(target_m, predict_m):
    
    loss = sigmoid_focal_loss(target_m, predict_m)
    return loss

def build_focal_sk_loss(mask_t,o_sk):
    
    loss_sk = sigmoid_focal_loss(mask_t,inputs =o_sk)
    # loss_m = loss_sk_fn(o_m, mask_s)
    return loss_sk

def build_gan_loss(x_pred):
    
    loss_adv = adv_loss(x_pred, 1)
        
    return loss_adv



def build_WBCE_loss(x_t, x_o):
       
    iflat = x_o.view(-1)
    tflat = x_t.view(-1)
    pos_weights = (tflat==1.).sum()
    neg_weights = (tflat==0.).sum()
    pos_weight = neg_weights/(pos_weights+1e-5)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return torch.mean(criterion(iflat,tflat))

def gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.cuda()
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1], device=x.device))

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=3, device=torch.device('cpu')):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device)
        
    def forward(self, input, target):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

def build_FEN_loss(o_m, labels):

    labels = (labels>0.5).float()
    # Dice Loss
    l_m_sk = cfg.l_dice_alpha * build_dice_loss(labels, o_m)
    # BCE Loss
    l_m_BCE = cfg.l_wbec_alpha * build_WBCE_loss(labels, o_m)
    # Lap Loss
    Lap_Loss = LapLoss(max_levels=3, channels=3)

    l_m_lap = cfg.l_lap_alpha * Lap_Loss(o_m,labels)

    # l = l_m_sk + l_m_BCE +l_m_FOC+l_text_angle+l_char_angle+l_char_dist+l_char_aspt
    l = l_m_sk + l_m_BCE +l_m_lap

    # return l, [l_m_sk, l_m_BCE,l_m_FOC,l_text_angle,l_char_angle,l_char_dist,l_char_aspt]
    return l, [l_m_sk, l_m_BCE,l_m_lap]



def build_ISN_loss(o_sk, mask_t):

    mask_t = (mask_t>0.5).float()
         
    #Dice loss
    l_t_sk = cfg.l_dice_alpha * build_dice_loss(mask_t, o_sk)
    #WBCE Loss
    l_t_sk_BCE = cfg.l_wbec_alpha * build_WBCE_loss(mask_t, o_sk)
    #lap_Loss
    Lap_Loss = LapLoss(max_levels=3, channels=3)
    l_t_lap = cfg.l_lap_alpha * Lap_Loss(o_sk,mask_t)

    l =  l_t_sk  + l_t_sk_BCE + l_t_lap

    return l, [l_t_sk,l_t_sk_BCE,l_t_lap]

def build_BIN_loss(o_b, o_db_pred, t_b,out_bg_vgg):
           
    #Background Inpainting module loss

    l_b_gan = build_gan_loss(o_db_pred)
    l_b_l1 = cfg.lb_beta * build_l1_loss(t_b, o_b)
    l_b = l_b_gan + l_b_l1
    # BAckgroung VGG
    l_bg_vgg_per, l_bg_vgg_style = build_vgg_loss(out_bg_vgg)
    

    l = l_b + l_bg_vgg_per + l_bg_vgg_style
    # return l, [ l_m, l_b_gan, l_b_l1, l_b]
    return l, [l_b_l1, l_b_gan, l_bg_vgg_per, l_bg_vgg_style]

def build_FN_loss(o_f, t_t, o_fn_vgg,o_df_pred):

    
    #Background Inpainting module loss

    l_fn_gan = build_gan_loss(o_df_pred)
    l_fn_l1 = cfg.lfn_alpha * build_l1_loss(t_t, o_f)
    l_fn_vgg_per, l_fn_vgg_style = build_vgg_loss(o_fn_vgg)
    

    l = l_fn_l1 + l_fn_vgg_per + l_fn_vgg_style + l_fn_gan

    return l, [l_fn_l1, l_fn_vgg_per, l_fn_vgg_style, l_fn_gan]

def build_recognizer_loss(preds, target):
    loss = F.cross_entropy(preds, target, ignore_index=0)
    return loss

def build_FN_loss_rec(out_g, labels, out_fn_vgg, o_df_pred):
    if cfg.with_recognizer:
        o_f, rec_preds = out_g
        t_b, t_f, mask_t, mask_s, rec_target = labels
    else:
        o_f = out_g
        t_b, t_f, mask_t, mask_s = labels
    # o_db_pred, o_df_pred = out_d
    # o_vgg = out_vgg
    l_fn_gan = build_gan_loss(o_df_pred)
    l_fn_l1 = cfg.lfn_alpha * build_l1_loss(t_f, o_f)
    l_fn_vgg_per, l_fn_vgg_style = build_vgg_loss(out_fn_vgg)
    if cfg.with_recognizer:
        l_f_rec = cfg.lf_rec * build_recognizer_loss(rec_preds.view(-1, rec_preds.shape[-1]), rec_target.contiguous().view(-1))
        l = l_fn_l1 + l_fn_vgg_per + l_fn_vgg_style + l_fn_gan + l_f_rec
    else:
        l = l_fn_l1 + l_fn_vgg_per + l_fn_vgg_style + l_fn_gan
    # l = cfg.lb * l_b + cfg.lf * l_f
    

    # l = l_fn_l1 + l_fn_vgg_per + l_fn_vgg_style + l_fn_gan
    if cfg.with_recognizer:
        return l, [l_fn_l1, l_fn_vgg_per, l_fn_vgg_style, l_fn_gan, l_f_rec] 
    else:
        return l, [l_fn_l1, l_fn_vgg_per, l_fn_vgg_style, l_fn_gan]

    

def build_FN_loss_ana(o_f, t_t, o_fn_vgg,o_df_pred):

    
    #Background Inpainting module loss

    l_fn_gan = build_gan_loss(o_df_pred)
    l_fn_l1 = cfg.lfn_alpha * build_l1_loss(t_t, o_f)
    l_fn_vgg_per, l_fn_vgg_style = build_vgg_loss(o_fn_vgg)
    

    l = l_fn_l1 + l_fn_vgg_per + l_fn_vgg_style + l_fn_gan

    return l, [l_fn_l1, l_fn_vgg_per, l_fn_vgg_style, l_fn_gan], l_fn_l1

def build_SPG_loss(o_f, t_t, o_fn_vgg,o_df_pred):

    
    #Background Inpainting module loss

    l_fn_gan = build_gan_loss(o_df_pred)
    l_fn_l1 = cfg.lfn_alpha * build_l1_loss(t_t, o_f)
    l_fn_vgg_per, l_fn_vgg_style = build_vgg_loss(o_fn_vgg)
    

    l = l_fn_l1*10 + l_fn_vgg_per + l_fn_vgg_style + l_fn_gan

    return l, [l_fn_l1, l_fn_vgg_per, l_fn_vgg_style, l_fn_gan]

def build_SPG_synth_loss(generate, groundtruth, vgg):

    [o_m_s,o_b_s,o_sk_s,o_f_s] = generate
    [mask_s_s,t_b_s,mask_t_s,t_f_s] = groundtruth
    [out_fn_vgg_s,out_bg_vgg_s] = vgg

    
    #Background Inpainting module loss

    l_fn_l1 = cfg.lb_beta * build_l1_loss(t_f_s, o_f_s)
    l_bg_l1 = cfg.lb_beta * build_l1_loss(t_b_s, o_b_s)

    l_fn = l_fn_l1+l_bg_l1

    l_fn_vgg_per, l_fn_vgg_style = build_vgg_loss(out_fn_vgg_s)
    l_bg_vgg_per, l_bg_vgg_style = build_vgg_loss(out_bg_vgg_s)

    l_per   = l_fn_vgg_per + l_bg_vgg_per
    l_sytle = l_fn_vgg_style + l_bg_vgg_style
       
    # mask_t

    l_t_sk_FOC = 1 * build_focal_sk_loss(mask_t_s, o_sk_s)
    l_t_sk_BCE = cfg.l_mask_s_beta * build_WBCE_loss(mask_t_s, o_sk_s)
    l_t_sk = cfg.lt_alpha * build_dice_loss(mask_t_s, o_sk_s)

    l_mask_t = l_t_sk_FOC + l_t_sk_BCE + l_t_sk

    # mask_s
    
    l_m_FOC = 1 * build_focal_sk_loss(mask_s_s, o_m_s)
    l_m_BCE = cfg.l_mask_s_beta * build_WBCE_loss(mask_s_s, o_m_s)
    l_m = cfg.lt_alpha * build_dice_loss(mask_s_s, o_m_s)

    l_mask_s = l_m_FOC + l_m_BCE + l_m
    

    l = l_fn*10 + l_per + l_sytle +  l_mask_t + l_mask_s

    return l, [l_fn, l_per, l_sytle, l_mask_t, l_mask_s]

def build_SPG_cycle_loss(predict_list, gt_list, vgg_list, Discriminator_list):

    o_f, i_s_r = predict_list
    o_d1f_pred, o_d2f_pred = Discriminator_list
    out_fn_vgg1, out_fn_vgg2 = vgg_list
    t_f, i_s = gt_list
    #Background Inpainting module loss

    # l_fn_gan = build_gan_loss(o_d1f_pred)
    l_fn_l1 = cfg.lb_beta * build_l1_loss(t_f, o_f)
    l_fn_vgg_per, l_fn_vgg_style = build_vgg_loss(out_fn_vgg1)
    # l_fn_gan2 = build_gan_loss(o_d2f_pred)
    l_fn_l12 = cfg.lb_beta * build_l1_loss(i_s, i_s_r)
    l_fn_vgg_per2, l_fn_vgg_style2 = build_vgg_loss(out_fn_vgg2)
    

    # l = l_fn_l1*10 + l_fn_vgg_per + l_fn_vgg_style + l_fn_gan + l_fn_gan2 + l_fn_l12*10 + l_fn_vgg_per2 + l_fn_vgg_style2
    l = l_fn_l1*10 + l_fn_vgg_per + l_fn_vgg_style + l_fn_l12*10 + l_fn_vgg_per2 + l_fn_vgg_style2

    # return l, [l_fn_l1, l_fn_vgg_per, l_fn_vgg_style, l_fn_gan, l_fn_gan2, l_fn_l12, l_fn_vgg_per2, l_fn_vgg_style2]
    return l, [l_fn_l1, l_fn_vgg_per, l_fn_vgg_style,l_fn_l12, l_fn_vgg_per2, l_fn_vgg_style2]
