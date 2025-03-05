import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
from torch.cuda.amp import autocast, GradScaler
import math
import numpy as np


class SuMi(nn.Module):
    def __init__(self, model, optimizer, device, args, steps=1, episodic=False, uent_thresh=math.e/10, ment_thresh=math.log(1000)/2-1, umu=1.0, lamda=5.0, e_margin=math.log(1000)/2-1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.args = args
        self.uent_thresh = uent_thresh
        self.ment_thresh = ment_thresh
        self.umu = umu
        self.lamda = lamda
        self.e_margin = e_margin
        self.scaler = GradScaler()
        self.device = device

    def forward(self, x, t, iters, adapt_flag=True):
        for _ in range(self.steps):
            if adapt_flag:
                outputs = forward_and_adapt(x, self.model, self.optimizer, self.args, self.scaler, self.uent_thresh, self.ment_thresh, self.umu, self.lamda, t, iters, self.e_margin)
            else:
                outputs, _ = self.model.module.forward_eval(a=x[0], v=x[1], mode=self.args.testmode)
                outputs = (outputs, outputs)

        return outputs


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, args, scaler, uent_thresh, ment_thresh, mu, lamda, t, iters, e_margin):
    with autocast():
        # forward
        aoutput, a_feat = model.module.forward_eval(a=x[0], v=x[1], mode='audioonly')
        outputs, _ = model.module.forward_eval(a=x[0], v=x[1], mode='multimodal')
        voutput, v_feat = model.module.forward_eval(a=x[0], v=x[1], mode='videoonly')
    # adapt
    a_feat_avg = torch.mean(a_feat, dim=1).detach()
    v_feat_avg = torch.mean(v_feat, dim=1).detach()
    feat = torch.cat([a_feat_avg, v_feat_avg], dim=1)
    feat_max, _ = torch.max(feat, dim=0)
    feat_min, _ = torch.min(feat, dim=0)
    Q1 = feat_min + 0.25 * (feat_max - feat_min)
    Q3 = feat_min + 0.75 * (feat_max - feat_min)
    IQR = Q3 - Q1
    upper = Q3 + 3 * t * IQR / (2 * iters)
    lower = Q1 - 3 * t * IQR / (2 * iters)
    upper = upper.unsqueeze(0).expand_as(feat)
    lower = lower.unsqueeze(0).expand_as(feat)
    mask_lower = feat>=lower
    mask_upper = feat<=upper
    mask = mask_lower & mask_upper
    col_and = mask.sum(dim=-1) >= feat.shape[1] * (0.6 + 0.4 * t / iters)
    smoothing_id = torch.nonzero(col_and).squeeze()

    aentropy = softmax_entropy(x=aoutput).detach()
    ventropy = softmax_entropy(x=voutput).detach()
    entropy = softmax_entropy(x=outputs)
    u_entropy = (aentropy + mu * ventropy) / (1 + mu)
    entropy_id = torch.where((u_entropy >= uent_thresh) & (entropy <= ment_thresh))[0]

    final_id = np.intersect1d(smoothing_id.cpu(), entropy_id.cpu())

    aoutput = aoutput[final_id]
    voutput = voutput[final_id]
    outputs = outputs[final_id]
    entropy = entropy[final_id]

    if len(entropy) > 0:
        p_sum = outputs.softmax(dim=-1).sum(dim=-2) 
        loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()
        coeff = 1 / (torch.exp(entropy.clone().detach() - e_margin))
        loss = entropy.mean(0)
        mutual_loss = F.kl_div(aoutput.log_softmax(1), (voutput.softmax(1)+outputs.softmax(1))/2, reduction='none') + F.kl_div(voutput.log_softmax(1), (aoutput.softmax(1)+outputs.softmax(1))/2, reduction='none')
        mutual_loss = mutual_loss.sum(-1)
        if t < 0.5 * iters:
            loss = entropy + lamda * mutual_loss
        loss = loss.mul(coeff)
        loss = loss.mean(0)
        loss -= loss_bal

        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    with torch.no_grad():
        with autocast():
        # forward
            outputs2, _ = model.module.forward_eval(a=x[0], v=x[1], mode=args.testmode)

    return (outputs, outputs2)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
    return model