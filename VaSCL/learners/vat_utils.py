import time
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def _l2_normalize(d, attention_mask=None):
    if attention_mask != None:
        attention_mask = attention_mask.unsqueeze(-1)
        d *= attention_mask
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
#     print("_l2_normalize, BEFORE:{} \t AFTER:{}".format(d.size(), d_reshaped.size()))
    return d

def _emb_norm(emb):
    e_reshaped = emb.view(emb.shape[0], -1, *(1 for _ in range(emb.dim() - 2)))
    enorm = torch.norm(e_reshaped, dim=1, keepdim=False) + 1e-8
#     print("BEFORE:{} \t AFTER:{}".format(emb.size(), e_reshaped.size()))
#     print("enorm:{}, {}".format(enorm.size(), enorm[:10]))
    return enorm

    
class VaSCL_Pturb(nn.Module):
    def __init__(self, xi=0.1, eps=1, ip=1, uni_criterion=None, bi_criterion=None):
        """VaSCL_Pturb on Transformer embeddings
            :param xi: hyperparameter of VaSCL_Pturb (default: 10.0)
            :param eps: hyperparameter of VaSCL_Pturb (default: 1.0)
            :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VaSCL_Pturb, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.delta = 1e-08
        
        self.uni_criterion = uni_criterion
        self.bi_criterion = bi_criterion
        print("\n VaSCL_Pturb on embeddings, xi:{}, eps:{} \n".format(xi, eps))

    def forward(self, model, inputs, hard_indices):
#         print(inputs.size(), "\n", _emb_norm(inputs)[:5])
        with torch.no_grad():
            cnst = model.module.contrast_logits(inputs)

        # prepare random unit tensor
        d = torch.rand(inputs.shape).sub(0.5).to(inputs.device)
        d = _l2_normalize(d)
    
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                cnst_hat = model.module.contrast_logits(inputs+self.xi*d)
                
                adv_cnst = self.uni_criterion(cnst, cnst_hat, hard_indices)
                adv_distance = adv_cnst['lds_loss']

                adv_distance.backward(retain_graph=True)
                d_grad = d.grad.clone().detach()
                d = (d + _l2_normalize(d.grad) * 1e-2).detach() # adv_lr
                model.zero_grad()

        cnst_hat = model.module.contrast_logits(inputs+self.eps*d)
        adv_cnst = self.bi_criterion(cnst, cnst_hat, hard_indices)
        return adv_cnst

class FreeLB(object):

    def __init__(self, adv_K, adv_lr, adv_init_mag, adv_max_norm=0., adv_norm_type='l2', base_model='bert'):
        self.adv_K = adv_K
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_init_mag = adv_init_mag
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model

    def attack(self, model, inputs, gradient_accumulation_steps=1):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        input_ids_1, _ = torch.unbind(input_ids, dim=1)
        attention_mask_1, _ = torch.unbind(attention_mask, dim=1)
        inputs = {'input_ids':input_ids_1, 'attention_mask':attention_mask_1}
        
        if isinstance(model, torch.nn.DataParallel):
            embeds_init = getattr(model.module, self.base_model).embeddings.word_embeddings(input_ids)
        else:
            embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)
            # embeds_init = model.embeddings.word_embeddings(input_ids)
        if self.adv_init_mag > 0:
            input_mask = inputs['attention_mask'].to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            if self.adv_norm_type == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.adv_norm_type == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-self.adv_init_mag, self.adv_init_mag)
                delta = delta * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embeds_init)

        for astep in range(self.adv_K):
            delta.requires_grad_()
            inputs['inputs_embeds'] = delta + embeds_init
            inputs['input_ids'] = None
            outputs = model(**inputs)
            
            loss, logits = outputs[:2]  # model outputs are always tuple in transformers (see doc)
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss = loss / gradient_accumulation_steps
            loss.backward()
            delta_grad = delta.grad.clone().detach()
            if self.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
                    reweights = (self.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif self.adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta = torch.clamp(delta, -self.adv_max_norm, self.adv_max_norm).detach()
            else:
                raise ValueError("Norm type {} not specified.".format(self.adv_norm_type))
            if isinstance(model, torch.nn.DataParallel):
                embeds_init = getattr(model.module, self.base_model).embeddings.word_embeddings(input_ids)
            else:
                embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)
                # embeds_init = model.embeddings.word_embeddings(input_ids)
        return loss, logits 
    
    
    

    

