import os
import sys
import numpy as np

import torch
import torch.nn as nn
from learners.contrastive_utils import ContrastiveLoss, VaSCL_NUniDir, VaSCL_NBiDir
from learners.vat_utils import VaSCL_Pturb, FreeLB 
from utils.utils import statistics_log

# Set PATHs
PATH_SENTEVAL = '../DownstreamEval/SentEval'
PATH_TO_DATA = '../DownstreamEval/SentEval/data'
sys.path.insert(0, PATH_SENTEVAL)
import senteval

class VaSCL_Trainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader, args):
        super(VaSCL_Trainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader

        self.args = args
        self.temperature = self.args.temperature
        self.eps = self.args.eps
        self.xi = self.args.xi 
        self.advk = self.args.advk 
        self.topk = self.args.topk

        self.adv_lr = self.args.adv_lr
        self.adv_max_norm = self.args.adv_max_norm
        self.adv_init_mag = self.args.adv_init_mag
        self.adv_norm_type = self.args.adv_norm_type
        self.freelb = self.args.freelb 
        self.paircon_loss = ContrastiveLoss(temperature=self.temperature, topk=self.topk).cuda()
        
        self.uni_criterion = VaSCL_NUniDir(temperature=self.temperature).cuda()
        self.bi_criterion = VaSCL_NBiDir(temperature=self.temperature).cuda()
        self.perturb_embd = VaSCL_Pturb(xi=self.xi, eps=self.eps, ip=self.advk, uni_criterion=self.uni_criterion, bi_criterion=self.bi_criterion).cuda()

        self.gstep = 0
        self.dev_objective = -np.inf
        print(f"\nInitializing VaSCL_Trainer \n")
        

    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text, 
            max_length=self.args.max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat
        

    def prepare_pairwise_input(self, batch):
        text1, text2 = batch['text1'], batch['text2']
        feat1 = self.get_batch_token(text1)
        feat2 = self.get_batch_token(text2)
        
        input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
        attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
        return input_ids.cuda(), attention_mask.cuda()


    def save_model(self, epoch, best_dev=False):
        if best_dev:
            # save the ckpt according to the performance on sts-b dev set
            self.model.module.save_pretrained(os.path.join(self.args.resPath, 'vascl_stsdev'))
            self.tokenizer.save_pretrained(os.path.join(self.args.resPath, 'vascl_stsdev'))
        else:
            # save the ckpt per epoch
            self.model.module.save_pretrained(os.path.join(self.args.resPath, f'vascl_epoch_{epoch}'))
            self.tokenizer.save_pretrained(os.path.join(self.args.resPath, f'vascl_epoch_{epoch}'))
      

    def train(self):

        all_iter = self.args.epochs * len(self.train_loader)
        print('\n={}/{}=Iterations/Batches'.format(all_iter, len(self.train_loader)))

        self.model.train()
        for epoch in range(self.args.epochs):
            for j, batch in enumerate(self.train_loader):

                input_ids, attention_mask = self.prepare_pairwise_input(batch)
                

                if self.freelb:
                    losses = self.freelb_train_step(input_ids, attention_mask)
                else:
                    losses = self.train_step(input_ids, attention_mask)
                
                if (self.gstep%self.args.logging_step==0) or (self.gstep==all_iter) or (self.gstep==self.args.max_iter):

                    if self.args.devset == "sts-b":
                        self.model.eval()
                        sts_metrics = self.eval_stsdev()
                        losses.update(sts_metrics)
                        
                        if sts_metrics["eval_stsb_spearman"] > self.dev_objective:
                            self.save_model(epoch, best_dev=True)
                            self.dev_objective = sts_metrics["eval_stsb_spearman"]

                        self.model.train()

                    statistics_log(self.args.tensorboard, losses=losses, global_step=self.gstep)
                        
                elif self.gstep > self.args.max_iter:
                    break
                    
                self.gstep += 1
            self.save_model(epoch, best_dev=False)
        return None
        

    def train_step(self, input_ids, attention_mask):

        embeddings, hard_indices, feat1, feat2 = self.model(input_ids, attention_mask, topk=self.topk)
        losses = self.paircon_loss(feat1, feat2)
        loss = losses["loss"]
        losses['vcl_loss'] = loss.item()
        
        if self.eps > 0:
            lds_losses = self.perturb_embd(self.model, embeddings.detach(), hard_indices)
            losses.update(lds_losses)
            loss += lds_losses["lds_loss"]
            losses['optimized_loss'] = loss
            
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses

    def freelb_train_step(self, input_ids, attention_mask):

        embeddings, _, feat1, feat2 = self.model(input_ids, attention_mask, topk=self.topk)
        losses = self.paircon_loss(feat1, feat2)
        loss = losses['loss']
        losses['vcl_loss'] = loss.item()

        input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
        attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1) 

        embeds_init_1 = self.model.module.get_init_embeddings(input_ids_1)
        embeds_init_2 = self.model.module.get_init_embeddings(input_ids_2)
        if self.adv_init_mag > 0:
            input_mask = attention_mask_1.to(embeds_init_1)
            input_lengths = torch.sum(input_mask, 1)
            if self.adv_norm_type == "l2":
                delta = torch.zeros_like(embeds_init_1).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init_1.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.adv_norm_type == "linf":
                delta = torch.zeros_like(embeds_init_1).uniform_(-self.adv_init_mag, self.adv_init_mag)
                delta = delta * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embeds_init_1)
        
        for astep in range(self.advk):
            delta.requires_grad_()
            inputs_embeds_1 = delta + embeds_init_1
            inputs_embeds_2 = embeds_init_2
            feat1, feat2 = self.model((inputs_embeds_1,inputs_embeds_2),(attention_mask_1, attention_mask_2),freelb=True)
            losses = self.paircon_loss(feat1, feat2)
            loss = losses['loss']
            loss = loss.mean()
            losses['adv_loss'] = loss.item()
            loss.backward()
            delta_grad = delta.grad.clone().detach()
            if self.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init_1)
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
            embeds_init_1 = self.model.module.get_init_embeddings(input_ids_1)
            embeds_init_2 = self.model.module.get_init_embeddings(input_ids_2)

        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses

    def eval_stsdev(self):
    
        def prepare(params, samples):
            return 

        def batcher(params, batch):
            sentences = [' '.join(s) for s in batch]

            features = self.tokenizer.batch_encode_plus(
                sentences, 
                max_length=params['max_length'], 
                return_tensors='pt', 
                padding=True, 
                truncation=True
            )
            input_ids = features['input_ids'].to(params['device']) 
            attention_mask = features['attention_mask'].to(params['device'])

            with torch.no_grad():
                embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask, task_type="evaluate")
            return embeddings.detach().cpu().numpy()

        # define senteval params
        params_senteval = {'task_path': self.args.path_sts_data, 'usepytorch': True, 'kfold': 5}
        params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 64,
                                         'tenacity': 3, 'epoch_size': 2}
        params_senteval['max_length'] = None
        params_senteval['device'] = torch.device("cuda:{}".format(0))

        se = senteval.engine.SE(params_senteval, batcher, prepare)
        transfer_tasks = ['SICKRelatedness', 'STSBenchmark']
        results = se.eval(transfer_tasks)

        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
        sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]
        metrics = {"eval_stsb_spearman": stsb_spearman, "eval_sickr_spearman": sickr_spearman, "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2} 
        return metrics

