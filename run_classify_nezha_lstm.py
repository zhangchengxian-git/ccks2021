import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.backends import cudnn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F


from transformers import (
    BertTokenizer,
    BertTokenizerFast
)

from modeling.modeling_nezha.modeling import NeZhaPreTrainedModel, NeZhaModel

import os 
import json
import time
import pickle
import random
import warnings
import numpy as np
from torch.optim import Optimizer
from collections import defaultdict


DROPOUT_RATE = 0.1
LSTM_LAYERS = 4
NEZHA_HIDDEN_SIZE = 768
LSTM_HIDDEN_SIZE = 768

class focal_loss(nn.Module):    
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)      
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            # print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            # print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算        
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数        
        :param labels:  实际类别. size:[B,N] or [B]        
        :return:
        """        
        # assert preds.dim()==2 and labels.dim()==1        
        preds = preds.view(-1,preds.size(-1))        
        self.alpha = self.alpha.to(preds.device)        
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)        
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )        
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))        
        self.alpha = self.alpha.gather(0,labels.view(-1))        
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())        
        if self.size_average:        
            loss = loss.mean()        
        else:            
            loss = loss.sum()        
        return loss

class LabelSmoothingLoss(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.01):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class NezhaForSequenceClassification_LSTM(NeZhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 3
        self.bert = NeZhaModel(config)
        # hard code for now
        self.dropout = nn.Dropout(DROPOUT_RATE)

        self.lstm = []
        for i in range(LSTM_LAYERS):
            self.lstm.append(nn.LSTM(NEZHA_HIDDEN_SIZE if i==0 else LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE, num_layers=1, bidirectional=True,batch_first=True).cuda())
        self.lstm = nn.ModuleList(self.lstm)
        self.classifier = nn.Linear(LSTM_HIDDEN_SIZE*2, self.num_labels)
        self.init_weights()

    def forward(self, 
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None):
        outputs = self.bert(input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids,
                           head_mask=head_mask,
                           inputs_embeds=inputs_embeds)

        # shape (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs[0]

        for lstm in self.lstm:
            try:
                lstm.flattern_parameters()
            except:
                pass
            #h_n shape: (batch_size, num_layers*num_directions == 2, lstm_hidden_size)
            output, (h_n, c_n) = lstm(last_hidden_state)

        x = h_n.permute(1,0,2).reshape(input_ids.size(0), -1).contiguous()

        x = self.dropout(x)
        logits = self.classifier(x)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
    
        return outputs

class FGM:
    def __init__(self, config, model):
        self.model = model
        self.backup = {}
        self.emb_name = config['emb_name']
        self.epsilon = config['epsilon']

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class PGD:
    def __init__(self, config, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon = config['epsilon']
        self.emb_name = config['emb_name']
        self.alpha = config['alpha']

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]

class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

def build_model_and_tokenizer(config):
    tokenizer_path = config['model_path'] + '/vocab.txt'
    if config['tokenizer_fast']:
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    if config['use_model'] == 'nezha':
        model = NezhaForSequenceClassification_LSTM.from_pretrained(config['model_path'])
    return tokenizer, model


def build_optimizer(config, model, train_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': config['weight_decay']},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    
    optimizer = AdaBelief(optimizer_grouped_parameters, lr=3e-5, eps=1e-16, betas=(0.9,0.999),weight_decay=1.2e-6, weight_decouple = True, rectify = True, fixed_decay=False, amsgrad=False)
    #optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=1e-8)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * config['warmup_ratio'],
                                     t_total=train_steps)

    return optimizer, scheduler


def block_shuffle(config, train_set):
    bs = config['batch_size'] * 100
    num_block = int(len(train_set) / bs)
    slice_ = num_block * bs

    train_set_to_shuffle = train_set[:slice_]
    train_set_left = train_set[slice_:]

    sorted_train_set = sorted(train_set_to_shuffle, key=lambda i: len(i[0]))
    shuffled_train_set = []

    tmp = []
    for i in range(len(sorted_train_set)):
        tmp.append(sorted_train_set[i])
        if (i+1) % bs == 0:
            random.shuffle(tmp)
            shuffled_train_set.extend(tmp)
            tmp = []

    random.shuffle(train_set_left)
    shuffled_train_set.extend(train_set_left)

    return shuffled_train_set

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def batch_loader(config, src, tgt, seg, mask):
    ins_num = src.size()[0]
    batch_size = config['batch_size']
    for i in range(ins_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        mask_batch = mask[i * batch_size: (i + 1) * batch_size, :]
        yield src_batch, tgt_batch, seg_batch, mask_batch
    if ins_num > ins_num // batch_size * batch_size:
        src_batch = src[ins_num // batch_size * batch_size:, :]
        tgt_batch = tgt[ins_num // batch_size * batch_size:]
        seg_batch = seg[ins_num // batch_size * batch_size:, :]
        mask_batch = mask[ins_num // batch_size * batch_size:, :]
        yield src_batch, tgt_batch, seg_batch, mask_batch


def read_dataset(config, tokenizer):
    start = time.time()
    dataset ,dev_set, label_set= [], [], []
    seq_length = config['max_seq_len']

    with open(config['train_path'], 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            if(len(line.strip().split('\t'))==3):
                sent_a, sent_b, tgt = line.strip().split('\t')
                src_a = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(sent_a) + ['[SEP]'])
                src_b = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent_b) + ['[SEP]'])
                src = src_a + src_b
                seg = [0] * len(src_a) + [1] * len(src_b)
                mask = [1] * len(src)
                tgt = int(tgt)
                if len(src) > seq_length:
                    src = src[: seq_length]
                    seg = seg[: seq_length]
                    mask = mask[: seq_length]
                while len(src) < seq_length:
                    src.append(0)
                    seg.append(0)
                    mask.append(0)
                dataset.append((src, tgt, seg, mask))
            elif(len(line.strip().split('\t'))==4):
                sent_a, sent_b, tgt, m = line.strip().split('\t')
                src_a = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(sent_a) + ['[SEP]'])
                src_b = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent_b) + ['[SEP]'])
                src = src_a + src_b
                seg = [0] * len(src_a) + [1] * len(src_b)
                mask = [1] * len(src)
                tgt = int(tgt)
                if len(src) > seq_length:
                    src = src[: seq_length]
                    seg = seg[: seq_length]
                    mask = mask[: seq_length]
                while len(src) < seq_length:
                    src.append(0)
                    seg.append(0)
                    mask.append(0)
                dataset.append((src, tgt, seg, mask))
            else:
                print(line)
    

    with open(config['dev_path'], 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            if len(line.strip().split('\t'))!=2:
                sent_a=''
                sent_b=line.strip().split('\t')[0]
            else:
                sent_a, sent_b = line.strip().split('\t')
            src_a = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(sent_a) + ['[SEP]'])
            src_b = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent_b) + ['[SEP]'])
            src = src_a + src_b
            seg = [0] * len(src_a) + [1] * len(src_b)
            mask = [1] * len(src)
                
            if len(src) > seq_length:
                src = src[: seq_length]
                seg = seg[: seq_length]
                mask = mask[: seq_length]
            while len(src) < seq_length:
                src.append(0)
                seg.append(0)
                mask.append(0)
            dev_set.append((src, seg, mask))
    
    with open(config['label_path'], 'r', encoding='utf-8') as f:
        label_set = f.readlines()

    print("\n>> loading sentences from {},Time cost:{:.2f}".
          format(config['train_path'], ((time.time() - start) / 60.00)))

    return dataset, dev_set, label_set

def argmax(res):
    k, tmp = 0, 0
    for i in range(len(res)):
        if res[i] > tmp:
            tmp = res[i]
            k = i

    return k

def batch_loader_pre(config, src, seg, mask):
    ins_num = src.size()[0]
    batch_size = config['batch_size']
    for i in range(ins_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        mask_batch = mask[i * batch_size: (i + 1) * batch_size, :]
        yield src_batch, seg_batch, mask_batch
    if ins_num > ins_num // batch_size * batch_size:
        src_batch = src[ins_num // batch_size * batch_size:, :]
        seg_batch = seg[ins_num // batch_size * batch_size:, :]
        mask_batch = mask[ins_num // batch_size * batch_size:, :]
        yield src_batch, seg_batch, mask_batch

def predict(dataset, pre_model, config):
    predict_logits, predictions = [], []
    p_logit = []

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])
    mask = torch.LongTensor([sample[2] for sample in dataset])

    for i, (src_batch, seg_batch, mask_batch) in \
            enumerate(batch_loader_pre(config, src, seg, mask)):
        src_batch = src_batch.to(config['device'])
        seg_batch = seg_batch.to(config['device'])
        mask_batch = mask_batch.to(config['device'])
        with torch.no_grad():
            output = pre_model(input_ids=src_batch, token_type_ids=seg_batch, attention_mask=mask_batch)

        logits = output[0]
        logits = torch.softmax(logits, 1)
        p_logits = logits.cpu().numpy().tolist()
        for i in p_logits:
            p_logit.append(i)

    final_logit, predict_data = [], []
    for i in range(len(p_logit)):
        merge = []
        for j in range(len(p_logit[i])):
            tmp = p_logit[i][j]
            merge.append(tmp)
        final_logit.append(merge)
        res = argmax(merge)
        predictions.append(res)

    return predictions

def main():
    config = {
        'use_model': 'nezha',
        'normal_data_cache_path': 'user_data/processed/nezha/all_data.pkl',  # 96423
        'train_path': 'data/train_test0.3.txt',
        'dev_path':'data/data_for_dev_without_label.txt',
        'label_path': 'data/data_for_dev_with_label.txt',
        'output_path': 'output_model/all_data_3e-5',
        'model_path': 'pretrain_code/mlm_model',  # your pretrain model path
        'shuffle_way': 'block_shuffle',
        'use_swa': True,
        'tokenizer_fast': False,
        'batch_size': 128,
        'num_epochs': 5,
        'max_seq_len': 96,
        'learning_rate': 5e-5,
        'alpha': 0.3,
        'epsilon': 1.0,
        'adv_k': 3,
        'emb_name': 'word_embeddings.',
        'adv': 'fgm',
        'warmup_ratio': 0.05,
        'weight_decay': 0.01,
        'device': 'cuda',
        'logging_step': 10,
        'seed': 2021
    }

    warnings.filterwarnings('ignore')
    localtime_start = time.asctime(time.localtime(time.time()))
    print(">> program start at:{}".format(localtime_start))
    print(">> batch size input:{}".format(config['batch_size']))
    #print(">> data_path:{}".format(config['data_path']))
    print("\n>> loading model from :{}".format(config['model_path']))


    tokenizer, model = build_model_and_tokenizer(config)
    '''if not os.path.exists(config['normal_data_cache_path']):'''
    train_set, dev_set, label_set = read_dataset(config, tokenizer)
    '''else:
        with open(config['normal_data_cache_path'], 'rb') as f:
            train_set = pickle.load(f)'''

    seed_everything(config['seed'])

    if config['shuffle_way'] == 'block_shuffle':
        train_set = block_shuffle(config, train_set)
    else:
        random.shuffle(train_set)

    train_num = len(train_set)
    train_steps = int(train_num * config['num_epochs'] / config['batch_size']) + 1

    optimizer, scheduler = build_optimizer(config, model, train_steps)
    model.to(config['device'])

    src = torch.LongTensor([example[0] for example in train_set])
    tgt = torch.LongTensor([example[1] for example in train_set])
    seg = torch.LongTensor([example[2] for example in train_set])
    mask = torch.LongTensor([example[3] for example in train_set])

    cudnn.benchmark = True

    total_loss, cur_avg_loss = 0.0, 0.0
    global_steps = 0

    if config['adv'] == '':
        print('\n>> start normal training ...')
    elif config['adv'] == 'fgm':
        print('\n>> start fgm training ...')
    elif config['adv'] == 'pgd':
        print('\n>> start pgd training ...')

    start = time.time()
    # from tqdm import tqdm
    f = open('log.txt','a',encoding='utf-8')
    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        for i, (src_batch, tgt_batch, seg_batch, mask_batch) \
                in enumerate(batch_loader(config, src, tgt, seg, mask)):

            src_batch = src_batch.to(config['device'])
            tgt_batch = tgt_batch.to(config['device'])
            seg_batch = seg_batch.to(config['device'])
            mask_batch = mask_batch.to(config['device'])

            output = model(input_ids=src_batch, labels=tgt_batch,
                           token_type_ids=seg_batch, attention_mask=mask_batch)
            loss = output[0]
            optimizer.zero_grad()
            loss.backward()

            total_loss += loss.item()
            cur_avg_loss += loss.item()

            if config['adv'] == 'fgm':
                fgm = FGM(config, model)
                fgm.attack()
                adv_loss = model(input_ids=src_batch, labels=tgt_batch,
                                 token_type_ids=seg_batch, attention_mask=mask_batch)[0]
                adv_loss.backward()
                fgm.restore()

            if config['adv'] == 'pgd':
                pgd = PGD(config, model)
                K = config['adv_k']
                pgd.backup_grad()
                for t in range(K):
                    pgd.attack(is_first_attack=(t == 0))
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    adv_loss = model(input_ids=src_batch, labels=tgt_batch,
                                     token_type_ids=seg_batch, attention_mask=mask_batch)[0]
                    adv_loss.backward()
                pgd.restore()
            optimizer.step()

            scheduler.step()
            model.zero_grad()
            import logging 

            if (i + 1) % config['logging_step'] == 0:
                print("\n>> epoch - {}, epoch steps - {}, global steps - {}, "
                      "epoch avg loss - {:.4f}, global avg loss - {:.4f}, time cost - {:.2f} min".format
                      (epoch, i + 1, global_steps + 1, cur_avg_loss / config['logging_step'],
                       total_loss / (global_steps + 1),
                       (time.time() - start) / 60.00))
                logging.info("\n>> epoch - {}, epoch steps - {}, global steps - {}, "
                      "epoch avg loss - {:.4f}, global avg loss - {:.4f}, time cost - {:.2f} min".format
                      (epoch, i + 1, global_steps + 1, cur_avg_loss / config['logging_step'],
                       total_loss / (global_steps + 1),
                       (time.time() - start) / 60.00))
                cur_avg_loss = 0.0
            global_steps += 1

        pre = predict(dev_set, model, config)
        '''f1 = f1_score(label_set, pre, average='macro')
        f.writelines(os.path.join(config['output_path'], f'checkpoint-{global_steps}')+'\t'+str(f1)+'\n')
        print(f1)'''
        model_save_path = os.path.join(config['output_path'], f'checkpoint-{global_steps}')
        model_to_save = model.module if hasattr(model, 'module') else model
        print('\n>> model saved ... ...')
        model_to_save.save_pretrained(model_save_path)

        conf = json.dumps(config)
        out_conf_path = os.path.join(config['output_path'], f'checkpoint-{global_steps}' +
                                     '/train_config.json')
        with open(out_conf_path, 'w', encoding='utf-8') as f:
            f.write(conf)

    localtime_end = time.asctime(time.localtime(time.time()))
    print("\n>> program end at:{}".format(localtime_end))


if __name__ == '__main__':
    main()

        


