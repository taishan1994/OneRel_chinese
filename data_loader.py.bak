from torch.utils.data import DataLoader, Dataset
import json
import os
import torch
from utils import get_tokenizer
from utils.tokenization import BasicTokenizer
import numpy as np
from random import choice
from transformers import BertTokenizer

tokenizer = get_tokenizer('pre_trained_bert/vocab.txt')
# tokenizer = BertTokenizer.from_pretrained('pre_trained_bert/vocab.txt')
basicTokenizer = BasicTokenizer(do_lower_case=False)
tag_file = 'data/tag.txt'

def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

class REDataset(Dataset):
    def __init__(self, config, prefix, is_test, tokenizer):
        self.config = config
        self.prefix = prefix
        self.is_test = is_test
        self.tokenizer = tokenizer

        if self.config.debug:
            self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))[:12]
        else:
            self.json_data = json.load(open(os.path.join(self.config.data_path, prefix + '.json')))
        self.rel2id = json.load(open(os.path.join(self.config.data_path, 'rel2id.json')))[1]
        self.tag2id = json.load(open('data/tag2id.json'))[1]

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        ins_json_data = self.json_data[idx]
        text = ins_json_data['text']
        # text = ' '.join(text.split()[:self.config.max_len])
        # text = basicTokenizer.tokenize(text)
        # text = " ".join(text[:self.config.max_len])
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.config.bert_max_len:
            tokens = tokens[: self.config.bert_max_len]
        text_len = len(tokens)

        if not self.is_test:
            s2ro_map = {}
            for triple in ins_json_data['triple_list']:
                triple = (self.tokenizer.tokenize(triple[0])[1:-1],
                         triple[1], self.tokenizer.tokenize(triple[2])[1:-1])
                # print(tokens, triple[0])
                # print(tokens, triple[2])
                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    if sub not in s2ro_map:
                        s2ro_map[sub] = []
                    s2ro_map[sub].append((obj_head_idx, obj_head_idx + len(triple[2]) - 1, self.rel2id[triple[1]]))
            if s2ro_map:
                token_ids, segment_ids = self.tokenizer.encode(first=text)
                masks = segment_ids
                if len(token_ids) > text_len:
                    token_ids = token_ids[:text_len]
                    masks = masks[:text_len]
                mask_length = len(masks)
                token_ids = np.array(token_ids)
                masks = np.array(masks) + 1
                loss_masks = np.ones((mask_length, mask_length))
                triple_matrix = np.zeros((self.config.rel_num, text_len, text_len))
                for s in s2ro_map:
                    sub_head = s[0]
                    sub_tail = s[1]
                    for ro in s2ro_map.get((sub_head, sub_tail), []):
                        obj_head, obj_tail, relation = ro
                        triple_matrix[relation][sub_head][obj_head] = self.tag2id['HB-TB']
                        triple_matrix[relation][sub_head][obj_tail] = self.tag2id['HB-TE']
                        triple_matrix[relation][sub_tail][obj_tail] = self.tag2id['HE-TE']
                
                return token_ids, masks, loss_masks, text_len, triple_matrix, ins_json_data['triple_list'], tokens
            else:
                print(ins_json_data)
                return None

        else:
            token_ids, masks = self.tokenizer.encode(first=text)
            if len(token_ids) > text_len:
                token_ids = token_ids[:text_len]
                masks = masks[:text_len]
            token_ids = np.array(token_ids)
            masks = np.array(masks) + 1
            mask_length = len(masks)
            loss_masks = np.array(masks) + 1
            triple_matrix = np.zeros((self.config.rel_num, text_len, text_len))
            return token_ids, masks, loss_masks, text_len, triple_matrix, ins_json_data['triple_list'], tokens

def re_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    batch.sort(key=lambda x: x[3], reverse=True)
    
    token_ids, masks, loss_masks, text_len, triple_matrix, triples, tokens = zip(*batch)
    cur_batch_len = len(batch)
    max_text_len = max(text_len)
    batch_token_ids = torch.LongTensor(cur_batch_len, max_text_len).zero_()
    batch_masks = torch.LongTensor(cur_batch_len, max_text_len).zero_()
    batch_loss_masks = torch.LongTensor(cur_batch_len, 1, max_text_len, max_text_len).zero_()
    # if use WebNLG_star, modify 24 to 171
    # if use duie, modify 24 to 48
    # batch_triple_matrix = torch.LongTensor(cur_batch_len, 24, max_text_len, max_text_len).zero_()
    batch_triple_matrix = torch.LongTensor(cur_batch_len, 48, max_text_len, max_text_len).zero_()

    for i in range(cur_batch_len):
        batch_token_ids[i, :text_len[i]].copy_(torch.from_numpy(token_ids[i]))
        batch_masks[i, :text_len[i]].copy_(torch.from_numpy(masks[i]))
        batch_loss_masks[i, 0, :text_len[i], :text_len[i]].copy_(torch.from_numpy(loss_masks[i]))
        batch_triple_matrix[i, :, :text_len[i], :text_len[i]].copy_(torch.from_numpy(triple_matrix[i]))

    return {'token_ids': batch_token_ids,
            'mask': batch_masks,
            'loss_mask': batch_loss_masks,
            'triple_matrix': batch_triple_matrix,
            'triples': triples,
            'tokens': tokens}

def get_loader(config, prefix, is_test=False, num_workers=0, collate_fn=re_collate_fn):
    dataset = REDataset(config, prefix, is_test, tokenizer)
    if not is_test:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    else:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    return data_loader

class DataPreFetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            for k, v in self.next_data.items():
                if isinstance(v, torch.Tensor):
                    self.next_data[k] = self.next_data[k].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

