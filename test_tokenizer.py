from torch.utils.data import DataLoader, Dataset
import json
import os
import torch
from utils import get_tokenizer
from utils.tokenization import BasicTokenizer
import numpy as np
from random import choice
from transformers import BertTokenizer

tokenizer = get_tokenizer('pre_trained_bert//chinese-bert-wwm-ext/vocab.txt')
basicTokenizer = BasicTokenizer(do_lower_case=False)

from transformers import BertTokenizer


class ZhTokenizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('pre_trained_bert/chinese-bert-wwm-ext/vocab.txt')
        self.vocab2id = self.tokenizer.vocab

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        return_tokens = ["[CLS]"]
        for token in tokens:
            return_tokens.append(token)
            return_tokens.append("[unused1]")
        return_tokens += ["[SEP]"]
        return return_tokens

    def encode(self, text):
        return_tokens = self.tokenize(text)
        input_ids = [int(self.vocab2id.get(token, 100)) for token in return_tokens]
        attention_mask = [1] * len(input_ids)
        return input_ids, attention_mask


tokenizer2 = ZhTokenizer()

text = {'text': '392013年剧场第3《陆贞传奇》全国网收视率（含央视）', 'triple_list': [['陆贞传奇', '上映时间', '2013年']]}
text = {'text': '大连市117中学学校系金州区教育局下属的一所公办普通中学，位于金州区东山路，占地面积24478㎡2,其中教学楼面积7519㎡',
        'triple_list': [['大连市117中学', '占地面积', '24478㎡']]}
text = {'text': '随便更一下最近喜欢的歌手，没找到组织家入レオ - Shine5月16日发行的第二首原创单曲，作为日剧青蛙公主的片尾曲附中日字幕的MV',
        'triple_list': [['青蛙公主', '主题曲', 'Shine']]}
print(tokenizer.tokenize(text["text"]))
print(tokenizer.tokenize(text["triple_list"][0][0]))
print(tokenizer.tokenize(text["triple_list"][0][2]))

print("=" * 100)
print(tokenizer2.tokenize(text["text"]))
print(tokenizer2.tokenize(text["triple_list"][0][0]))
print(tokenizer2.tokenize(text["triple_list"][0][2]))