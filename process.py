import json

from utils.tokenization import BasicTokenizer
from transformers import BertTokenizer

basicTokenizer = BasicTokenizer(do_lower_case=False)
bertTokenizer = BertTokenizer.from_pretrained('pre_trained_bert/')


def process(in_path, out_path):
    res = []
    with open(in_path, 'r') as fp:
        data = fp.readlines()
        total = len(data)
        for i,d in enumerate(data):
            print(i+1, total)
            d = eval(d)
            tmp = {}
            text = d['text']
            tmp_text = basicTokenizer.tokenize(text)
            if len(tmp_text) > 100:
                continue
            spo_list = d['spo_list']
            tmp['text'] = text
            tmp['triple_list'] = []
            for spo in spo_list:
                rel = spo['predicate']
                subject = spo['subject']
                object = spo['object']['@value']
                if subject in text[:100] and object in text[:100]:
                    tmp['triple_list'].append([subject, rel, object])
            if tmp['triple_list']:
                res.append(tmp)
    with open(out_path, 'w') as fp:
        json.dump(res, fp, ensure_ascii=False)

def get_rels(out_path):
    train_path = 'data/DUIE/duie_train.json'
    dev_path = 'data/DUIE/duie_dev.json'

    def get_rel(path):
        res = []
        with open(path, 'r') as fp:
            data = fp.readlines()
            for d in data:
                d = eval(d)
                spo_list = d['spo_list']
                for spo in spo_list:
                    rel = spo['predicate']
                    if rel not in res:
                        res.append(rel)
        return res
    rels = get_rel(train_path) + get_rel(dev_path)
    rels = list(set(rels))
    rel2id = {}
    id2rel = {}
    for i,rel in enumerate(rels):
        rel2id[rel] = i
        id2rel[str(i)] = rel
    with open(out_path, 'w') as fp:
        json.dump([id2rel, rel2id], fp, ensure_ascii=False)

if __name__ == '__main__':
    process('data/DUIE/duie_train.json', 'data/DUIE/train_triples.json')
    process('data/DUIE/duie_dev.json', 'data/DUIE/dev_triples.json')
    process('data/DUIE/duie_dev.json', 'data/DUIE/test_triples.json')
    # get_rels('./rel2id.json')
