import torch.optim as optim
from torch import nn
import os
import data_loader
import torch.nn.functional as F
import torch
import numpy as np
import json
import time

def helper(text):
    text = text.split(' ')
    text = ''.join(text)
    return text 

class Framework(object):
    def __init__(self, config):
        self.config = config
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(self.config.log_dir, self.config.log_save_name), 'a+') as f_log:
                f_log.write(s + '\n')

    def train(self, model_pattern):
        ori_model = model_pattern(self.config)
        ori_model.cuda()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, ori_model.parameters()), lr = self.config.learning_rate)

        # whether use multi gpu:
        if self.config.multi_gpu:
            model = nn.DataParallel(ori_model)
        else:
            model = ori_model

        # define the loss function
        def cal_loss(target, predict, mask):
            loss = self.loss_function(predict, target)
            loss = torch.sum(loss * mask) / torch.sum(mask)
            return loss

        # check the check_point dir
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)

        # check the log dir
        if not os.path.exists(self.config.log_dir):
            os.mkdir(self.config.log_dir)

        # training data
        train_data_loader = data_loader.get_loader(self.config, prefix = self.config.train_prefix, num_workers = 2)
        # dev data
        test_data_loader = data_loader.get_loader(self.config, prefix=self.config.dev_prefix, is_test=True)

        # other
        model.train()
        global_step = 0
        loss_sum = 0

        best_f1_score = 0
        best_precision = 0
        best_recall = 0

        best_epoch = 0
        init_time = time.time()
        start_time = time.time()

        # the training loop
        for epoch in range(self.config.max_epoch):
            train_data_prefetcher = data_loader.DataPreFetcher(train_data_loader)
            data = train_data_prefetcher.next()
            epoch_start_time = time.time()

            while data is not None:

                pred_triple_matrix = model(data)

                triple_loss = cal_loss(data['triple_matrix'], pred_triple_matrix, data['loss_mask'])

                optimizer.zero_grad()
                triple_loss.backward()
                optimizer.step()

                global_step += 1
                loss_sum += triple_loss.item()

                if global_step % self.config.period == 0:
                    cur_loss = loss_sum / self.config.period
                    elapsed = time.time() - start_time
                    self.logging("epoch: {:3d}, step: {:4d}, speed: {:5.2f}ms/b, train loss: {:5.3f}".
                                 format(epoch, global_step, elapsed * 1000 / self.config.period, cur_loss))
                    loss_sum = 0
                    start_time = time.time()

                # 或者每500步进行验证
                if global_step % 500 == 0:
                    eval_start_time = time.time()
                    model.eval()
                    # call the test function
                    precision, recall, f1_score = self.test(test_data_loader, model, current_f1=best_f1_score,
                                                            output=self.config.result_save_name)

                    self.logging('epoch {:3d}, eval time: {:5.2f}s, f1: {:4.3f}, precision: {:4.3f}, recall: {:4.3f}'.
                                 format(epoch, time.time() - eval_start_time, f1_score, precision, recall))

                    if f1_score > best_f1_score:
                        best_f1_score = f1_score
                        best_epoch = epoch
                        best_precision = precision
                        best_recall = recall
                        self.logging(
                            "saving the model, epoch: {:3d}, precision: {:4.3f}, recall: {:4.3f}, best f1: {:4.3f}".
                            format(best_epoch, best_precision, best_recall, best_f1_score))
                        # save the best model
                        path = os.path.join(self.config.checkpoint_dir, self.config.model_save_name)
                        if not self.config.debug:
                            torch.save(ori_model.state_dict(), path)

                    model.train()

                data = train_data_prefetcher.next()
                
            print("total time {}".format(time.time() - epoch_start_time))

            # if epoch > 20 and (epoch + 1) % self.config.test_epoch == 0:
            # if epoch == 0:
            # 每一个epoch后都进行验证
            eval_start_time = time.time()
            model.eval()
            # call the test function
            precision, recall, f1_score = self.test(test_data_loader, model, current_f1=best_f1_score, output=self.config.result_save_name)

            self.logging('epoch {:3d}, eval time: {:5.2f}s, f1: {:4.3f}, precision: {:4.3f}, recall: {:4.3f}'.
                         format(epoch, time.time() - eval_start_time, f1_score, precision, recall))

            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_epoch = epoch
                best_precision = precision
                best_recall = recall
                self.logging("saving the model, epoch: {:3d}, precision: {:4.3f}, recall: {:4.3f}, best f1: {:4.3f}".
                             format(best_epoch, best_precision, best_recall, best_f1_score))
                # save the best model
                path = os.path.join(self.config.checkpoint_dir, self.config.model_save_name)
                if not self.config.debug:
                    torch.save(ori_model.state_dict(), path)

            model.train()

            # manually release the unused cache
            torch.cuda.empty_cache()

        self.logging("finish training")
        self.logging("best epoch: {:3d}, precision: {:4.3f}, recall: {:4.3}, best f1: {:4.3f}, total time: {:5.2f}s".
                     format(best_epoch, best_precision, best_recall, best_f1_score, time.time() - init_time))

    def test(self, test_data_loader, model, current_f1, output=True):
        
        orders = ['subject', 'relation', 'object']

        def to_tup(triple_list):
            ret = []
            for triple in triple_list:
                ret.append(tuple(triple))
            return ret

        test_data_prefetcher = data_loader.DataPreFetcher(test_data_loader)
        data = test_data_prefetcher.next()
        id2rel = json.load(open(os.path.join(self.config.data_path, 'rel2id.json')))[0]
        id2tag, tag2id = json.load(open('data/tag2id.json'))
        correct_num, predict_num, gold_num = 0, 0, 0

        results = []
        test_num = 0

        s_time = time.time()
        while data is not None:

            with torch.no_grad():

                print('\r Testing step {} / {}, Please Waiting!'.format(test_num, test_data_loader.dataset.__len__()), end="")

                token_ids = data['token_ids']
                tokens = data['tokens'][0]
                mask = data['mask']
                # pred_triple_matrix: [1, rel_num, seq_len, seq_len]
                pred_triple_matrix = model(data, train = False).cpu()[0]
                rel_numbers, seq_lens, seq_lens = pred_triple_matrix.shape
                relations, heads, tails = np.where(pred_triple_matrix > 0)

                triple_list = [] 

                pair_numbers = len(relations)

                if pair_numbers > 0:
                    # print('current sentence contains {} triple_pairs'.format(pair_numbers))
                    for i in range(pair_numbers):
                        r_index = relations[i]
                        h_start_index = heads[i]
                        t_start_index = tails[i]
                        # 如果当前第一个标签为HB-TB
                        if pred_triple_matrix[r_index][h_start_index][t_start_index] == tag2id['HB-TB'] and i+1 < pair_numbers:
                            # 如果下一个标签为HB-TE
                            t_end_index = tails[i+1]
                            if pred_triple_matrix[r_index][h_start_index][t_end_index] == tag2id['HB-TE']:
                                # 那么就向下找
                                for h_end_index in range(h_start_index, seq_lens):
                                    # 向下找到了结尾位置
                                    if pred_triple_matrix[r_index][h_end_index][t_end_index] == tag2id['HE-TE']:

                                        sub_head, sub_tail = h_start_index, h_end_index
                                        obj_head, obj_tail = t_start_index, t_end_index
                                        sub = tokens[sub_head : sub_tail+1]
                                        # sub
                                        sub = ''.join([i.lstrip("##") for i in sub])
                                        sub = ' '.join(sub.split('[unused1]')).strip()
                                        obj = tokens[obj_head : obj_tail+1]
                                        # obj
                                        obj = ''.join([i.lstrip("##") for i in obj])
                                        obj = ' '.join(obj.split('[unused1]')).strip()
                                        rel = id2rel[str(int(r_index))]
                                        if len(sub) > 0 and len(obj) > 0:
                                            triple_list.append((sub, rel, obj))
                                        break


                triple_set = set()

                for s, r, o in triple_list:
                    s = helper(s)
                    o = helper(o)
                    triple_set.add((s, r, o))

                pred_list = list(triple_set)

                gold_triple = data['triples'][0]
                                    
                pred_triples = set(pred_list)
                gold_triples = set(to_tup(data['triples'][0]))

                correct_num += len(pred_triples & gold_triples)
                predict_num += len(pred_triples)
                gold_num += len(gold_triples)

                if output:
                    results.append({
                        'text': ' '.join(tokens[1:-1]).replace(' [unused1]', '').replace(' ##', ''),
                        'triple_list_gold': [
                            dict(zip(orders, triple)) for triple in gold_triples
                        ],
                        'triple_list_pred': [
                            dict(zip(orders, triple)) for triple in pred_triples
                        ],
                        'new': [
                            dict(zip(orders, triple)) for triple in pred_triples - gold_triples
                        ],
                        'lack': [
                            dict(zip(orders, triple)) for triple in gold_triples - pred_triples
                        ]
                    })

                data = test_data_prefetcher.next()
            
            test_num += 1

        print('\n' + self.config.model_save_name )
        print("\n correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))

        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)


        if output and f1_score > current_f1:
            if not os.path.exists(self.config.result_dir):
                os.mkdir(self.config.result_dir)

            path = os.path.join(self.config.result_dir, self.config.result_save_name)

            fw = open(path, 'w')

            for line in results:
                fw.write(json.dumps(line, ensure_ascii=False, indent=4) + "\n")
            fw.close()
        return precision, recall, f1_score

    def testall(self, model_pattern, model_name):
        model = model_pattern(self.config)
        path = os.path.join(self.config.checkpoint_dir, model_name)
        model.load_state_dict(torch.load(path))

        model.cuda()
        model.eval()
        test_data_loader = data_loader.get_loader(self.config, prefix=self.config.test_prefix, is_test=True)
        precision, recall, f1_score = self.test(test_data_loader, model, current_f1=0, output=True)
        print("f1: {:4.4f}, precision: {:4.4f}, recall: {:4.4f}".format(f1_score, precision, recall))

