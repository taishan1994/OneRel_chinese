class Config(object):
    def __init__(self, args):
        self.args = args

        self.multi_gpu = args.multi_gpu
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.max_len = args.max_len
        self.rel_num = args.rel_num
        self.bert_max_len = args.bert_max_len
        self.bert_dim = 768
        self.tag_size = 4
        self.dropout_prob = args.dropout_prob
        self.entity_pair_dropout = args.entity_pair_dropout

        # dataset
        self.dataset = args.dataset

        # path and name
        self.data_path = './data/' + self.dataset
        self.checkpoint_dir = './checkpoint/' + self.dataset
        self.log_dir = './log/' + self.dataset
        self.result_dir = './result/' + self.dataset
        self.train_prefix = args.train_prefix
        self.dev_prefix = args.dev_prefix
        self.test_prefix = args.test_prefix


        self.model_save_name = args.model_name + '_DATASET_' + self.dataset + "_LR_" + str(self.learning_rate) + "_BS_" + str(self.batch_size) + "Max_len" + str(self.max_len) + "Bert_ML" + str(self.bert_max_len) + "DP_" + str(self.dropout_prob) + "EDP_" + str(self.entity_pair_dropout)
        self.log_save_name = 'LOG_' + args.model_name + '_DATASET_' + self.dataset + "_LR_" + str(self.learning_rate) + "_BS_" + str(self.batch_size) + "Max_len" + str(self.max_len) + "Bert_ML" + str(self.bert_max_len) + "DP_" + str(self.dropout_prob) + "EDP_" + str(self.entity_pair_dropout)
        self.result_save_name = 'RESULT_' + args.model_name + '_DATASET_' + self.dataset + "_LR_" + str(self.learning_rate) + "_BS_" + str(self.batch_size) + "Max_len" + str(self.max_len) + "Bert_ML" + str(self.bert_max_len)+ "DP_" + str(self.dropout_prob) + "EDP_" + str(self.entity_pair_dropout) + ".json"
        

        # log setting
        self.period = args.period
        self.test_epoch = args.test_epoch

        # debug
        self.debug = args.debug
        if self.debug:
            self.dev_prefix = self.train_prefix
            self.test_prefix = self.train_prefix
