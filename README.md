# OneRel_chinese
OneRel在中文关系抽取中的使用。使用的数据集是百度关系抽取数据集DUIE。中文预训练模型是bert-base-chinese。数据和训练好的模型下载：<br>

# 依赖
```
keras_bert==0.88.0
matplotlib==3.3.2
numpy==1.19.2
scikit_learn==1.0.1
torch==1.8.1+cu111
transformers==4.5.1
```

# 说明
基于原始论文《OneRel: Joint Entity and Relation Extraction with One Module in One Step》的代码：https://github.com/ssnvxia/OneRel 进行的修改，主要变动的地方如下：
- utils/tokenizer.py：里面修改为针对于中文的token化。
- framework/framework.py：里面test时解码去除掉token化时添加的空格。在re_collate_fn函数里面```batch_triple_matrix = torch.LongTensor(cur_batch_len, 48, ax_text_len, max_text_len).zero_()```需要修改为关系的数目。
- process.py：新增的数据处理文件，主要将duie的数据转换为onerel所需要的格式。
特别需要注意的是onerel会在每一个token后面都添加一个空格，也就是说原始文本为100，那么输入到模型里面的文本长度就是200，因此要考虑到显存的问题。

# 训练和测试
```python
python train.py --dataset=DUIE --batch_size=4 --rel_num=48
```
由于数据量太大，这里只运行了一个epoch。具体可以在config/config.py以及framework/framework.py的里面进行需修改。结果：
```
......
epoch:   0, step: 35500, speed: 257.69ms/b, train loss: 0.001
epoch:   0, step: 35600, speed: 252.92ms/b, train loss: 0.001
epoch   0, eval time: 979.91s, f1: 0.672, precision: 0.622, recall: 0.730
saving the model, epoch:   0, precision: 0.622, recall: 0.730, best f1: 0.672
finish training
best epoch:   0, precision: 0.622, recall: 0.73, best f1: 0.672, total time: 10157.71s
```
此时会保存结果在result/DUIE/RESULT_OneRel_DATASET_DUIE_LR_1e-05_BS_4Max_len100Bert_ML200DP_0.2EDP_0.1.json。部分结果如下所示：
```
{       
    "text": "摩 尔 多 瓦 共 和 国 （ 摩 尔 多 瓦 语 ： republica moldova ， 英 语 ： republic of moldova ） ， 简
 称 摩 尔 多 瓦 ， 是 位 于 东 南 欧 的 内 陆 国 ， 与 罗 马 尼 亚 和 乌 克 兰 接 壤 ， 首 都 基 希 讷 乌",
    "triple_list_gold": [
        {   
            "subject": "摩尔多瓦",
            "relation": "首都",
            "object": "基希讷乌"
        }
    ],      
    "triple_list_pred": [
        {   
            "subject": "摩尔多瓦",
            "relation": "首都",
            "object": "基希讷乌"
        }   
    ],      
    "new": [],
    "lack": []
}
{
    "text": "这 件 婚 事 原 本 与 陈 国 峻 无 关 ， 但 陈 国 峻 却 [UNK] 欲 求 配 而 无 由 ， 夜 间 乃 潜 入 天 >城 公 主 所 居 通 之",
    "triple_list_gold": [
        {
            "subject": "国峻",
            "relation": "妻子",
            "object": "天城公主"
        },
        {
            "subject": "天城公主",
            "relation": "丈夫",
            "object": "国峻"
        }
    ],
    "triple_list_pred": [
        {
            "subject": "天城公主",
            "relation": "丈夫",
            "object": "陈国峻"
        },
        {
            "subject": "陈国峻",
            "relation": "妻子",
            "object": "天城公主"
        }
    ],
	"new": [
        {
            "subject": "天城公主",
            "relation": "丈夫",
            "object": "陈国峻"
        },
        {
            "subject": "陈国峻",
            "relation": "妻子",
            "object": "天城公主"
        }
    ],
    "lack": [
        {
            "subject": "国峻",
            "relation": "妻子",
            "object": "天城公主"
        },
        {
            "subject": "天城公主",
            "relation": "丈夫",
            "object": "国峻"
        }
    ]
}
```
测试：
```python
python test.py --dataset=DUIE --rel_num=48
```

# 感谢
> https://github.com/ssnvxia/OneRel