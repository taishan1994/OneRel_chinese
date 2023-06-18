# OneRel_chinese
2023-06-18

- 修改data_loader.py，不用再额外需要keras_bert、keras、tensorflow。
- 修改训练时验证的策略。

- 增加一般步骤：

```python
git clone https://github.com/taishan1994/OneRel_chinese.git

在pre_trained_bert下新建chinese-bert-wwm-ext，并去hugging face上下载config.json、pytorch_model.bin、vocab.txt到其下

针对自己数据集在data下新建数据集文件夹，比如DUIE，里面包含的数据为：
train_triples.json
dev_triples.json
test_triples.json
rel2id.json
其中xxx_triples.json里面的格式都是一样的，具体为：
[{"text": "摩尔多瓦共和国（摩尔多瓦语：Republica Moldova，英语：Republic of Moldova），简称摩尔多瓦，是位于东南欧的内陆国，与罗马>尼亚和乌克兰接壤，首都基希讷乌", "triple_list": [["摩尔多瓦", "首都", "基希讷乌"]]}, ......]
rel2id.json为标签，具体为：
[{"0": "注册资本", "1": "嘉宾", "2": "国籍", "3": "制片人", "4": "配音", "5": "作者", "6": "总部地点", "7": "导演", "8": "简称", "9": "票房", "10": "主题曲", "11": "号", "12": "主角", "13": "母亲", "14": "编剧", "15": "邮政编码", "16": "主演", "17": "作曲", "18": "获奖", "19": "主持人", "20": "董事长", "21": "上映时间", "22": "丈夫", "23": "代言人", "24": "人口数量", "25": "气候", "26": "作词", "27": "占地面积", "28": "官方语言", "29": "祖籍", "30": "毕业院校", "31": "所在城市", "32": "海拔", "33": "专业代码", "34": "首都", "35": "朝代", "36": "妻子", "37": "修业年限", "38": "所属专辑", "39": "成立日期", "40": "创始人", "41": "饰演", "42": ">校长", "43": "改编自", "44": "歌手", "45": "出品公司", "46": "面积", "47": "父亲"}, {"注册资本": 0, "嘉宾": 1, "国籍": 2, "制片人": 3, "配音": 4, "作者": 5, "总部地点": 6, "导演": 7, "简称": 8, "票房": 9, "主题曲": 10, "号": 11, "主角": 12, "母亲": 13, "编剧": 14, "邮政编码": 15, "主演": 16, "作曲": 17, "获奖": 18, "主持人": 19, "董事长": 20, "上映时间": 21, "丈夫": 22, "代言人": 23, "人
口数量": 24, "气候": 25, "作词": 26, "占地面积": 27, "官方语言": 28, "祖籍": 29, "毕业院校": 30, "所在城市": 31, "海拔": 32, "专业
代码": 33, "首都": 34, "朝代": 35, "妻子": 36, "修业年限": 37, "所属专辑": 38, "成立日期": 39, "创始人": 40, "饰演": 41, "校长": 42, "改编自": 43, "歌手": 44, "出品公司": 45, "面积": 46, "父亲": 47}]

在data_loader.py里面re_collate_fn函数里面batch_triple_matrix = torch.LongTensor(cur_batch_len, 48, ax_text_len, max_text_len).zero_()需要修改为关系的数目。DUIE关系数目为48。

训练：python train.py --dataset=DUIE --batch_size=4 --rel_num=48
默认训练为1个epoch，可通过--max_epoch来指定。同时，默认长度为100，可通过--max_len指定，并设置--bert_max_len为max_len的一倍。
注意：可能需要训练足够长的时间，具体可见train.log

测试：python test.py --dataset=DUIE --batch_size=4 --rel_num=48
```

****

OneRel在中文关系抽取中的使用。使用的数据集是百度关系抽取数据集DUIE。中文预训练模型是bert-base-chinese。数据和训练好的模型下载：<br>
链接：https://pan.baidu.com/s/1vyDIqCspTIaGOj5tSGlCJg<br>
提取码：uccp

# 依赖
```python
# keras_bert==0.88.0
matplotlib==3.3.2
numpy==1.19.2
scikit-learn==1.0.1
torch==1.8.1+cu111
transformers==4.5.1
# tensorflow==2.2.0
# keras==2.4.3
```

# 说明
基于原始论文《OneRel: Joint Entity and Relation Extraction with One Module in One Step》的代码：~~https://github.com/ssnvxia/OneRel~~ https://github.com/China-ChallengeHub/OneRel 进行的修改，主要变动的地方如下：
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
