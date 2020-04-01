# “公益AI之星”挑战赛-新冠疫情相似句对判定大赛 解决方案
**Team**:tbam3  youfeng@buaa.edu.cn final LB:4th(96.30)  [比赛地址](https://tianchi.aliyun.com/competition/entrance/231776/introduction)
## Index
1. 算法说明
2. 代码说明
3. 运行环境
4. 运行说明
5. 参考资料
____
## 1. 算法说明
本解决方案使用了基于病名\药名的数据增强+模型融合+训练时-测试时增强+伪标签的解决方案
* 基于病名\药名的数据增强 Data augmentation  

根据比赛组织方的信息，总共肺炎”、“支原体肺炎”、“支气管炎”、“上呼吸道感染”、“肺结核”、“哮喘”、“胸膜炎”、“肺气肿”、“感冒”、“咳血”十个病种，但是在train和dev数据集中仅仅出现了八个病种，其他的两个“肺结核”与“支气管炎”病种并没有出现，推测在test中包括了剩下的两个病种，是这次比赛的一个关键信息。  

本次比赛需要模型学习的内容主要包括以下几个点：匹配语义信息，病名信息，药名信息，病理信息，我们需要针对这四个点来进行数据增强。

在测试集中，“肺结核”和“支气管炎”两个病种的测试数据中显然含有我们已有标注数据没有的病名、药名信息，但是这些信息是较为易得的；对于语义匹配信息和病理信息，1. 其生成难度要远远高于前两者，2.且很可能改变原数据集中的语义匹配和病理信息，出于这两点考虑，本解决方案采取了替换原数据中病名\药名的数据增强。

在实现过程中，挑选了病理与“肺结核”、“支气管炎”较为接近的“支原体肺炎”与“哮喘”标注数据中的部分样本，作病名替换，添加到原始标注数据中作为训练数据集。LB上升1.9个千分点(96.10->96.29)

* 模型融合 models fushion

本解决方案使用了ernie + bert_wwm_ext + roberta_large_pair的融合模型，对最后的结果使用平均值。具体的来源和下载地址见参考资料。提升2.5个千分点(95.75->96.10)

* 训练时-测试时增强 train-test time augmentation

本解决方案中，在预测时，首先用原测试集预测一遍标签；然后将原测试集的query1和query2字段交换，再次预测一遍；最后将两个结果相加作为最后的预测结果。出于训练时模型拟合方向的偏差考虑，在训练时也训练了两种模型，分别用于预测正序\逆序时的数据集，这一做法的提高非常稳定。

这样的技巧是为了让模型在学习\预测过程中看到数据的更多方面，结合数据中包含的边角信息。LB上升2个千分点(95.59->95.75)
> 注：这个地方的提升不仅是添加了train-test time augmentation, 另外考虑时间因素移除了pseudo_label, 故估计实际上升为2个千分点左右。

* 伪标签  pseudo label

在预测完成后，使用预测结果和原训练集一起作为新的训练集再次训练一个模型做预测。LB上升1个万分点(96.29->96.30)

### 主要提升的过程
|algo|LB|
|-|-|
|bert-base|94.45|
|ernie|95.08|
|ernie + pseudo_label|95.16|
|ernie + bert-base + cwe + pseudo_label|95.59|
|ernie + bert-base + cwe + train-test_time_aug|95.75|
|ernie + cwe + roberta-large-pair + train-test_time_aug|96.10|
|erinie + cwe + roberta-large-pair + train-test_time_aug + oov_sick_data_augmentation|96.29|
|erinie + cwe + roberta-large-pair + train-test_time_aug + oov_sick_data_augmentation + pseudo_label|96.30|

### 题外话
* 基本没调参，roberta-large-pair稍微调了一下，但是毕竟dev不是特别可信... 中间一段时间(3.10~3.17)提升比较多，然后最后10天开始玩杂技，比较想通过数据增强来获得更多的提升，一直没有提升...毕竟猜到测试集里的语义匹配信息以及病例信息难度过高...最后2天幡然悔悟调了调参，dev上暴涨3个千分点，然后LB暴跌了3个千分点。。。还是头太铁了，调参绝非一日之功。
* 没有用K折的原因：因为使用train-test time augmentation已经6个模型了，太多了我心里过意不去也不好调，故放弃了K折，实际上我的train-test time augmentation也有一定的去随机性作用，其实可以考虑加个3折比较合适。
* 没有用fgm & pgd的原因：纯菜。看了大佬们的post后试了一下，pgd效果爆棚，已经在这个repo里更新了这两个算法和相关的参数。
* 本人第一次打nlp的比赛，打过两次feature-based，都被吊打。。。奇淫技巧实在不会，所以在一开始的时候就把重心放在数据上，毕竟对新手比较友好，提升的机会更大。

## 2. 代码说明
* 代码主体结构  
依照代码规范的代码主体结构
```
project
	|-- README.md               
	|-- data                           
	|-- user_data                            
	|-- code                   
    	|-- prediction_result       
```
* `data` 文件夹  
其中 `Dataset` 文件夹下内容已经确定，这里对 `External` 文件夹下进行说明
```
|-- data
    |-- Dataset
    |-- External
        |-- models
            |-- ernie
            |-- chinese_wwm_ext_pytorch
            |-- roberta_large_pair
        |-- knowledge
            |-- sick2drugs.json
```
其中， `models` 文件夹下为下载的预训练模型， `knowledge` 文件夹下为从互联网获取的医药专业知识映射：适用症的用药列表。获取的爬虫文件放在 `code` 文件夹下。
* `user_data` 文件夹
```
|-- user_data
	|-- model_data
		|-- pretrained
 		|-- pretrained_reverse
        	|-- pseudo
	|-- tmp_data
		|-- tmp.dat
```
`model_data` 文件夹下放有三个文件夹，分别为 `pretrained` ， `pretrained_reverse` 以及 `pseudo` 文件夹，由于本解决方案使用了正序-逆序的训练时-测试时增强，故需要使用正序以及逆序两套模型，最后使用伪标签也会再次训练一个模型，分别保存在这三个文件夹下。
* `code` 文件夹
```
|-- code
    |-- run.sh
    |-- run.py
    |-- run_glue_for_test.py
    |-- data_augmentation.py
    |-- drug_crawler.py
```
`code` 文件下存放着   
* `run.sh` 为训练以及测试的入口脚本
* `run.py` 为算法的主体代码 
* `run_glue_for_test.py` 为训练模型的主体代码; 修改自huggingface的 [`transformers/examples/run_glue.py`](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py)
* `data_augmentation.py` 为数据增强的代码
* `drug_crawler.py` 为获取病名——药品名映射词典的爬虫
> 注: **修改了 `transformers` 的 `processor` 部分源码来适应本次比赛任务**，修改后的源码为 `code` 文件夹下 `data` 文件夹，运行时安装完 `transformers` 后替换即可。
## 3. 运行环境
* 软件环境  

|dependency|version|
|-|-|
|pytorch|1.3.1|
|cuda|10.1.243|
|numpy|1.17.3|
|pandas|0.25.3|
|transformers|2.5.1|
|tqdm|4.36.1|
* 硬件配置  
因为 `roberta_large_pair` 使用**128**的`max_seq_length`，故16G显卡会OOM，原本训练时使用的是一张NVIDIA V100 NVLINK 32GB。

## 4. 运行说明
* step1 使用 `python data_augmentaion.py` 生成增强数据
* step2 使用 `sh run.sh` 来训练和测试，具体的参数可以使用 `python run.py --help` 来查看参数列表及其用途。 

## 5. 参考资料
### 预训练模型来源
* [ernie](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)  from 百度，这里提供的地址是pytorch版本的仓库。
[下载地址](https://pan.baidu.com/s/1qSAD5gwClq7xlgzl_4W3Pw)
* [bert_wwm_ext](https://github.com/ymcui/Chinese-BERT-wwm) from 哈工大讯飞联合实验室
[下载地址](https://pan.iflytek.com/link/B9ACE1C9F228A0F42242672EF6CE1721)
* [roberta_large_pair](https://github.com/CLUEbenchmark/CLUEPretrainedModels) from CLUE
[下载地址](https://pan.baidu.com/s/1hoR01GbhcmnDhZxVodeO4w)

### 先验医药知识来源
* [丁香医生用药助手查询](http://drugs.dxy.cn/search/indication.htm)

