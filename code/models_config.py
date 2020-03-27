# 预测结果位置
RESULT_PATH = '../prediction_result/result.csv'

# 预测结果logits临时位置
LOGITS_PATH = '../user_data/tmp_data/result.npy'

# 训练时的模型配置
train_config = {
    'ernie': {},
    'chinese_wwm_ext_pytorch':{},
    'roberta_large_pair': {
        'learning_rate': 9e-6,
    }
}

# 伪标签训练时的模型配置
pseudo_config = {
    'ernie': {},
}