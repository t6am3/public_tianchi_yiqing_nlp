from run_glue_for_test import *
from models_config import train_config, pseudo_config, RESULT_PATH, LOGITS_PATH

import pandas as pd
import time
import numpy as np
import sys
import os
import argparse
import logging

run_logger = logging.getLogger(__name__)


BASE_MODEL_PATH = '../data/External/models'
PRETRAINED_MODEL_PATH = '../user_data/model_data'
TMP_DATA_PATH = '../user_data/tmp_data'
MIDDLE_DATA_PATH = '../data/Dataset'


# 测试时增强 翻转数据临时文件路径
reverse_test_path = os.path.join(MIDDLE_DATA_PATH, 'reverse_test.csv')

'''
用来替代run_glue.py的输入参数
'''
class Args():
    data_dir = None
    predict_dir = None
    seed = 43
    
    def __init__(self, model_name):
        # 默认参数列表
        ## Base args
        self.model_type = 'bert'
        self.output_dir = 'output'
        self.do_train = True
        self.do_eval = False
        self.task_name = 'CUSTOM'
        
        ## Key args
        self.max_seq_length = 128
        self.per_gpu_train_batch_size = 32
        self.per_gpu_eval_batch_size = 32
        self.gradient_accumulation_steps = 1
        self.learning_rate = 2e-5
        self.num_train_epochs = 3
        self.weight_decay = 0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1
        self.max_steps = -1
        self.warmup_steps = 0
        
        self.config_name = ''
        self.tokenizer_name = ''
        self.cache_dir = ''
        self.evaluate_during_training = False
        self.do_lower_case = True
        self.logging_steps = 10000
        self.save_steps = 10000
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_cache = False
        self.fp16 = False
        self.fp16_opt_level = '01'
        self.local_rank = -1
        self.server_ip = ''
        self.server_port = ''
        self.overwrite_output_dir = True
        self.pred_poss = True
        
        self.model_name_or_path = os.path.join(BASE_MODEL_PATH, model_name)

# Help funcs
'''
将数据取翻转
'''
def get_reverse_df(df):
    query1 = list(df['query1'])
    df['query1'] = df['query2']
    df['query2'] = query1
    return df

'''
验证
'''
def simple_acc(pred, labels): return sum(pred==labels) / len(labels)

'''
得到带有前缀的路径字符串
'''
def get_prefix_path(path, prefix): return path.replace(path.split('/')[-1], prefix + path.split('/')[-1])
    
'''
生成伪标签训练集
'''
def generate_pseudo_data(execute_args):
    train = pd.read_csv(os.path.join(execute_args.pseudo_data_dir, 'train.csv'))
    test = pd.read_csv(execute_args.predict_file)
    pred = pd.read_csv(RESULT_PATH)

    # print('*'*50)
    # print('Initial acc: ', sum(test['label']==pseudo.values.argmax(1))/len(test))
    # print('*'*50)

    test['label'] = pred
    train = pd.concat([train, test]).reset_index(drop=True)
    
    pseudo_path = get_prefix_path(execute_args.data_dir, 'pseudo_')
    if not os.path.exists(pseudo_path):
        os.mkdir(pseudo_path)

    train.to_csv(os.path.join(pseudo_path, 'train.csv'), index=False)
    
'''
设定融合的模型以及参数
'''
def set_model_args(model_config):
    model_args = {}
    for model, config in model_config.items():
        model_args[model] = Args(model)
        for argk, argv in config.items():
            setattr(model_args[model], argk, argv)
    return model_args

'''
根据设定的模型以及参数运行代码
'''
def run_fushion_tta(execute_args, configs, do_train=True, pseudo=False, reverse=True):
    tmp_time = time.time()
    
    if execute_args.eval_verbose:
        test_df = pd.read_csv(execute_args.predict_file)
            
    model_args = set_model_args(configs)

    model_pred_results = {}
    model_reverse_pred_results = {}

    for name, args in model_args.items():
        if pseudo:
            args.data_dir = execute_args.pseudo_data_dir
            
        ## Normal prediction
        print(f"Use {name} to predict:")
        if do_train:
            # If train, switch save path
            if pseudo:
                args.output_dir = os.path.join(PRETRAINED_MODEL_PATH, 'pseudo', name)
            else:
                args.output_dir = os.path.join(PRETRAINED_MODEL_PATH, 'pretrained', name)
        else:
            args.model_name_or_path = os.path.join(PRETRAINED_MODEL_PATH, 'pretrained', name)
            args.do_train = False

        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        main(args)
        model_pred_results[name] = np.load(LOGITS_PATH)

        if execute_args.eval_verbose:           
            run_logger.info('*'*50)
            run_logger.info(f"{name} predicted, time_spent: {time.time()-tmp_time}")
            tmp_time = time.time()
            run_logger.info(f"eval_acc: {simple_acc(model_pred_results[name].argmax(axis=1), test_df['label'])}")
            run_logger.info('*'*50)

        
        if reverse:
            ## Reverse prediction
            print(f"Use {name} to predict reverse:")
            args.data_dir = execute_args.reverse_data_path
            if do_train:
                # If train, switch save path
                if pseudo:
                    args.output_dir = os.path.join(PRETRAINED_MODEL_PATH, 'pseudo', name)
                else:
                    args.output_dir = os.path.join(PRETRAINED_MODEL_PATH, 'pretrained_reverse', name)
            else:
                args.model_name_or_path = os.path.join(PRETRAINED_MODEL_PATH, 'pretrained_reverse', name)
                args.do_train = False

            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
        #     args.data_dir = reverse_data_path
            if pseudo:    
                args.model_name_or_path = os.path.join(PRETRAINED_MODEL_PATH, 'pseudo', name)

            args.predict_dir = reverse_test_path
            main(args)
            model_reverse_pred_results[name] = np.load(LOGITS_PATH)

        if execute_args.eval_verbose and reverse:
            run_logger.info('*'*50)
            run_logger.info(f"{name} reverse predicted, time_spent: {time.time()-tmp_time}")
            tmp_time = time.time()
            run_logger.info(f"eval_acc: {simple_acc(model_reverse_pred_results[name].argmax(axis=1), test_df['label'])}")
            run_logger.info('*'*50)


            ## Merge prediction evaluation
            merge_pred = ((model_reverse_pred_results[name] + model_pred_results[name])/2).argmax(axis=1)
            run_logger.info('*'*50)
            run_logger.info(f"merge eval_acc: {simple_acc(merge_pred, test_df['label'])}")
            run_logger.info('*'*50)

    normal_preds = (sum(model_pred_results.values()) / len(model_pred_results))
    if reverse:
        reverse_preds = (sum(model_reverse_pred_results.values()) / len(model_reverse_pred_results))
    
    if reverse:
        final_preds_poss = sum([normal_preds, reverse_preds]) / 2
    else:
        final_preds_poss = normal_preds
    
    final_preds = final_preds_poss.argmax(axis=1)

    res_df = pd.DataFrame({"id":range(len(final_preds)), "label":final_preds})
    res_df.to_csv(RESULT_PATH, index=False)
    
    if execute_args.save_logits:
        np.save(os.path.join(TMP_DATA_PATH, 'result.npy'), final_preds_poss)
    if execute_args.save_ind_logits:
        np.save(os.path.join(TMP_DATA_PATH, 'model_pred_results.npy'), model_pred_results)
        if reverse:
            np.save(os.path.join(TMP_DATA_PATH, 'model_reverse_pred_results.npy'), model_reverse_pred_results)


'''
主函数
'''
def execute_main():
    parser = argparse.ArgumentParser()
    
    # 可以指定的参数
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="数据文件的文件夹路径，下面应该有train.csv作为训练数据集",
    )
    
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        required=True,
        help="待预测数据文件的路径",
    )
    
    parser.add_argument(
        "--random_seed",
        default=43,
        type=int,
        help="指定随机数种子",
    )
    
    parser.add_argument("--do_train", action="store_true", help="是否进行训练。如果训练，需要在models_setting中指定train_config")
    parser.add_argument("--do_eval", action="store_true", help="是否开启验证。如果需要开启验证，待预测数据集的标签列应该为真实值")
    parser.add_argument("--eval_verbose", action="store_true", help="是否输出各个模型正序-反序的验证结果")
    parser.add_argument("--use_reverse_in_train", action="store_true", help="是否使用正序-反序训练时增强")
    parser.add_argument("--use_reverse_in_test", action="store_true", help="是否使用正序-反序测试时增强")
    parser.add_argument("--pseudo", action="store_true", help="是否使用伪标签。如果使用，需要在models_setting中指定pseudo_config")
    parser.add_argument(
        "--pseudo_data_dir",
        default=None,
        type=str,
        help="伪标签训练的文件夹路径，下面应该有train.csv作为原始训练数据集",
    )
    parser.add_argument("--save_logits", action="store_true", help="是否保存多模型产生的最终logits结果")
    parser.add_argument("--save_ind_logits", action="store_true", help="是否保存多模型分别产生的logits结果")
    
    execute_args = parser.parse_args()
    
    Args.data_dir = execute_args.data_dir
    Args.predict_dir = execute_args.predict_file
    Args.seed = execute_args.random_seed
    
    # 准备
    ## 训练时增强数据路径
    reverse_data_path = pseudo_path = get_prefix_path(execute_args.data_dir, 'reverse_')
    execute_args.reverse_data_path = reverse_data_path
    
    if not os.path.exists(reverse_data_path):
        os.mkdir(reverse_data_path)

    train_df = pd.read_csv(os.path.join(execute_args.data_dir, 'train.csv'))
    get_reverse_df(train_df).to_csv(os.path.join(reverse_data_path, 'train.csv'), index=False)

    test_df = pd.read_csv(execute_args.predict_file)
    get_reverse_df(test_df).to_csv(reverse_test_path, index=False)
    
    # 计时
    start = time.time()
    
    # 训练并预测
    run_fushion_tta(execute_args, train_config, do_train=execute_args.do_train, reverse=execute_args.use_reverse_in_train)
    
    if execute_args.do_eval:
        pred = pd.read_csv(RESULT_PATH)['label']
        run_logger.info(f"{'*'*5}Train and predict done.{'*'*5}")
        run_logger.info(f"{'*'*5}Eval acc: {simple_acc(pred, test_df['label'])}{'*'*5}")
        run_logger.info(f"{'*'*5}Time spent: {time.time()-start}{'*'*5}")
    
    for i in range(2):
        run_logger.info(f"{i+1} pseudo prediction:")
        # 伪标签训练并预测
        if execute_args.pseudo:
            generate_pseudo_data(execute_args)
            run_fushion_tta(execute_args, pseudo_config, reverse=execute_args.use_reverse_in_test, pseudo=True)

        if execute_args.do_eval:
            pred = pd.read_csv(RESULT_PATH)['label']
            run_logger.info(f"{'*'*5}PSEUDO Train and predict done.{'*'*5}")
            run_logger.info(f"{'*'*5}Eval acc: {simple_acc(pred, test_df['label'])}{'*'*5}")
            run_logger.info(f"{'*'*5}Time spent: {time.time()-start}{'*'*5}")
    
if __name__ == '__main__':
    execute_main()