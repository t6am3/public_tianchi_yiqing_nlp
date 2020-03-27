import pandas as pd
import numpy as np

import os
import json



INPUT_PATH = '../data/Dataset'
CONCAT_PATH = '../data/Dataset/concat_input'
AUG_PATH = '../data/Dataset/aug_input'
KNOWLEDGE_PATH = '../data/External/knowledge'

# 导入病名到药名映射词典
sick2drugs = json.load(open(os.path.join(KNOWLEDGE_PATH, 'sick2drugs.json'), 'r', encoding='utf-8'))

# Help funcs
'''
判断两个字符串的最大公共子串长度是否大于等于3
'''
def lcs(s1, s2):
    max_len = 0
    sub = ''
    
    dp = [[0 for __ in range(len(s2))] for _ in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                dp[i][j] = dp[i-1][j-1]+1 if (i>0 and j>0) else 1
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                sub = s1[i-max_len+1:i+1]
    return max_len>=3

'''
返回两个字符串的最大公共子串
'''
def lcs_sub(s1, s2):
    max_len = 0
    sub = ''
    
    dp = [[0 for __ in range(len(s2))] for _ in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                dp[i][j] = dp[i-1][j-1]+1 if (i>0 and j>0) else 1
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                sub = s1[i-max_len+1:i+1]
    return sub

'''
判断数据是否包含病名(仅病种名)
'''
def tell_if_with_sick(sample): 
    return True if sample['category'] in sample['query1'] or sample['category'] in sample['query2'] else False

'''
判断数据是否包含药名
'''
def tell_if_with_drug(sample): 
    drugs = sick2drugs[sample['category']]
    return True if sum([lcs(i, j) for i in drugs for j in [sample['query1'], sample['query2']]]) else False

'''
将一个DataFrame的所有病名s1替换成病名s2
'''
def replace(df, s1, s2):
    sample = df.copy()
    sample['category'] = sample['category'].map(lambda x: x.replace(s1, s2))
    sample['query1'] = sample['query1'].map(lambda x: x.replace(s1, s2))
    sample['query2'] = sample['query2'].map(lambda x: x.replace(s1, s2))
    return sample

'''
在字符串中寻找药名
'''
def find_drug_in_text(query, cat):
    maybes = sick2drugs[cat]
    common_subs = [lcs_sub(query, drug) for drug in maybes]
    base_drug_name = max(common_subs, key=len)
    
    return base_drug_name

'''
主函数
'''
def main():
    train = pd.read_csv(os.path.join(INPUT_PATH, 'train.csv'))
    dev = pd.read_csv(os.path.join(INPUT_PATH, 'dev.csv'))

    # 将train和dev合在一起
    train = pd.concat([train, dev], sort=False).reset_index(drop=True)
    
    # 保存
    if not os.path.exists(CONCAT_PATH):
        os.mkdir(CONCAT_PATH)
        
    train.to_csv(os.path.join(CONCAT_PATH, 'train.csv'), index=False)
    
    # 病名药名标记
    train['with_sick'] = train.apply(tell_if_with_sick, axis=1)
    train['with_drug'] = train.apply(tell_if_with_drug, axis=1)

    # 替换病名
    fjh_fake_df = replace(train[train['category']=='支原体肺炎'][train['with_sick'] & (train['with_drug']==False)], '支原体肺炎', '肺结核')
    zqgy_fake_df = replace(train[train['category']=='哮喘'][train['with_sick'] & (train['with_drug']==False)], '哮喘', '支气管炎')
    aug_train = pd.concat([train, fjh_fake_df, zqgy_fake_df], sort=False).reset_index(drop=True)
    
    if not os.path.exists(AUG_PATH):
        os.mkdir(AUG_PATH)
        
    aug_train.to_csv(os.path.join(AUG_PATH, 'train.csv'), index=False)

if __name__ == '__main__':
    main()