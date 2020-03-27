import requests
import json
##from tqdm import tqdm
from bs4 import BeautifulSoup as bs



def get_soup(url):
    return bs(requests.get(url).text)

def parse_soup(soup):
    try:
        drugs = soup.find('div', class_='m49').ul.find_all('h3')
        return [i.text.strip().split('\t')[0] for i in drugs]
    except:
        return None

if __name__ == '__main__':
    sicknesses = [
        '上呼吸道感染',
        '感冒',
        '肺炎',
        '哮喘',
        '胸膜炎',
        '肺气肿',
        '支原体肺炎',
        '咳血',
        '肺结核',
        '支气管炎'
    ]
    
    sick_drugs_dic = {sickness:[] for sickness in sicknesses}
    for sickness in sicknesses:
        print(f'Crawling for {sickness} on page {pageNo}')
        for pageNo in range(1, 51):
            url = f'http://drugs.dxy.cn/search/indication.htm?page={pageNo}&keyword={sickness}'
            drugs = parse_soup(get_soup(url))
            if drugs==None:
                break
            sick_drugs_dic[sickness] += drugs

    json.dump(sick_drugs_dic, open('疾病_药品.json', 'w', encoding='utf-8'), \
              ensure_ascii=False, \
              indent=2)
