# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from parameters import OUT_DIR,DATA_DIR

import codecs

def _load_char_vec(path):
    char_vec = {}
    with codecs.open(path, 'r', 'utf8') as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            char_vec[parts[0]] = list(map(float, parts[1:]))
    char_vec[''] = [0.0] * len(char_vec[u'的'])
    return char_vec

# 读取fasttext embedding
fasttext_char = _load_char_vec('../%schar.300.vec' % DATA_DIR)

#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import json
import re

def simple_sentEM(sentence,method='mean'):
    if not sentence:
        return np.zeros(300)
    sentence = str(sentence)
    sent_em = []
    for character in sentence:
        if character in fasttext_char:
            sent_em.append(np.array(fasttext_char[character]))
        else:
            sent_em.append(np.zeros(300))
    if method == 'mean':
        return list(np.mean(sent_em,axis = 0))
    if method == 'max':
        return list(np.max(sent_em,axis = 0))

# 对训练集和测试集中的视频做embedding
def get_video_info():

    # 读取视频列表
    video1 = list(pd.read_csv('../%strain_youku_ts.csv' % OUT_DIR, index_col=0).index)
    video2 = list(pd.read_csv('../%stest_youku_ts.csv' % OUT_DIR, index_col=0).index)
    all_video = list(set(video1+video2))
    print(all_video[:10])
    print('train+test video num:%d' % len(all_video))

    # 读取视频的文本特征
    video_info_path = r'../%syouku_info_all.csv' % DATA_DIR
    video_info = pd.read_csv(video_info_path,lineterminator = '\n')
    video_info['video_id'] = video_info['link'].apply(lambda x:re.search(r'id_(.*)\.html',x).group(1)\
              if re.search(r'id_(.*).html',x) else '')
    video_info['video_id'] = video_info['video_id'].apply(lambda x:x.replace('=',''))
    video_info_inHistory = video_info[video_info['video_id'].\
                                      apply(lambda x:x in set(all_video))].copy()
    video_info_inHistory['user'] =  video_info_inHistory['user'].\
                                      apply(lambda x:json.loads(x.replace('\'','"'))['name'])
    print('video with info num : %d' % video_info_inHistory.shape[0])

    
    video_char_em = video_info_inHistory[['video_id','tags','title','description']].copy()
    for column in ['tags','title','description']:
        video_char_em[column] = video_char_em[column].apply(simple_sentEM)

    video_info_dict = video_char_em.to_dict(orient="records")
    video_info_dict = {i['video_id']:{'tags':i['tags'],
                                      'title':i['title'],
                                      'description':i['description']} for i in video_info_dict}

    print('save fasttext-char embedding to ../%sfasttext_char_em.json' % OUT_DIR)
    out = open(r'../%sfasttext_char_em.json' % OUT_DIR, 'w')
    out.write(json.dumps(video_info_dict,indent = 1,ensure_ascii=False, sort_keys=False))
    out.close()

if __name__ == '__main__':
    get_video_info()
