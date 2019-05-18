# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from config import MAIN_PATH,WORD_EMB_PATH
from parameters import MAX_WORD_LEN_TITLE,MAX_WORD_LEN_TAGS,MAX_WORD_LEN_DESCRIPTION,CHAR_EMB_DIM,WORD_EMB_DIM

import re
import json
import jieba
import pandas as pd
import codecs
import pickle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from lmdb_embeddings.writer import LmdbEmbeddingsWriter
from lmdb_embeddings.reader import LmdbEmbeddingsReader
from lmdb_embeddings.exceptions import MissingWordError
jieba.load_userdict(MAIN_PATH+'/out/tencent_word_small.txt')

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

def _write_word_vec_lmdb(path,output):
    #print('Loading gensim model...')
    #gensim_model = KeyedVectors.load_word2vec_format(path, binary=False)
    fin = open(MAIN_PATH+'/out/tencent_word.txt','w')
    def iter_embeddings():
        '''
        for word in gensim_model.vocab.keys():
            yield word, gensim_model[word]
        '''
        count = 0
        with open(path,'r',errors='ignore') as f:
            for line in f:
                word,*vector = line.strip().split(' ')
                fin.write(word+'\n')
                count += 1
                if count % 10000 == 0:print(count)
                yield word,np.asarray(vector,dtype='float32')
    
    print('Writing vectors to a LMDB database...')
    writer = LmdbEmbeddingsWriter(iter_embeddings()).write(output)
    
    fin.close()
    
def _filter_dict():
    #腾讯字典太大，去掉太长的word得到小字典
    fin = open(MAIN_PATH+'/out/tencent_word_small.txt','w')
    with open(MAIN_PATH+'/out/tencent_word.txt','r') as f:
        for line in f:
            word = line.strip()
            if len(word) <= 5:
                fin.write(word+'\n')
    fin.close()

def _load_word_vec_lmdb(word,path=MAIN_PATH+'/out/wv_lmdb',word_wv_dim=200):
    embeddings = LmdbEmbeddingsReader(path)
    try:
        vector = embeddings.get_word_vector(word)
        return vector
    except MissingWordError:
        print('%s not in dict' % word)
        return np.zeros(word_wv_dim)

def _word_segment(sentence):
    #去除停用词，加额外字典分词
    print(sentence)
    if not sentence or not isinstance(sentence,str):
        return []
    seg_list = jieba.cut(sentence, cut_all=False)
    stopwords = open(MAIN_PATH+'/data/chinese_stopwords.txt').read().split('\n')
    #print(seg_list)
    return [i for i in seg_list if i not in stopwords]

#针对单个sentence的主要embedding函数
def sentence_embedding(sentence,padding=20,char_pooling='max',word_wv_dim=200,
                       char_cv_dim=300):
    #输出维数：char_dim+word_dim,一般是500
    global fasttext_char
    fasttext_char = _load_char_vec(MAIN_PATH+'/data/char.300.vec')
    
    words = _word_segment(sentence)
    print(words)
    embedding = []
    for word in words:
        word_vec = _load_word_vec_lmdb(word)
        char_vecs = np.array([fasttext_char.get(i,np.zeros(char_cv_dim)) for i in word])
        #print(char_vecs.shape)
        if char_pooling == 'max':
            char_vec = np.max(char_vecs,axis=0)
        elif char_pooling == 'mean':
            char_vec = np.mean(char_vecs,axis=0)
        vec = np.hstack([word_vec,char_vec])
        embedding.append(vec.tolist())
    embedding = np.array(embedding)
    #print(embedding.shape)
    if not embedding.tolist():
        return np.zeros((padding,word_wv_dim+char_cv_dim),dtype='float')
    w_len = len(words)
    if  w_len < padding:
        embedding = np.vstack([embedding,np.zeros((padding-w_len,word_wv_dim+char_cv_dim),
                                                  dtype='float')])
    else:
        embedding = embedding[:padding,:]
    return embedding

def get_video_info(flag='youku'):

    all_video = list(pd.read_csv(MAIN_PATH + '/out/train_test_%s.csv' % flag,index_col=0).index)
    print(all_video[:10])
    print('%s video num : %d' % (flag,len(all_video)))

    video_info_path = MAIN_PATH + r'/data/%s_info_all.csv' % flag
    video_info = pd.read_csv(video_info_path,lineterminator = '\n')
    video_info['video_id'] = video_info['link'].apply(lambda x:re.search(r'id_(.*)\.html',x).group(1)\
              if re.search(r'id_(.*).html',x) else '')
    video_info['video_id'] = video_info['video_id'].apply(lambda x:x.replace('=',''))
    video_info_inHistory = video_info[video_info['video_id'].\
                                      apply(lambda x:x in set(all_video))].copy()
    video_info_inHistory['user'] =  video_info_inHistory['user'].\
                                      apply(lambda x:json.loads(x.replace('\'','"'))['name'])
    print('%s video with info num : %d' % (flag,video_info_inHistory.shape[0]))
    return video_info_inHistory

def main(word_wv_dim=200,char_cv_dim=300,paddings=[4,8,30]):
    '''
    params:
    -------
    paddings:tag,title,description分别的padding长度
    '''
    #读取字向量
    global fasttext_char
    fasttext_char = _load_char_vec(MAIN_PATH+'/data/char.300.vec')
    #将词向量写入本地lmdb
    _write_word_vec_lmdb(WORD_EMB_PATH,MAIN_PATH+'/out/wv_lmdb')
    #读取视频信息
    history_info = get_video_info(flag='youku')
    video_word_char_em = history_info[['video_id','tags','title','description']].copy()
    for column,padding in zip(['tags','title','description'],paddings):
        video_word_char_em[column] = video_word_char_em[column].apply(
                lambda x:sentence_embedding(x,padding=padding))

    video_info_dict = video_word_char_em.to_dict(orient="records")
    video_info_dict = {i['video_id']:{'tags':i['tags'],
                                      'title':i['title'],
                                      'description':i['description']} for i in video_info_dict}
    #保存数据
    try:
        out = open(MAIN_PATH+'/out/word_char_em.json','w')
        out.write(json.dumps(video_info_dict,indent = 1,ensure_ascii=False, sort_keys=False))
        out.close()
    except:
        out = open(MAIN_PATH+'/out/word_char_em.pkl','wb')
        pickle.dump(video_info_dict,out)
        out.close()
    
if __name__ == '__main__':
    main(word_wv_dim=WORD_EMB_DIM,char_cv_dim=CHAR_EMB_DIM,paddings=[MAX_WORD_LEN_TAGS,MAX_WORD_LEN_TITLE,
                                                                     MAX_WORD_LEN_DESCRIPTION])

