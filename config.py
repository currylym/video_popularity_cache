# -*- coding: utf-8 -*-

import os

MAIN_PATH = '/home/luyiming/下载/youku_cache_test_2/youku_cache_pp'
print(MAIN_PATH)

CHAR_EMB_PATH = MAIN_PATH + '/data/char.300.vec'
WORD_EMB_PATH = '/Users/luyiming/Downloads/NLP/nlp_data/Tencent_AILab_ChineseEmbedding.txt'
IQIYI_DATA = MAIN_PATH + '/data/iqiyidata.xlsx'
YOUKU_INFO = MAIN_PATH + '/data/youku_info_all.csv'
YOUKU_DATA = MAIN_PATH + '/data/youkudata.csv'

SIMILAR_PATH = MAIN_PATH + '/out/similar_video.json'
IQIYI_HISTORY = MAIN_PATH + '/out/iqiyi_history.csv'
YOUKU_HISTORY = MAIN_PATH + '/out/youku_history.csv'
TRAIN_TEST_IQIYI = MAIN_PATH + '/out/train_test_iqiyi.csv'
TRAIN_TEST_YOUKU = MAIN_PATH + '/out/train_test_youku.csv'

def check():
    for path in [CHAR_EMB_PATH,WORD_EMB_PATH,IQIYI_DATA,YOUKU_INFO,YOUKU_DATA,SIMILAR_PATH,IQIYI_HISTORY,
                 YOUKU_HISTORY,TRAIN_TEST_IQIYI,TRAIN_TEST_YOUKU]:
        print('path %s exists:%s' % (path,os.path.exists(path)))
        
if __name__ == '__main__':
    check()
