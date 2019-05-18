# -*- coding: utf-8 -*-

'''
根据youku和iqiyi的历史请求记录来选择相似视频
'''

import json
import pandas as pd

import sys
sys.path.append('..')
from parameters import OUT_DIR,SIMILAR_VIDEO_NUMS

def get_similar_videos(youku_history,iqiyi_history):
    # 重命名
    youku_history = youku_history.rename(columns={'video_id':'youku_video'})
    iqiyi_history = iqiyi_history.rename(columns={'video_id':'iqiyi_video'})
    # 统计用户观看视频次数
    youku_count = youku_history.groupby(['youku_video','user']).apply(len).reset_index(level=[0,1]).rename(columns={0:'youku_count'})
    iqiyi_count = iqiyi_history.groupby(['iqiyi_video','user']).apply(len).reset_index(level=[0,1]).rename(columns={0:'iqiyi_count'})
    # 统计视频总观看数
    youku_all_count = youku_history.groupby(['youku_video']).apply(len)
    iqiyi_all_count = iqiyi_history.groupby(['iqiyi_video']).apply(len)
    # 按用户进行join
    joined = youku_count.join(iqiyi_count.set_index('user'), on='user', how ='left')
    joined[['youku_count','iqiyi_count']] = joined[['youku_count','iqiyi_count']].astype(float)
    # 按youku视频进行group
    youku_group = joined.groupby('youku_video')

    # 对每个youku视频下的iqiyi视频进行相似度计算
    res = {}
    for youku_video,group in youku_group:
        group = group[['iqiyi_video','youku_count','iqiyi_count']]
        group['mul'] = group['iqiyi_count'] * group['youku_count']
        group = group.groupby(['iqiyi_video'])['mul'].apply(sum)
        for iqiyi_video in group.index:
            group[iqiyi_video] = group[iqiyi_video]/iqiyi_all_count[iqiyi_video]
        # 排序
        group = group.sort_values(ascending=False)[:SIMILAR_VIDEO_NUMS]
        res[youku_video] = list(group.index)
    return res

# 保存
def save(res):
    print('saving similar video info to ../%ssimilar_video.json' % OUT_DIR)
    out = open(r'../%ssimilar_video.json' % OUT_DIR, 'w')
    out.write(json.dumps(res,indent = 1))
    out.close()

# 测试
def test():
    df1 = pd.DataFrame({'video_id':['y1','y2','y3','y2','y4','y1'],'user':[1,2,3,4,2,3]})
    df2 = pd.DataFrame({'video_id':['i1','i1','i2','i2','i3','i3','i1'],'user':[1,2,4,4,2,3,1]})
    print(df1)
    print(df2)
    print(get_similar_videos(df1,df2))

if __name__ == '__main__':
    #test()

    youku_history = pd.read_csv(r'../%syouku_history.csv' % OUT_DIR)[['video_id','user']]
    iqiyi_history = pd.read_csv(r'../%siqiyi_history.csv' % OUT_DIR)[['video_id','user']]
    res = get_similar_videos(youku_history,iqiyi_history)
    print(len(res))
    save(res)
