# -*- coding: utf-8 -*-

import pandas as pd
import os
import re
import json

import sys
sys.path.append('..')
from parameters import OUT_DIR,DATA_DIR

history_path = r'../%syouku_history.csv' % OUT_DIR
history = pd.read_csv(history_path)

video_info_path = r'../%syouku_info_all.csv' % DATA_DIR
video_info = pd.read_csv(video_info_path,lineterminator = '\n')
video_info['video_id'] = video_info['link'].apply(lambda x:re.search(r'id_(.*)\.html',x).group(1)\
          if re.search(r'id_(.*).html',x) else '')
video_info['video_id'] = video_info['video_id'].apply(lambda x:x.replace('=',''))
video_info_inHistory = video_info[video_info['video_id'].\
                                  apply(lambda x:x in set(list(history['video_id'])))].copy()
video_info_inHistory['user'] =  video_info_inHistory['user'].\
                                  apply(lambda x:json.loads(x.replace('\'','"'))['name'])
print('history video num : %d' % len(set(list(history['video_id']))))
print('history video with info num : %d' % video_info_inHistory.shape[0])
#video_info_inHistory.to_csv(r'history_info.csv',index = False)
#assert 1==2
relation2id = {'watched':0,
             'belongsTo':1,
             'actedIn':2,
             'published':3}

entitys = []
entitys.extend(list(set(history['video_id'])))
entitys.extend(list(set(history['user'])))
entitys.extend(list(set(video_info_inHistory['user'])))
entitys.extend(list(set(video_info_inHistory['category'])))
entity2Ix = dict(zip(entitys,range(len(entitys))))
#print(entitys)

#relation1
r1 = history[['user','video_id']].copy()
r1['relation'] = 0
r1.rename(columns={'user':'entity1','video_id':'entity2'},inplace = True)

#relation2
r2 = video_info_inHistory[['video_id','category']].copy()
r2['relation'] = 1
r2.rename(columns={'video_id':'entity1','category':'entity2'},inplace = True)

#relation3


#relation4
r4 = video_info_inHistory[['user','video_id']].copy()
r4['relation'] = 3
r4.rename(columns={'user':'entity1','video_id':'entity2'},inplace = True)

triples = pd.concat([r1,r2,r4])
for column in ['entity1','entity2']:
    triples[column] = triples[column].apply(lambda x:entity2Ix[x])

#save
out_dir = r'../%skg_embedding/kg_data/' % OUT_DIR
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

out = open(out_dir + 'train2id.txt','w')
out.write(str(len(triples)) + '\n')
for r in triples.values:
    out.write(' '.join(map(str,list(r)))+ '\n')
out.close()

out = open(out_dir + 'entity2id.txt','w',encoding='utf8')
out.write(str(len(entitys)) + '\n')
for r in entity2Ix.items():
    out.write(' '.join(map(str,list(r)))+ '\n')
out.close()

out = open(out_dir + 'relation2id.txt','w')
out.write(str(len(relation2id)) + '\n')
for r in relation2id.items():
    out.write(' '.join(map(str,list(r)))+ '\n')
out.close()
