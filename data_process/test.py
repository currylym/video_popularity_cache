# -*- coding: utf-8 -*-
import pandas as pd
import json
import numpy as np

import sys
sys.path.append('..')
from parameters1 import OUT_DIR

triples = pd.read_csv(r'../%skg_embedding/kg_data/train2id.txt' % OUT_DIR, names = ['entity1','entity2','relation'],skiprows = 1,sep = ' ')
embedding = json.loads(open(r'../%skg_embedding/kge_res/embedding.vec.json' % OUT_DIR).read())
triples['test'] = triples['entity1'].apply(lambda x:np.array(embedding['ent_embeddings'][x])) + \
                  triples['relation'].apply(lambda x:np.array(embedding['rel_embeddings'][x])) -\
                  triples['entity2'].apply(lambda x:np.array(embedding['ent_embeddings'][x]))

triples['test'] = triples['test'].apply(lambda x:np.sum(x*x))

triples['entity1_mold'] = triples['entity1'].apply(lambda x:np.array(embedding['ent_embeddings'][x]))
triples['entity1_mold'] = triples['entity1_mold'].apply(lambda x:np.sum(x*x))

triples['entity2_mold'] = triples['entity2'].apply(lambda x:np.array(embedding['ent_embeddings'][x]))
triples['entity2_mold'] = triples['entity2_mold'].apply(lambda x:np.sum(x*x))

#np.sum((np.array(embedding['ent_embeddings'][0]) - np.array(embedding['ent_embeddings'][1]))^2)

#save
entity_id = pd.read_csv(r'../%skg_embedding/kg_data/entity2id.txt' % OUT_DIR, names = ['entity','id'],skiprows = 1,sep = ' ')
entity_id_enm = dict(zip(list(entity_id['entity']),embedding['ent_embeddings']))
out = open(r'../%skg_embedding/kge_res/entity_embedding.json' % OUT_DIR, 'w')
out.write(json.dumps(entity_id_enm,indent = 1))
out.close()
