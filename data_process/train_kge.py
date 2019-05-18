# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
from parameters1 import OUT_DIR

from OpenKE import config
from OpenKE import models
import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
#Input training files from benchmarks/FB15K/ folder.
con = config.Config()
#True: Input test files from the same folder.
con.set_in_path("../%skg_embedding/kg_data/" % OUT_DIR)
con.set_work_threads(8)
con.set_train_times(1000)
con.set_nbatches(100)
con.set_alpha(0.001)
con.set_margin(1.0)
con.set_bern(0)
con.set_dimension(100)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")

#Models will be exported via tf.Saver() automatically.
con.set_export_files("../%skg_embedding/kge_res/model.vec.tf" % OUT_DIR, 0)
#Model parameters will be exported to json files automatically.
con.set_out_files("../%skg_embedding/kge_res/embedding.vec.json" % OUT_DIR)
#Initialize experimental settings.
con.init()
#Set the knowledge embedding model
con.set_model(models.TransE)
#Train the model.
con.run()
