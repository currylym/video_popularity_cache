
#python create_parameters.py --out parameters.py --out_dir out/ --update_cycle 3H
#cd data_process/
#python get_train_test.py
#python get_similar_video.py
#python fasttext_em.py
#python build_kg_data.py
#python train_kge.py
#python test.py
#cd ../popularity_prediction
#sh main.sh
#cd ../

#python create_parameters.py --out parameters.py --out_dir out/ --update_cycle 12H
#cd data_process/
#python get_train_test.py
#python get_similar_video.py
#python fasttext_em.py
#python build_kg_data.py
#python train_kge.py
#python test.py
#cd ../popularity_prediction
#sh main.sh
#cd ../

python create_parameters.py --out parameters.py --out_dir out/ --history_num 5
cd data_process/
python get_train_test.py
python get_similar_video.py
python fasttext_em.py
#python build_kg_data.py
#python train_kge.py
#python test.py
cd ../popularity_prediction
sh main.sh
cd ../

python create_parameters.py --out parameters.py --out_dir out/ --history_num 9
cd data_process/
python get_train_test.py
python get_similar_video.py
python fasttext_em.py
#python build_kg_data.py
#python train_kge.py
#python test.py
cd ../popularity_prediction
sh main.sh
cd ../






