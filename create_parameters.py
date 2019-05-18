
'''
创建参数文件
'''

import argparse

parser = argparse.ArgumentParser(description="create parameter file")
parser.add_argument('--data_dir', default='data/', help='raw data dir')
parser.add_argument('--out_dir', default='out/', help='processed dat dir')

parser.add_argument('--min_popularity', default=5, type=int, help='filter popular videos')
parser.add_argument('--update_cycle', default='6H', help='sample cycle')
parser.add_argument('--history_num', default=7, type=int, help='history cycle nums to predict next cycle')

# 验证集在训练集里面抽取
parser.add_argument('--train_start_date', default='2016-10-3', help='')
parser.add_argument('--train_end_date', default='2016-10-23', help='')
parser.add_argument('--test_start_date', default='2016-10-24', help='')
parser.add_argument('--test_end_date', default='2016-10-30', help='')

parser.add_argument('--bound', default=200, help='')
parser.add_argument('--similar_video_num', default=10, type=int, help='similar iqiyi video num')
parser.add_argument('--encode_model', default='lstm', choices=['lstm','multi-cnn','transfomer'], help='encoder model')
parser.add_argument('--use_pairwise_loss', default='yes', choices=['pairwise','pointwise'], help='')

parser.add_argument('--max_word_len_title', default=8, type=int, help='')
parser.add_argument('--max_word_len_tags', default=4, type=int, help='')
parser.add_argument('--max_word_len_description', default=30, type=int, help='')
parser.add_argument('--char_emb_dim', default=300, type=int, help='')
parser.add_argument('--word_emb_dim', default=200, type=int, help='')

parser.add_argument('--out' ,default='parameters.py', help='out path')

def main(args):
    f = open(args.out,'w')
    f.write('DATA_DIR = \'%s\'\n' % args.data_dir)
    f.write('OUT_DIR = \'%s\'\n' % args.out_dir)
    f.write('MIN_POPULARITY = %d\n' % args.min_popularity)
    f.write('UPDATE_CYCLE = \'%s\'\n' % args.update_cycle)
    f.write('HISTORY_NUM = %d\n' % args.history_num)
    f.write('TRAIN_START_DATE = \'%s\'\n' % args.train_start_date)
    f.write('TRAIN_END_DATE = \'%s\'\n' % args.train_end_date)
    f.write('TEST_START_DATE = \'%s\'\n' % args.test_start_date)
    f.write('TEST_END_DATE = \'%s\'\n' % args.test_end_date)
    f.write('BOUND = %d\n' % args.bound)
    f.write('SIMILAR_VIDEO_NUMS = %d\n' % args.similar_video_num)
    f.write('ENCODE_MODEL = \'%s\'\n' % args.encode_model)
    f.write('USE_PAIRWISE_LOSS = \'%s\'\n' % args.use_pairwise_loss)
    f.write('MAX_WORD_LEN_TITLE = %d\n' % args.max_word_len_title)
    f.write('MAX_WORD_LEN_TAGS = %d\n' % args.max_word_len_tags)
    f.write('MAX_WORD_LEN_DESCRIPTION = %d\n' % args.max_word_len_description)
    f.write('CHAR_EMB_DIM = %d\n' % args.char_emb_dim)
    f.write('WORD_EMB_DIM = %d\n' % args.word_emb_dim)
    f.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

