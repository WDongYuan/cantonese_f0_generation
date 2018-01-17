import matplotlib
matplotlib.use('TkAgg')
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', dest='mode')
	parser.add_argument('--dct_num', dest='dct_num')
	parser.add_argument('--train_small', dest='train_small',type=int)
	parser.add_argument('--dev_small', dest='dev_small',type=int)
	parser.add_argument('--dct_option', dest='dct_option')
	args = parser.parse_args()
	mode = args.mode
	if mode=="how_to_run":
		print("python update_dataset.py"+
			" --mode update_phrase_dataset"+
			" --dct_num 3"+
			" --train_small 1/0"+
			" --dev_small 1/0"
			" --dct_option utterance/phrase")
	elif mode=="update_phrase_dataset":
		dct_num = args.dct_num
		print(">>>>>>>>>>>>>>dct processing<<<<<<<<<<<<<")
		os.system("python ../../DCT/run.py --mode phrase_f0_dct"+
			" --subsample_file ./subsample_f0.save"+
			" --phrase_dir ./phrase_dir/phrase_syllable"+
			" --out_dir ./phrase_dir/phrase_f0_dct"+
			" --option "+args.dct_option+
			" --number "+str(dct_num))

		print(">>>>>>>>>>>>>>calculate residual<<<<<<<<<<<<<")
		os.system("python ../../decision_tree/data_preprocessing.py"+
			" --mode dir_minus"+
			" --dir1 f0_in_file"+
			" --dir2 phrase_dir/phrase_f0_dct/f0"+
			" --out_dir ./phrase_dir/phrase_f0_dct/res")

		print(">>>>>>>>>>>>>>updating train data<<<<<<<<<<<<<")
		os.system("python ../../decision_tree/data_preprocessing.py"+
			" --mode map_to_new_f0_vector"+
			" --map_file ./train_dev_data_vector/train_data/syllable_map"+
			" --f0_dir ./phrase_dir/phrase_f0_dct/f0"+
			" --out_file ./train_dev_data_vector/train_data_f0_vector_phrase"+
			" --add_index_prefix 0")
		os.system("python ../../decision_tree/data_preprocessing.py"+
			" --mode map_to_new_f0_vector"+
			" --map_file ./train_dev_data_vector/train_data/syllable_map"+
			" --f0_dir ./phrase_dir/phrase_f0_dct/res"+
			" --out_file ./train_dev_data_vector/train_data_f0_vector_res"+
			" --add_index_prefix 0")
		if args.train_small==1:
			print(">>>>>>>>>>>>>>updating small train data<<<<<<<<<<<<<")
			os.system("python ../../decision_tree/data_preprocessing.py"+
				" --mode map_to_new_f0_vector"+
				" --map_file ./train_dev_data_vector/train_data/syllable_map_small"+
				" --f0_dir ./phrase_dir/phrase_f0_dct/f0"+
				" --out_file ./train_dev_data_vector/train_data_f0_vector_phrase_small"+
				" --add_index_prefix 0")
			os.system("python ../../decision_tree/data_preprocessing.py"+
				" --mode map_to_new_f0_vector"+
				" --map_file ./train_dev_data_vector/train_data/syllable_map_small"+
				" --f0_dir ./phrase_dir/phrase_f0_dct/res"+
				" --out_file ./train_dev_data_vector/train_data_f0_vector_res_small"+
				" --add_index_prefix 0")

		print(">>>>>>>>>>>>>>updating dev data<<<<<<<<<<<<<")
		os.system("python ../../decision_tree/data_preprocessing.py"+
			" --mode map_to_new_f0_vector"+
			" --map_file ./train_dev_data_vector/dev_data/syllable_map"+
			" --f0_dir ./phrase_dir/phrase_f0_dct/f0"+
			" --out_file ./train_dev_data_vector/dev_data_f0_vector_phrase"+
			" --add_index_prefix 0")
		os.system("python ../../decision_tree/data_preprocessing.py"+
			" --mode map_to_new_f0_vector"+
			" --map_file ./train_dev_data_vector/dev_data/syllable_map"+
			" --f0_dir ./phrase_dir/phrase_f0_dct/res"+
			" --out_file ./train_dev_data_vector/dev_data_f0_vector_res"+
			" --add_index_prefix 0")
		if args.dev_small==1:
			print(">>>>>>>>>>>>>>updating small dev data<<<<<<<<<<<<<")
			os.system("python ../../decision_tree/data_preprocessing.py"+
				" --mode map_to_new_f0_vector"+
				" --map_file ./train_dev_data_vector/dev_data/syllable_map_small"+
				" --f0_dir ./phrase_dir/phrase_f0_dct/f0"+
				" --out_file ./train_dev_data_vector/dev_data_f0_vector_phrase_small"+
				" --add_index_prefix 0")
			os.system("python ../../decision_tree/data_preprocessing.py"+
				" --mode map_to_new_f0_vector"+
				" --map_file ./train_dev_data_vector/dev_data/syllable_map_small"+
				" --f0_dir ./phrase_dir/phrase_f0_dct/res"+
				" --out_file ./train_dev_data_vector/dev_data_f0_vector_res_small"+
				" --add_index_prefix 0")
