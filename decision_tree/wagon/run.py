import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import sys
from scipy.fftpack import idct, dct
import numpy as np
import random
import argparse
import re
import math
import warnings
warnings.filterwarnings('error')
from data_processing import *


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', dest='mode')

	parser.add_argument('--desc_file', dest='desc_file')

	parser.add_argument('--train_file', dest='train_file')
	parser.add_argument('--train_dir', dest='train_dir')
	parser.add_argument('--train_label', dest='train_label')

	parser.add_argument('--test_file', dest='test_file')
	parser.add_argument('--test_dir', dest='test_dir')
	parser.add_argument('--test_label', dest='test_label')

	parser.add_argument('--f0_dir', dest='f0_dir')
	parser.add_argument('--predict_dir', dest='predict_dir')
	parser.add_argument('--image_number', dest='image_number',type=int,default=1)
	parser.add_argument('--sample_index', dest='sample_index',type=int,default=1)

	parser.add_argument('--file1', dest='file1')
	parser.add_argument('--file2', dest='file2')

	parser.add_argument('--f0_file_map', dest='f0_file_map')
	parser.add_argument('--f0_val', dest='f0_val')
	parser.add_argument('--out_dir', dest='out_dir')

	parser.add_argument('--map_file', dest='map_file')
	parser.add_argument('--in_dir', dest='in_dir')
	parser.add_argument('--out_file', dest='out_file')


	args = parser.parse_args()
	mode = args.mode

	if mode=="how_to_run":
		# print("python run.py build_tree ../train_dev_data/train_data/dct_0(train file) ../train_dev_data/feature_desc(description file)")
		# print("python run.py predict ../train_dev_data/dev_data/dct_0 ../train_dev_data/train_data/feature_desc ./tree/dct_0_tree")
		print("#################################################################")
		print("python run.py --mode train_predict_dir"+
			" --train_dir ../train_dev_data/train_data"+
			" --test_dir ../train_dev_data/dev_data"+
			" --desc_file ../train_dev_data/feature_desc"+
			" --out_dir ./predict")
		print("#################################################################")
		print("python run.py --mode train_predict_individual_val"+
			" --train_file ../train_dev_data/train_data/dct_0"+
			" --test_file ../train_dev_data/dev_data/dct_0"+
			" --desc_file ../train_dev_data/feature_desc"+
			" --out_dir ./predict")
		print("#################################################################")
		print("python run.py --mode train_predict_vector"+
			" --desc_file ../train_dev_data_vector/feature_desc"+
			" --train_file ../train_dev_data_vector/train_data/dct_0"+
			" --train_label ../train_dev_data_vector/train_data_f0_vector"+
			" --test_file ../train_dev_data_vector/dev_data/dct_0"+
			" --test_label ../train_dev_data_vector/dev_data_f0_vector"+
			" --out_dir ./vector_predict")
		print("#################################################################")
		print("python run.py --mode train_predict_mean_std"+
			" --desc_file ../train_dev_data_vector/feature_desc"+
			" --train_file ../train_dev_data_vector/train_data/dct_0"+
			" --train_label ../train_dev_data_vector/train_mean_std"+
			" --test_file ../train_dev_data_vector/dev_data/dct_0"+
			" --test_label ../train_dev_data_vector/dev_mean_std")
		print("#################################################################")
		# print("python run.py plot_f0 ../train_dev_data/dev_data ./predict 10(index of the sample to plot)")
		print("python run.py --mode save_f0"+
			" --f0_dir ../train_dev_data/dev_data"+
			" --predict_dir ./predict"+
			" --image_number 10")
		print("#################################################################")
		print("python run.py --mode predict_statistic"+
			" --file1 ./vector_predict/dct_0_val_unnorm"+
			" --file2 ../train_dev_data_vector/dev_data_f0_vector")
		print("#################################################################")
		print("python run.py --mode put_back_f0_in_file"+
			" --f0_file_map ../train_dev_data_vector/dev_data/syllable_map"+
			" --f0_val ./vector_predict/dct_0_val_unnorm"+
			" --out_dir ./predict_f0_in_file")
		print("#################################################################")
		print("python run.py --mode update_feature_using_map"+
			" --in_dir ../../dumpfeats/concat_feature"+
			" --map_file ../../decision_tree/train_dev_data_vector/dev_data/syllable_map"+
			" --out_file ../train_dev_data_vector/dev_data/dct_0")
		print("#################################################################")
		exit()

	if mode=="build_tree":
		os.system("mkdir tree")
		# train_file = sys.argv[2]
		# desc_file = sys.argv[3]
		train_file = args.train_file
		desc_file = args.desc_file
		file_name = train_file.split("/")[-1]
		ESTDIR = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools"
		os.system(ESTDIR+"/bin/wagon -data "+train_file+" -desc "+desc_file+" -stop 10 -output ./tree/"+file_name+"_tree")
	if mode=="predict":
		os.system("mkdir predict")
		# test_file = sys.argv[2]
		# desc_file = sys.argv[3]
		# tree_file = sys.argv[4]
		test_file = args.test_file
		desc_file = args.desc_file
		tree_file = args.tree_file
		file_name = test_file.split("/")[-1]
		ESTDIR = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools"
		os.system(ESTDIR+"/main/wagon_test -desc "+desc_file+" -data "+test_file+" -tree "+tree_file+" -o "+"./predict/"+file_name+" -predict_val")
		os.system(ESTDIR+"/main/wagon_test -desc "+desc_file+" -data "+test_file+" -tree "+tree_file+" -o "+"./predict/"+file_name+"_error ")
	if mode=="train_predict_dir":
		# train_dir = sys.argv[2]
		# test_dir = sys.argv[3]
		# desc_file = sys.argv[4]
		train_dir = args.train_dir
		test_dir = args.test_dir
		desc_file = args.desc_file
		out_dir = args.out_dir
		ESTDIR = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools"

		os.system("mkdir "+out_dir)
		os.system("rm "+out_dir+"/train_info")
		os.system("mkdir "+out_dir+"/tree")

		train_file_list = os.listdir(train_dir)
		for train_file in train_file_list:
			if ".DS_Store" == train_file:
				continue
			print("Processing "+train_file)

			os.system("nohup "+ESTDIR+"/bin/wagon"+\
				" -data "+train_dir+"/"+train_file+\
				" -desc "+desc_file+\
				" -stop 10"+\
				" -output "+out_dir+"/tree/"+train_file+"_tree"+
				" >> "+out_dir+"/train_info 2>&1")

			os.system(ESTDIR+"/main/wagon_test"+\
				" -desc "+desc_file+\
				" -data "+test_dir+"/"+train_file+\
				" -tree "+out_dir+"/tree/"+train_file+"_tree"+\
				" -o "+out_dir+"/"+train_file+\
				" -predict_val")

			os.system(ESTDIR+"/main/wagon_test"+\
				" -desc "+desc_file+\
				" -data "+test_dir+"/"+train_file+\
				" -tree "+out_dir+"/tree/"+train_file+"_tree"+\
				" -o "+out_dir+"/"+train_file+"_error")

		##Collect the error data and put them in the file predict_error
		error_file_list = [file_name for file_name in os.listdir(out_dir) if "error" in file_name]
		# print(error_file_list)
		error_info = []
		for file_name in error_file_list:
			file = open(out_dir+"/"+file_name)
			error_info.append(file_name+" "+file.readline().strip())
			file.close()
			os.system("rm "+out_dir+"/"+file_name)
		with open(out_dir+"/predict_info","w+") as f:
			for tmp_s in error_info:
				f.write(tmp_s+"\n")

	if mode=="train_predict_individual_val":
		train_file = args.train_file
		test_file = args.test_file
		desc_file = args.desc_file
		out_dir = args.out_dir
		ESTDIR = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools"

		os.system("mkdir "+out_dir)
		os.system("rm "+out_dir+"/train_info")
		os.system("mkdir "+out_dir+"/tree")

		print("Building decision tree")

		test_file_name = test_file.split("/")[-1]
		os.system("nohup "+ESTDIR+"/bin/wagon"+
			" -data "+train_file+
			" -desc "+desc_file+
			" -stop 10"+
			" -output "+out_dir+"/tree/train_tree"+
			" >> "+out_dir+"/train_info 2>&1")

		print("Predicting on dev data")
		os.system(ESTDIR+"/main/wagon_test"+
			" -desc "+desc_file+
			" -data "+test_file+
			" -tree "+out_dir+"/tree/train_tree"+
			" -o "+out_dir+"/"+test_file_name+
			" -predict_val")

		os.system(ESTDIR+"/main/wagon_test"+
			" -desc "+desc_file+
			" -data "+test_file+
			" -tree "+out_dir+"/tree/train_tree"+
			" -o "+out_dir+"/"+test_file_name+"_error")

	if mode=="train_predict_vector":
		os.system("mkdir dump")

		desc_file = args.desc_file
		train_file = args.train_file
		train_label_file = args.train_label
		dev_file = args.test_file
		dev_label_file = args.test_label
		out_dir = args.out_dir
		os.system("mkdir "+out_dir)

		train_file_name = train_file.split("/")[-1]

		ESTDIR = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools"

		##Conver the file into track file
		print("Converting track file for train data and test data...")
		os.system("cat "+train_label_file+" | "+ESTDIR+"/bin/ch_track -itype ascii -otype est_binary -s 0.005 -o ./dump/train_track_file")
		os.system("cat "+dev_label_file+" | "+ESTDIR+"/bin/ch_track -itype ascii -otype est_binary -s 0.005 -o ./dump/dev_track_file")

		print("Building the decision tree...")
		ESTDIR = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools"
		os.system("nohup "+ESTDIR+"/bin/wagon"+
			" -data "+train_file+
			" -desc "+desc_file+
			# " -test ./dump/dev_data/dct_0"+
			" -track ./dump/train_track_file"+
			" -stop 10"+
			" -output "+out_dir+"/"+train_file_name+"_tree"+
			"> "+out_dir+"/"+train_file_name+"_test_info")

		print("Predicting on the dev set...")
		dev_file_name = dev_file.split("/")[-1]
		os.system(ESTDIR+"/main/wagon_test"+
				" -desc "+desc_file+
				" -data "+dev_file+
				" -tree "+out_dir+"/"+train_file_name+"_tree"+
				" -track ./dump/dev_track_file"+
				" -o "+out_dir+"/"+dev_file_name+
				" -predict")

		##Transcript the Festival format into the readable format
		predict_val = []
		with open(out_dir+"/"+dev_file_name) as f:
			for line in f:
				line = line.strip()
				arr = re.split(r"\(|\)|\s",line)
				arr = [tmp_s for tmp_s in arr if tmp_s!=""]
				arr.pop(-1)
				arr = [arr[i] for i in range(len(arr)) if i%2==0]
				predict_val.append(arr)
		with open(out_dir+"/"+dev_file_name+"_val","w+") as f:
			for sample in predict_val:
				f.write(" ".join(sample)+"\n")

		os.system("rm -r dump")

	if mode=="plot_f0":
		# true_dir = sys.argv[2]
		# predict_dir = sys.argv[3]
		# sample_idx = int(sys.argv[4])
		true_dir = args.f0_dir
		predict_dir = args.predict_dir
		sample_idx = args.sample_index
		file_list = sorted([tmp_name for tmp_name in os.listdir(true_dir) if "dct" in tmp_name])
		true_dct = []
		predict_dct = []
		for file in file_list:
			true_dct.append(float(open(true_dir+"/"+file).readlines()[sample_idx].strip().split(" ")[0]))
			predict_dct.append(float(open(predict_dir+"/"+file).readlines()[sample_idx].strip()))
		true_dct = np.array(true_dct)
		predict_dct = np.array(predict_dct)

		true_dct = idct(true_dct)/(2*len(true_dct))
		predict_dct = idct(predict_dct)/(2*len(predict_dct))

		plt.plot(true_dct,label="true_dct")
		plt.plot(predict_dct,label="predict_dct")
		plt.legend()
		plt.show()
	if mode=="save_f0":
		# true_dir = sys.argv[2]
		# predict_dir = sys.argv[3]
		# img_num = int(sys.argv[4])
		true_dir = args.f0_dir
		predict_dir = args.predict_dir
		img_num = args.image_number
		file_list = sorted([tmp_name for tmp_name in os.listdir(true_dir) if "dct" in tmp_name])
		true_dct = []
		predict_dct = []
		for file in file_list:
			tmp_true_list = open(true_dir+"/"+file).readlines()
			tmp_predict_list = open(predict_dir+"/"+file).readlines()
			tmp_true_list = [float(line.strip().split(" ")[0]) for line in tmp_true_list]
			tmp_predict_list = [float(line.strip()) for line in tmp_predict_list]

			if len(true_dct)==0 and len(predict_dct)==0:
				true_dct = [[] for i in range(len(tmp_true_list))]
				predict_dct = [[] for i in range(len(tmp_predict_list))]

			for i in range(len(true_dct)):
				true_dct[i].append(tmp_true_list[i])
				predict_dct[i].append(tmp_predict_list[i])

		# os.system("mkdir saved_ori_f0")
		os.system("mkdir saved_idct_f0")
		idx = range(len(true_dct))
		random.shuffle(idx)
		idx = idx[0:img_num]
		for i in idx:
			tmp_true_dct = true_dct[i]
			tmp_predict_dct = predict_dct[i]
			tmp_true_dct = idct(tmp_true_dct)/(2*len(tmp_true_dct))
			tmp_predict_dct = idct(tmp_predict_dct)/(2*len(tmp_predict_dct))
			plt.plot(tmp_true_dct,label="true_f0")
			plt.plot(tmp_predict_dct,label="predict_f0")
			plt.legend()
			# plt.savefig("saved_ori_f0/"+str(i)+".png")
			plt.savefig("saved_idct_f0/"+str(i)+".png")
			plt.clf()

	if mode=="predict_statistic":
		file1 = args.file1
		file2 = args.file2

		arr1 = np.loadtxt(file1,delimiter=" ")
		arr2 = np.loadtxt(file2,delimiter=" ")

		print("RMSE:")
		print(np.sqrt(np.mean(np.square(arr1-arr2),axis=0)))
		print("Global RMSE:")
		print(np.sqrt(np.mean(np.square(arr1-arr2),axis=1)).mean())

		print("Correlation: ")
		cor = 0
		for row in range(arr1.shape[0]):
			try:
				tmp_cor = np.corrcoef(arr1[row,:],arr2[row,:])[0][1]
			except Exception as e:
				# print(e)
				continue
			# print(tmp_cor)
			cor += tmp_cor
		print(cor/arr1.shape[0])

		print("ABS ERROR:")
		print(np.mean(np.abs(arr1-arr2),axis=0))

	if mode=="put_back_f0_in_file":
		f0_file_map = args.f0_file_map
		f0_val = args.f0_val
		out_dir = args.out_dir
		os.system("rm -r "+out_dir)
		os.system("mkdir "+out_dir)
		cur_file = None
		with open(f0_file_map) as fmap:
			with open(f0_val) as fval:
				token = None
				map_line = fmap.readline().strip()
				val_line = fval.readline().strip()
				while map_line!="":
					file_name,syl = map_line.split(" ")
					if cur_file is None:
						cur_file = file_name
						token = open(out_dir+"/"+file_name,"w+")
					elif cur_file!=file_name:
						cur_file = file_name
						token.close()
						token = open(out_dir+"/"+file_name,"w+")
					token.write(syl+" "+val_line+"\n")
					map_line = fmap.readline().strip()
					val_line = fval.readline().strip()
				token.close()

	if mode=="train_predict_mean_std":
		os.system("mkdir dump")
		os.system("mkdir mean_std_tree")
		os.system("mkdir mean_std_predict")
		desc_file = args.desc_file
		train_file = args.train_file
		train_label_file = args.train_label
		dev_file = args.test_file
		dev_label_file = args.test_label

		train_file_name = train_file.split("/")[-1]

		ESTDIR = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools"

		##Conver the file into track file
		print("Converting track file for train data and test data...")
		os.system("cat "+train_label_file+" | "+ESTDIR+"/bin/ch_track -itype ascii -otype est_binary -s 0.005 -o ./dump/train_track_file")
		os.system("cat "+dev_label_file+" | "+ESTDIR+"/bin/ch_track -itype ascii -otype est_binary -s 0.005 -o ./dump/dev_track_file")

		print("Building the decision tree...")
		ESTDIR = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools"
		os.system("nohup "+ESTDIR+"/bin/wagon"+
			" -data "+train_file+
			" -desc "+desc_file+
			# " -test ./dump/dev_data/dct_0"+
			" -track ./dump/train_track_file"+
			" -stop 10"+
			" -output ./mean_std_tree/"+train_file_name+"_tree"+
			"> ./mean_std_predict/test_info")

		print("Predicting on the dev set...")
		dev_file_name = dev_file.split("/")[-1]
		os.system(ESTDIR+"/main/wagon_test"+
				" -desc "+desc_file+
				" -data "+dev_file+
				" -tree ./mean_std_tree/"+train_file_name+"_tree"+
				" -track ./dump/dev_track_file"+
				" -o "+"./mean_std_predict/"+dev_file_name+
				" -predict")

		predict_val = []
		with open("./mean_std_predict/"+dev_file_name) as f:
			for line in f:
				line = line.strip()
				arr = re.split(r"\(|\)|\s",line)
				arr = [tmp_s for tmp_s in arr if tmp_s!=""]
				arr.pop(-1)
				arr = [arr[i] for i in range(len(arr)) if i%2==0]
				predict_val.append(arr)
		with open("./mean_std_predict/"+dev_file_name+"_val","w+") as f:
			for sample in predict_val:
				f.write(" ".join(sample)+"\n")

		os.system("rm -r dump")

	if mode=="update_feature_using_map":
		##update the feature data after dump new features or concat some customized features
		# map_file = "../decision_tree/train_dev_data_vector/dev_data/syllable_map"
		map_file = args.map_file
		# indir = "../dumpfeats/concat_feature"
		indir = args.in_dir
		# outfile = "./outfile"
		outfile = args.out_file
		counter = 0.0
		with open(map_file) as mf, open(outfile,"w+") as of:
			data_name = ""
			file = None
			for line in mf:
				arr = line.strip().split(" ")
				name = arr[0]
				syl = arr[1]
				if name != data_name:
					data_name = name
					file = open(indir+"/"+data_name+".feats")
				new_feat = file.readline().strip()
				assert new_feat.split(" ")[0] in syl
				of.write(str(counter)+" "+new_feat+"\n")
				counter += 1
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print("you need to update the description file:")
		print("/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/gen_audio/src/decision_tree/train_dev_data_vector/feature_desc")
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")











