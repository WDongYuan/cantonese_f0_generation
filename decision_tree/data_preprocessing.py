# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('TkAgg')
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sets import Set
from scipy.fftpack import idct, dct

def IsSetOfNumber(my_set):
	for val in my_set:
		try:
			float(val)
			return True
		except ValueError:
			return False
def CreateFeatureDesc(train_feat,feature_file,desc_file):
	known_feature = {}
	# if os.path.isfile(my_desc_file):
	# 	with open(my_desc_file) as f:
	# 		for line in f:
	# 			line = line.strip()
	# 			arr = line.split(" ")
	# 			known_feature[arr[0]] = " ".join(arr[1:len(arr)])

	desc_token = open(desc_file,"w+")
	desc_token.write("((dct_coef vector)\n")

	feature_list = []
	with open(feature_file) as f:
		for line in f:
			line = line.strip()
			feature_list.append([line,Set()])
	# print(feature_list)


	with open(train_feat) as f:
		for line in f:
			line = line.strip()
			arr = line.split(" ")
			arr = arr[1:len(arr)]
			# print(len(arr))
			# print(len(feature_list))
			for i in range(len(feature_list)):
				feature_list[i][1].add(arr[i])
				# print(feature_list[i][0])

	for feature_name,val_set in feature_list:
		desc_token.write("("+feature_name)
		if feature_name in known_feature:
			desc_token.write(" "+known_feature[feature_name])
		elif IsSetOfNumber(val_set) and len(val_set)>8:
			desc_token.write(" float")
		else:
			for val in val_set:
				desc_token.write(" "+val)
		desc_token.write(")\n")
	desc_token.write(")")

def ReadWordTxtDoneData(file):
	dic = {}
	with open(file) as f:
		for line in f:
			line = line.strip().split(" ")
			word = line[2].decode("utf-8")[1:-2]
			dic[line[1]] = word
	return dic
def normalize(arr):
	mean = arr.mean(axis=1).reshape((-1,1))
	std = (arr.std(axis=1)+0.000001).reshape((-1,1))
	norm = (arr-mean)/std
	return norm
if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', dest='mode')
	parser.add_argument("--map_file",dest="map_file")
	parser.add_argument("--f0_dir",dest="f0_dir")
	parser.add_argument("--in_file",dest="in_file")
	parser.add_argument("--out_file",dest="out_file")
	parser.add_argument("--out_phrase_map",dest="out_phrase_map")
	parser.add_argument("--add_index_prefix",dest="add_index_prefix",type=int)
	parser.add_argument("--dir1",dest="dir1")
	parser.add_argument("--dir2",dest="dir2")
	parser.add_argument("--out_dir",dest="out_dir")
	parser.add_argument("--phrase_syl_dir",dest="phrase_syl_dir")
	parser.add_argument("--dct_dir",dest="dct_dir")
	parser.add_argument("--suffix",dest="suffix",default="")
	parser.add_argument("--syllable_index",dest="syllable_index",type=int)
	parser.add_argument("--begin1",dest="begin1",type=int,default=0)
	parser.add_argument("--begin2",dest="begin2",type=int,default=0)
	parser.add_argument("--word_txt_done_data",dest="word_txt_done_data")
	parser.add_argument("--vocab_size",dest="vocab_size",type=int)
	parser.add_argument("--norm_flag",dest="norm_flag",type=int,default=0)
	parser.add_argument("--syl_in_file",dest="syl_in_file")
	parser.add_argument("--f0_in_file",dest="f0_in_file")
	parser.add_argument("--feat_in_file",dest="feat_in_file")
	parser.add_argument("--train_ratio",dest="train_ratio",type=float)
	parser.add_argument("--dev_ratio",dest="dev_ratio",type=float,default=0)
	parser.add_argument("--feature_file",dest="feature_file")
	parser.add_argument("--train_test_file_list",dest="train_test_file_list")
	args = parser.parse_args()
	mode = args.mode

	if mode=="how_to_run":
		print("python data_preprocessing.py"+
			" --mode map_to_new_f0_vector"+
			" --map_file ./train_dev_data_vector/train_data/syllable_map"+
			" --f0_dir ../mandarine/gen_f0/f0_in_file"+
			" --out_file ./train_data_f0_vector_phrase"+
			" --add_index_prefix 0"+
			" --suffix(optional default is \"\") .feats")
		print("python data_preprocessing.py"+
			" --mode map_to_new_phrase_f0_vector"+
			" --map_file ./train_dev_data_vector/train_data/syllable_map"+
			" --f0_dir ../mandarine/gen_f0/f0_in_file"+
			" --out_file ./train_data_f0_vector_phrase"+
			" --out_phrase_map ./phrase_map"
			" --add_index_prefix 0")
		print("python data_preprocessing.py"+
			" --mode dir_plus/dir_minus"+
			" --dir1 "+
			" --dir2 "+
			" --begin1(begin index in dir1 files) "+
			" --begin2(begin index in dir2 files) "+
			" --out_dir")
		print("python data_preprocessing.py"+
			" --mode dir_norm"+
			" --dir1 "+
			" --begin1(begin index in dir1 files) "+
			" --out_dir")
		print("python data_preprocessing.py"+
			" --mode dir_rmse"+
			" --dir1 "+
			" --dir2 "+
			" --begin1(begin index in dir1 files) "+
			" --begin2(begin index in dir2 files) ")
		print("python data_preprocessing.py"+
			" --mode put_phrase_prediction_in_file"+
			" --map_file "+
			" --in_file "+
			" --out_dir ")
		print("python data_preprocessing.py"+
			" --mode idct_phrase_f0_dir"+
			" --phrase_syl_dir "+
			" --dct_dir "+
			" --out_dir ")
		print("python data_preprocessing.py"+
			" --mode remove_tone"+
			" --in_file "+
			" --syllable_index "+
			" --out_file ")
		print("python data_preprocessing.py"+
			" --mode generate_word_pattern"+
			" --word_txt_done_data ../mandarine/txt.done.data-all"+
			" --f0_dir ../mandarine/gen_f0/f0_in_file"+
			" --map_file ../mandarine/gen_f0/train_dev_data_vector/train_data/syllable_map"+
			" --vocab_size 3000"+
			" --norm_flag 1"
			" --out_dir")
		print("python data_preprocessing.py"+
			" --mode create_train_dev_vector_data"+
			" --syl_in_file "+
			" --f0_in_file "+
			" --feat_in_file "+
			" --train_test_file_list"+
			" --out_dir")
		print("python data_preprocessing.py"+
			" --mode create_general_train_dev_vector_data"+
			" --f0_in_file "+
			" --feat_in_file "+
			" --train_test_file_list"+
			" --out_dir")
		print("python data_preprocessing.py"+
			" --mode create_train_test_file_list"+
			" --f0_in_file "+
			" --train_ratio "+
			" --dev_ratio(option) "+
			" --out_file")
		print("python data_preprocessing.py"+
			" --mode create_feature_desc"+
			" --in_file(train feature file) "+
			" --feature_file(the feature list) "+
			" --out_file")

	elif mode=="create_train_test_file_list":
		out_file = args.out_file
		f0_in_file = args.f0_in_file
		train_ratio = args.train_ratio
		dev_ratio = max(0,args.dev_ratio)

		file_list = os.listdir(f0_in_file)
		file_list = [file for file in file_list if "data" in file]
		random.shuffle(file_list)
		train_list = file_list[0:int(train_ratio*len(file_list))]
		test_list = None
		if dev_ratio==0:
			test_list = file_list[int(train_ratio*len(file_list)):]
		else:
			test_list = file_list[int(train_ratio*len(file_list)):int(train_ratio*len(file_list))+int(dev_ratio*len(file_list))]

		train_list = sorted(train_list)
		test_list = sorted(test_list)
		with open(out_file,"w+") as f:
			f.write(",".join(train_list)+"\n")
			f.write(",".join(test_list)+"\n")

	elif mode=="create_train_dev_vector_data":
		out_dir = args.out_dir
		syl_in_file = args.syl_in_file
		f0_in_file = args.f0_in_file
		feat_in_file = args.feat_in_file
		train_test_file_list = args.train_test_file_list
		# train_ratio = args.train_ratio
		# dev_ratio = max(0,args.dev_ratio)
		os.system("mkdir "+out_dir)

		# file_list = os.listdir(f0_in_file)
		# file_list = [file for file in file_list if "data" in file]
		# random.shuffle(file_list)
		# train_list = file_list[0:int(train_ratio*len(file_list))]
		# test_list = None
		# if dev_ratio==0:
		# 	test_list = file_list[int(train_ratio*len(file_list)):]
		# else:
		# 	test_list = file_list[int(train_ratio*len(file_list)):int(train_ratio*len(file_list))+int(dev_ratio*len(file_list))]
		train_list = None
		test_list = None
		with open(train_test_file_list) as f:
			train_list = f.readline().strip().split(",")
			test_list = f.readline().strip().split(",")
			test_list = [tmp_test_file for tmp_test_file in test_list if tmp_test_file!=""]

		train_list = sorted(train_list)
		test_list = sorted(test_list)
		# print(train_list)

		os.system("mkdir "+out_dir+"/train_data")
		train_feat = out_dir+"/train_data/train_feat"
		train_f0 = out_dir+"/train_data/train_f0"
		train_map = out_dir+"/train_data/train_syllable_map"
		with open(train_feat,"w+") as feat_file, open(train_f0,"w+") as f0_file, open(train_map,"w+") as map_file:
			feat_l = []
			map_l = []
			f0_l = []
			for data_name in train_list:
				with open(f0_in_file+"/"+data_name) as f:
					for line in f:
						f0_l.append(line.strip())
				with open(feat_in_file+"/"+data_name+".feats") as f:
					for line in f:
						feat_l.append(str(len(feat_l))+" "+line.strip())
				with open(syl_in_file+"/"+data_name) as f:
					for line in f:
						map_l.append(data_name+" "+line.strip())
			feat_file.write("\n".join(feat_l)+"\n")
			f0_file.write("\n".join(f0_l)+"\n")
			map_file.write("\n".join(map_l)+"\n")


		os.system("mkdir "+out_dir+"/test_data")
		test_feat = out_dir+"/test_data/test_feat"
		test_f0 = out_dir+"/test_data/test_f0"
		test_map = out_dir+"/test_data/test_syllable_map"
		with open(test_feat,"w+") as feat_file, open(test_f0,"w+") as f0_file, open(test_map,"w+") as map_file:
			feat_l = []
			map_l = []
			f0_l = []
			for data_name in test_list:
				with open(f0_in_file+"/"+data_name) as f:
					for line in f:
						f0_l.append(line.strip())
				with open(feat_in_file+"/"+data_name+".feats") as f:
					for line in f:
						feat_l.append(str(len(feat_l))+" "+line.strip())
				with open(syl_in_file+"/"+data_name) as f:
					for line in f:
						map_l.append(data_name+" "+line.strip())
			feat_file.write("\n".join(feat_l)+"\n")
			f0_file.write("\n".join(f0_l)+"\n")
			map_file.write("\n".join(map_l)+"\n")

	elif mode=="create_general_train_dev_vector_data":
		out_dir = args.out_dir
		f0_in_file = args.f0_in_file
		feat_in_file = args.feat_in_file
		train_test_file_list = args.train_test_file_list
		os.system("mkdir "+out_dir)

		train_list = None
		test_list = None
		with open(train_test_file_list) as f:
			train_list = f.readline().strip().split(",")
			test_list = f.readline().strip().split(",")


		train_list = sorted(train_list)
		test_list = sorted(test_list)
		# print(train_list)

		os.system("mkdir "+out_dir+"/train_data")
		train_feat = out_dir+"/train_data/train_feat"
		train_f0 = out_dir+"/train_data/train_f0"
		train_map = out_dir+"/train_data/train_syllable_map"
		with open(train_feat,"w+") as feat_file, open(train_f0,"w+") as f0_file, open(train_map,"w+") as map_file:
			feat_l = []
			map_l = []
			f0_l = []
			for data_name in train_list:
				counter = 0
				with open(f0_in_file+"/"+data_name) as f:
					for line in f:
						f0_l.append(line.strip())
						counter += 1
				with open(feat_in_file+"/"+data_name+".feats") as f:
					for line in f:
						feat_l.append(str(len(feat_l))+" "+line.strip())
				for i in range(counter):
					map_l.append(data_name+" "+str(i))
			feat_file.write("\n".join(feat_l)+"\n")
			f0_file.write("\n".join(f0_l)+"\n")
			map_file.write("\n".join(map_l)+"\n")


		os.system("mkdir "+out_dir+"/test_data")
		test_feat = out_dir+"/test_data/test_feat"
		test_f0 = out_dir+"/test_data/test_f0"
		test_map = out_dir+"/test_data/test_syllable_map"
		with open(test_feat,"w+") as feat_file, open(test_f0,"w+") as f0_file, open(test_map,"w+") as map_file:
			feat_l = []
			map_l = []
			f0_l = []
			for data_name in test_list:
				counter = 0
				with open(f0_in_file+"/"+data_name) as f:
					for line in f:
						f0_l.append(line.strip())
						counter += 1
				with open(feat_in_file+"/"+data_name+".feats") as f:
					for line in f:
						feat_l.append(str(len(feat_l))+" "+line.strip())
				for i in range(counter):
					map_l.append(data_name+" "+str(i))
			feat_file.write("\n".join(feat_l)+"\n")
			f0_file.write("\n".join(f0_l)+"\n")
			map_file.write("\n".join(map_l)+"\n")

	elif mode=="create_feature_desc":
		train_feat = args.in_file
		feature_file = args.feature_file
		out_file = args.out_file
		CreateFeatureDesc(train_feat,feature_file,out_file)


	elif mode=="map_to_new_f0_vector":
		map_file = args.map_file
		f0_dir = args.f0_dir
		suffix = args.suffix
		out_file = args.out_file
		add_index_prefix = args.add_index_prefix
		file_list = [file for file in os.listdir(f0_dir) if "data" in file]

		with open(map_file) as mapf, open(out_file,"w+") as outf:
			map_data = mapf.readlines()
			map_data = [line.strip().split(" ") for line in map_data]
			row_p = 0
			while row_p<len(map_data):
				data_name = map_data[row_p][0]
				with open(f0_dir+"/"+data_name+suffix) as tmpf:
					tmp_data = tmpf.readlines()
					if add_index_prefix==1:
						tmp_data = [str(row_p+i)+" "+tmp_data[i] for i in range(len(tmp_data))]
					file_len = len(tmp_data)
					outf.writelines(tmp_data)
					row_p += file_len
					if row_p<len(map_data):
						assert map_data[row_p][0]!=data_name and map_data[row_p-1][0]==data_name
	elif mode=="map_to_new_phrase_f0_vector":
		map_file = args.map_file
		f0_dir = args.f0_dir
		out_file = args.out_file
		phrase_map = args.out_phrase_map
		add_index_prefix = args.add_index_prefix

		file_list = [file for file in os.listdir(f0_dir) if "data" in file]

		with open(map_file) as mapf, open(out_file,"w+") as outf, open(phrase_map,"w+") as out_mapf:
			map_data = mapf.readlines()
			map_data = [line.strip().split(" ") for line in map_data]
			##index in syllable map file
			row_p = 0

			##count the phrase number so far
			row_count = 0
			while row_p<len(map_data):
				data_name = map_data[row_p][0]
				with open(f0_dir+"/"+data_name) as tmpf:
					tmp_data = tmpf.readlines()
					if add_index_prefix==1:
						tmp_data = [str(row_count+i)+" "+tmp_data[i] for i in range(len(tmp_data))]
					outf.writelines(tmp_data)
					row_count += len(tmp_data)
					for i in range(len(tmp_data)):
						out_mapf.write(data_name+"\n")
					while row_p<len(map_data) and map_data[row_p][0] == data_name:
						row_p += 1

	elif mode=="dir_plus":
		dir1 = args.dir1
		dir2 = args.dir2
		begin1 = args.begin1
		begin2 = args.begin2
		out_dir = args.out_dir
		os.system("mkdir "+out_dir)
		file_list = os.listdir(dir1)
		for file in file_list:
			a1 = np.loadtxt(dir1+"/"+file,delimiter=" ",dtype=str)
			a1 = a1.reshape((a1.shape[0],-1))
			a2 = np.loadtxt(dir2+"/"+file,delimiter=" ",dtype=str)
			a2 = a2.reshape((a2.shape[0],-1))
			a3 = a1[:,begin1:].astype(np.float32) + a2[:,begin2:].astype(np.float32)
			np.savetxt(out_dir+"/"+file,a3,delimiter=" ",fmt="%.5f")

	elif mode=="dir_minus":
		dir1 = args.dir1
		dir2 = args.dir2
		begin1 = args.begin1
		begin2 = args.begin2
		out_dir = args.out_dir
		os.system("mkdir "+out_dir)
		file_list = os.listdir(dir1)
		for file in file_list:
			a1 = np.loadtxt(dir1+"/"+file,delimiter=" ",dtype=str)
			a1 = a1.reshape((a1.shape[0],-1))
			a2 = np.loadtxt(dir2+"/"+file,delimiter=" ",dtype=str)
			a2 = a2.reshape((a2.shape[0],-1))
			a3 = a1[:,begin1:].astype(np.float32) - a2[:,begin2:].astype(np.float32)
			np.savetxt(out_dir+"/"+file,a3,delimiter=" ",fmt="%.5f")
	elif mode=="dir_rmse":
		dir1 = args.dir1
		dir2 = args.dir2
		begin1 = args.begin1
		begin2 = args.begin2
		file_list = os.listdir(dir1)
		l1 = []
		l2 = []
		for file in file_list:
			l1.append(np.loadtxt(dir1+"/"+file,delimiter=" ",dtype=str)[:,begin1:].astype(np.float32))
			l2.append(np.loadtxt(dir2+"/"+file,delimiter=" ",dtype=str)[:,begin2:].astype(np.float32))
		a1 = np.vstack(l1)
		a2 = np.vstack(l2)
		rmse = np.sqrt((np.square(a1-a2)).mean(axis=1)).mean()
		print("rmse: "+str(rmse))

	elif mode=="dir_norm":
		dir1 = args.dir1
		begin1 = args.begin1
		out_dir = args.out_dir
		os.system("mkdir "+out_dir)
		file_list = os.listdir(dir1)
		for file in file_list:
			a = np.loadtxt(dir1+"/"+file,delimiter=" ",dtype=str)[:,begin1:].astype(np.float32)
			np.savetxt(out_dir+"/"+file,normalize(a),delimiter=" ",fmt="%.5f")

	elif mode=="put_phrase_prediction_in_file":
		map_file = args.map_file
		in_file = args.in_file
		out_dir = args.out_dir
		os.system("mkdir "+out_dir)

		dic = {}
		with open(map_file) as mapf, open(in_file) as inf:
			for line in mapf:
				data_name = line.strip()
				if data_name not in dic:
					dic[data_name] = []
				dic[data_name].append(inf.readline())
		for data_name,lines in dic.items():
			with open(out_dir+"/"+data_name,"w+") as f:
				f.writelines(lines)
	elif mode=="idct_phrase_f0_dir":
		phrase_dir = args.phrase_syl_dir
		dct_dir = args.dct_dir
		out_dir = args.out_dir
		os.system("mkdir "+out_dir)
		file_list = os.listdir(dct_dir)
		for file in file_list:
			dct_arr = np.loadtxt(dct_dir+"/"+file,delimiter=" ")
			dct_arr = dct_arr.reshape((-1,dct_arr.shape[-1]))
			with open(phrase_dir+"/"+file) as pf, open(out_dir+"/"+file,"w+") as outf:
				lines = pf.readlines()
				for i in range(len(lines)):
					syl_num = len(lines[i].strip().split(" "))
					tmp_arr = np.zeros((syl_num*10,))
					tmp_arr[0:dct_arr.shape[1]] = dct_arr[i]
					tmp_arr = (idct(tmp_arr)/(2*len(tmp_arr))).reshape((syl_num,10))
					for j in range(syl_num):
						outf.write(" ".join(tmp_arr[j].astype(np.str).tolist())+"\n")
	elif mode=="remove_tone":
		in_file = args.in_file
		syl_idx = args.syllable_index
		out_file = args.out_file
		with open(in_file) as inf, open(out_file,"w+") as outf:
			for line in inf:
				line = line.split(" ")
				line[syl_idx] = line[syl_idx][0:-1]
				outf.write(" ".join(line))

	elif mode=="generate_word_pattern":
		f0_dir = args.f0_dir
		txt = args.word_txt_done_data
		train_map = args.map_file
		vocab_size = args.vocab_size
		out_dir = args.out_dir
		# for data_name,word in data_dic.items():
		# 	print(data_name),
		# 	print(word)

		data_dic = ReadWordTxtDoneData(txt)

		data_set = Set([])
		with open(train_map) as f:
			for line in f:
				data_set.add(line.split(" ")[0])

		##word_f0_dic["word"] = np.array of f0
		word_f0_dic = {}
		for data_name in data_set:
			word = data_dic[data_name]
			f0 = np.loadtxt(f0_dir+"/"+data_name,delimiter=" ")
			assert len(word)==len(f0)
			for i in range(len(word)):
				w = word[i].encode("utf-8")
				if w not in word_f0_dic:
					word_f0_dic[w] = []
				word_f0_dic[w].append(f0[i])
		for data_name in word_f0_dic.keys():
			if args.norm_flag==1:
				word_f0_dic[data_name] = normalize(np.hstack(word_f0_dic[data_name]).reshape((-1,10)))
			else:
				word_f0_dic[data_name] = np.hstack(word_f0_dic[data_name]).reshape((-1,10))

		##truncate the dictionary
		word_list = [[data_name,f0] for data_name,f0 in word_f0_dic.items()]
		word_list = sorted(word_list,key=lambda tup:tup[1].shape[0],reverse=True)
		unk_word = ["UNK",[]]
		for i in range(vocab_size,len(word_list)):
			unk_word[1].append(word_list[i][1])
		unk_word[1] = np.vstack(unk_word[1])
		word_list = [tup for tup in word_list[0:vocab_size]]
		word_list.append(unk_word)

		##mean f0 for every word
		word_mean_dic = {}
		for tup in word_list:
			word_mean_dic[tup[0]] = np.around(tup[1].mean(axis=0),decimals=5)

		##apply mean in every f0 file
		# print(word_mean_dic["姜"])
		os.system("mkdir "+out_dir)
		for data_name,word in data_dic.items():
			with open(out_dir+"/"+data_name,"w+") as f:
				for w in word:
					w = w.encode("utf-8")
					if w not in word_mean_dic:
						f.write(" ".join(word_mean_dic["UNK"].astype(np.str).tolist())+"\n")
					else:
						f.write(" ".join(word_mean_dic[w].astype(np.str).tolist())+"\n")
		
















