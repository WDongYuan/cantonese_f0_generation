import matplotlib
matplotlib.use('TkAgg')
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scipy.fftpack import idct, dct

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
	args = parser.parse_args()
	mode = args.mode

	if mode=="how_to_run":
		print("python data_preprocessing.py"+
			" --mode map_to_new_f0_vector"+
			" --map_file ./train_dev_data_vector/train_data/syllable_map"+
			" --f0_dir ../mandarine/gen_f0/f0_in_file"+
			" --out_file ./train_data_f0_vector_phrase"+
			" --add_index_prefix 0"+
			" --suffix .feats")
		print("python data_preprocessing.py"+
			" --mode map_to_new_phrase_f0_vector"+
			" --map_file ./train_dev_data_vector/train_data/syllable_map"+
			" --f0_dir ../mandarine/gen_f0/f0_in_file"+
			" --out_file ./train_data_f0_vector_phrase"+
			" --out_phrase_map ./phrase_map"
			" --add_index_prefix 0")
		print("python data_preprocessing.py"+
			" --mode dir_add/dir_minus"+
			" --dir1 "+
			" --dir2 "+
			" --out_dir")
		print("python data_preprocessing.py"+
			" --mode dir_rmse"+
			" --dir1 "+
			" --dir2 ")
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
		out_dir = args.out_dir
		file_list = os.listdir(dir1)
		for file in file_list:
			a1 = np.loadtxt(dir1+"/"+file,delimiter=" ")
			a2 = np.loadtxt(dir2+"/"+file,delimiter=" ")
			a3 = a1 + a2
			np.savetxt(out_dir+"/"+file,a3,delimiter=" ")

	elif mode=="dir_minus":
		dir1 = args.dir1
		dir2 = args.dir2
		out_dir = args.out_dir
		file_list = os.listdir(dir1)
		for file in file_list:
			a1 = np.loadtxt(dir1+"/"+file,delimiter=" ")
			a2 = np.loadtxt(dir2+"/"+file,delimiter=" ")
			a3 = a1 - a2
			np.savetxt(out_dir+"/"+file,a3,delimiter=" ")
	elif mode=="dir_rmse":
		dir1 = args.dir1
		dir2 = args.dir2
		file_list = os.listdir(dir1)
		l1 = []
		l2 = []
		for file in file_list:
			l1.append(np.loadtxt(dir1+"/"+file,delimiter=" "))
			l2.append(np.loadtxt(dir2+"/"+file,delimiter=" "))
		a1 = np.vstack(l1)
		a2 = np.vstack(l2)
		rmse = np.sqrt((np.square(a1-a2)).mean(axis=1)).mean()
		print("rmse: "+str(rmse))

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
					tmp_arr = (idct(tmp_arr)/(len(tmp_arr))).reshape((syl_num,10))
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














