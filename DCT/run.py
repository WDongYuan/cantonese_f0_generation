#-*- coding: utf-8 -*-#
import matplotlib
matplotlib.use('TkAgg')

from scipy.fftpack import dct,idct
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
import sys
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import random
from scipy.fftpack import idct, dct
def ExtractF0(in_dir,out_dir):
	# os.system("mkdir f0_value")
	os.system("mkdir "+out_dir)
	os.system("export ESTDIR=/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools")
	# f0_path = "../cmu_yue_wdy_normal_build/f0"
	# f0_path = "../../jyutping_correction/cmu_yue_wdy_correction/f0"
	f0_path = in_dir
	estdir = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools"

	file_list = os.listdir(f0_path)
	for file in file_list:
		if "f0" not in file:
			continue
		# print(estdir+"/bin/ch_track f0/"+f0_path+"/"+file+" > f0_value/"+file.split(".")[1]+".out")
		os.system(estdir+"/bin/ch_track "+f0_path+"/"+file+" > "+out_dir+"/"+file.split(".")[0]+".f0")

def ExtractPhoneme(in_dir,out_dir):
	file_list = os.listdir(in_dir)
	for file in file_list:
		if "data" not in file or "lab" not in file:
			continue
		# print(estdir+"/bin/ch_track f0/"+f0_path+"/"+file+" > f0_value/"+file.split(".")[1]+".out")
		tmp_file = open(out_dir+"/"+file.split(".")[0]+".phoneme","w+")
		with open(in_dir+"/"+file) as f:
			for line in f:
				if "#" in line:
					continue
				line = line.strip()
				arr = line.split(" ")
				tmp_file.write(arr[0]+" "+arr[2]+"\n")
		tmp_file.close()

def SplitSyllable(s):
	l = []
	begin = 0
	for i in range(len(s)):
		if s[i].isdigit():
			l.append(s[begin:i+1])
			begin = i+1
	return l

def word_syllable_f0(text_syllable,syllable_f0):
	text_syl_pair = []
	with open(text_syllable) as f:
		cont = f.readlines()
		for i in range(len(cont)/4):
			data_name = cont[i*4].strip().split(".")[0]
			word = "".join(cont[i*4+1].strip().split(" ")).decode("utf-8")
			syl_list = cont[i*4+2].strip().split(" ")
			syl = []
			for tmp_syl in syl_list:
				if tmp_syl=="PUNCT":
					syl.append(tmp_syl)
				else:
					syl += SplitSyllable(tmp_syl)
			assert len(word)==len(syl)

			for i in range(len(word)):
				if syl[i] == "PUNCT":
					continue
				text_syl_pair.append([data_name,syl[i],word[i]])
	with open(syllable_f0) as f:
		cont = f.readlines()
		for i in range(len(cont)):
			line = cont[i].strip().split(",")
			assert "_".join(text_syl_pair[i][0:2])==line[0]
			text_syl_pair[i].append([float(val) for val in line[1:]])
	# for tup in text_syl_pair:
	# 	print(tup[0]+" "+tup[1]+" "+tup[2])
	# 	print(tup[3])
	return text_syl_pair


def RMSE(arr):
	return np.sqrt(np.mean(np.square(arr-arr.mean(axis=0)),axis=0))
def PlotFileF0(data_name):
	timeline = []
	f0_list = []
	time = 0

	tmp_f0_list = []
	tmp_timeline = []
	with open("./f0_value/"+data_name+".f0") as f:
		for line in f:
			line = line.strip()

			f0_value = float(line.split(" ")[0])
			if f0_value>0:
				tmp_f0_list.append(f0_value)
				tmp_timeline.append(time)
			elif f0_value<=0 and len(tmp_timeline)!=0:
				timeline.append(tmp_timeline)
				f0_list.append(tmp_f0_list)

				# print(timeline)

				tmp_timeline = []
				tmp_f0_list = []

			time += 5

	for i in range(len(timeline)):
		plt.plot(timeline[i],f0_list[i],'r-',lw=0.5)
		# plt.axvline(timeline[i][-1],linestyle="dotted")

	pho_timeline = []
	with open("./f0_value/"+data_name+".phoneme") as f:
		for line in f:
			line = line.strip()
			# pho_timeline.append(float(line.split(" ")[0])*1000)
			plt.axvline(float(line.split(" ")[0])*1000,linestyle="dotted",lw=0.5)

	# pho_y = [100 for i in range(len(pho_timeline))]

	# plt.plot(pho_timeline,pho_y,".")



	plt.ylabel("f0 value")
	plt.xlabel("time(ms)")

	# plt.show()
	os.system("mkdir f0_plot")
	plt.savefig("f0_plot/"+data_name+".jpg", dpi=1200)
	# plt.savefig("f0_plot/"+f0_filename+".jpg")
	print("Plot saved to f0_plot!")
	plt.clf()

def ReadTxtDoneData(path):
	dic = {}
	with open(path) as f:
		for line in f:
			line = line.strip()
			syllable_list = line.split("\"")[1].split(" ")[1:-1]
			syllable_list = [(syllable[0:-1],syllable[-1]) for syllable in syllable_list]

			data_name = line.split(" ")[1]

			dic[data_name] = syllable_list
	return dic
def AlignF0Syllable(phoneme_f0_list,syllable_list):
	syllable_f0_list = []
	# print(phoneme_f0_list)

	phoneme_index = 0
	tmp_syllable_f0 = [[],""]
	for syllable,tone in syllable_list:
		# print(syllable)

		while phoneme_f0_list[phoneme_index][1]=="pau" or phoneme_f0_list[phoneme_index][1]=="ssil":
			phoneme_index += 1

		while tmp_syllable_f0[1]!=syllable:
			tmp_syllable_f0[0] += phoneme_f0_list[phoneme_index][0]
			tmp_syllable_f0[1] += phoneme_f0_list[phoneme_index][1]
			phoneme_index += 1
			# print(tmp_syllable_f0)

		tmp_syllable_f0[1] += tone
		syllable_f0_list.append([tmp_syllable_f0[1],tmp_syllable_f0[0]])
		tmp_syllable_f0 = [[],""]
		# print(syllable_f0_list)

	##The format will be [syllable_label,f0_value]
	return syllable_f0_list

def AlignF0Phoneme(data_name,f0_dir):
	# print(data_name)
	timeline = []
	f0_list = []
	time = 0

	# f0_file = "./f0_value/"+data_name+".f0"
	f0_file = f0_dir+"/"+data_name+".f0"

	with open(f0_file) as f:
		for line in f:
			line = line.strip()

			f0_value = max(float(line.split(" ")[0]),0)
			# if f0_value == -1:
			# 	break

			f0_list.append(f0_value)
			timeline.append(time)
			time += 1
	# print(f0_list)
	# print(timeline)

	pre_end = 0
	phoneme_list = []

	##Align the f0 value and the phoneme
	phoneme_file = "./f0_value/"+data_name+".phoneme"
	with open(phoneme_file) as f:
		for line in f:
			line = line.strip()

			arr = line.split(" ")
			tmp_time = float(arr[0])*1000
			tmp_time = int(tmp_time/5)
			tmp_label = arr[1]

			phoneme_list.append([f0_list[pre_end:tmp_time+1],tmp_label])
			pre_end = tmp_time
	# print(phoneme_list)

	# if data_name=="data_00544":
	# 	print(phoneme_list)

	##Clean the f0 value
	for i in range(len(phoneme_list)):
		tmp_phoneme = phoneme_list[i][0]
		##The majority is 0
		if float(len([1 for pho in tmp_phoneme if pho==0]))/len(tmp_phoneme)>0.8:
			while tmp_phoneme[0]!=0:
				phoneme_list[i-1][0].append(tmp_phoneme.pop(0))
			while tmp_phoneme[-1]!=0:
				phoneme_list[i+1][0] = [tmp_phoneme.pop(-1)]+phoneme_list[i+1][0]
		##The majority is non zero
		elif float(len([1 for pho in tmp_phoneme if pho==0]))/len(tmp_phoneme)<0.2:
			while tmp_phoneme[0]==0:
				phoneme_list[i-1][0].append(tmp_phoneme.pop(0))
			while tmp_phoneme[-1]==0:
				phoneme_list[i+1][0] = [tmp_phoneme.pop(-1)]+phoneme_list[i+1][0]

	##The format of every pari will be [f0_value,phoneme_label]
	return phoneme_list

def Plot(y):
	x = [i for i in range(len(y))]
	plt.plot(x,y)
	plt.show()
	plt.clf()

def SavePlot(y,path,name):
	x = [i for i in range(len(y))]
	plt.plot(x,y)
	# plt.show()
	plt.savefig(path+"/"+name+".jpg")
	plt.clf()

def SaveSyllableF0List(list,path):
	file = open(path,"w+")
	for syllable,value in list:
		file.write(syllable)
		for tmp_value in value:
			file.write(","+str(tmp_value))
		file.write("\n")
	file.close()

def LoadSyllableF0List(path):
	syllable_f0_list = []
	with open(path) as f:
		for line in f:
			line = line.strip()
			tmp_arr = line.split(",")

			tmp_syllable = tmp_arr[0]
			tmp_f0_value = [float(tmp_arr[i]) for i in range(1,len(tmp_arr))]

			syllable_f0_list.append([tmp_syllable,tmp_f0_value])
	return syllable_f0_list
def ExtendSampleF0(syllable_f0_list,sample_num):
	tmp_f0_list = []
	for tmp_syllable, tmp_f0 in syllable_f0_list:
		if len(tmp_f0)>=sample_num:
			tmp_f0_list.append([tmp_syllable,tmp_f0])
		else:
			tmp_factor = sample_num/len(tmp_f0)+1
			tmp_list = []
			for tmp_value in tmp_f0:
				for i in range(tmp_factor):
					tmp_list.append(tmp_value)
			tmp_f0_list.append([tmp_syllable,tmp_list])
	# print(len([tup for tup in tmp_f0_list if len(tup[1])<sample_num]))
	return tmp_f0_list

def SubsampleF0(syllable_f0_list,sample_num):
	tmp_f0_list = []
	for tmp_syllable, tmp_f0 in syllable_f0_list:
		tmp_list = []
		# tmp_list.append(tmp_f0[0])
		tmp_segment = float(len(tmp_f0)-1)/(sample_num)
		cur_idx = 0
		for i in range(sample_num):
			cur_idx += tmp_segment
			tmp_list.append(np.mean(tmp_f0[int(round(cur_idx-tmp_segment)):int(round(cur_idx))+1]))
		tmp_f0_list.append([tmp_syllable,tmp_list])
	return tmp_f0_list
def ReadSubsampleFile(subsample_file):
	##data_f0["data_00002"] = [["ha",[100,200,...]],...]
	data_f0 = {}
	pre_data_name = ""
	with open(subsample_file) as f:
		for line in f:
			line = line.strip().split(",")
			data_name = "_".join(line[0].split("_")[0:2])
			syl = line[0].split("_")[2][0:-1]
			f0 = [float(val) for val in line[1:11]]
			if pre_data_name==data_name:
				data_f0[data_name].append([syl,f0])
			else:
				pre_data_name = data_name
				data_f0[data_name] = [[syl,f0]]
	return data_f0

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', dest='mode')
	parser.add_argument('--out_dir', dest='out_dir')
	parser.add_argument('--in_dir', dest='in_dir')
	parser.add_argument('--f0_dir', dest='f0_dir')
	parser.add_argument('--out_file', dest='out_file')
	parser.add_argument('--txt_done_data', dest='txt_done_data')
	parser.add_argument('--number', dest='number',type=int)
	parser.add_argument('--syllable_f0_file', dest='syllable_f0_file')
	parser.add_argument('--consonant_vowel_file', dest='consonant_vowel_file')
	parser.add_argument('--subsample_file', dest='subsample_file')
	parser.add_argument('--phrase_dir', dest='phrase_dir')
	parser.add_argument('--f0_in_file', dest='f0_in_file')
	parser.add_argument('--option', dest='option')


	args = parser.parse_args()
	mode = args.mode

	if mode=="how_to_run":
		print("1. python run.py --mode extract_f0 --in_dir ../../jyutping_correction/cmu_yue_wdy_correction/f0 --out_dir ./f0_value")
		print("--->Change file name into format data_00001.f0")
		print("2. python run.py --mode extract_phoneme --in_dir ./phoneme --out_dir ./f0_value")
		print("--->Change file name into format data_00001.phoneme")
		print("3. python run.py"+
			" --mode test/save_f0_syllable_list"+
			" --txt_done_data ../../txt.done.data"+
			" --f0_dir ./f0_value"+
			" --out_file ./syllable_f0_list.save")
		print("4. python run.py"+
			" --mode f0_subsampling"+
			" --syllable_f0_file ./syllable_f0_list.save"+
			" --out_file ./subsample_f0.save"
			" --number 10(subsample number)")
		print("5. python run.py"+
			" --mode put_f0_in_file"+
			" --subsample_file ./subsample_f0.save"+
			" --out_dir ./f0_in_file")
		print("##########################################################################")
		print("Additional function:")
		print("python run.py --mode plot_file_f0/plot_syllable_f0 --data_name data_00002(data_name)")
		print("python run.py --mode dct_representation --number 5(dct coefficience number)")
		print("python run.py --mode tone_experiment")
		print("python run.py --mode text_syllable_f0_experiment")
		print("python run.py --mode extract_phrase --f0_dir ./f0_value --out_dir ./phrase_dir --consonant_vowel_file ./consonant_vowel")
		print("python run.py --mode phrase_f0_dct"+
			" --subsample_file ./subsample_f0.save"+
			" --phrase_dir ./phrase_dir/phrase_syllable"+
			" --out_dir ./phrase_dir/phrase_f0_dct"+
			" --option phrase/utterance"
			" --number 5")
		print("python run.py --mode generate_residual_dir --f0_in_file ./f0_in_file --phrase_dir ./phrase_dir/phrase_f0_dct/f0 --out_dir ./res_dir")
		exit()

	if mode=="extract_f0":
		in_dir = args.in_dir
		out_dir = args.out_dir
		ExtractF0(in_dir,out_dir)
	elif mode=="extract_phoneme":
		in_dir = args.in_dir
		out_dir = args.out_dir
		ExtractPhoneme(in_dir,out_dir)

	elif mode == "plot_file_f0":
		data_name = args.data_name
		# data_name = "data_00002"
		PlotFileF0(data_name)
		exit()

	elif mode == "plot_phoneme_f0":
		data_name = args.data_name
		# data_name = "data_00002"
		os.system("mkdir ./phoneme_f0_plot/")
		os.system("rm -r ./phoneme_f0_plot/"+data_name)
		os.system("mkdir ./phoneme_f0_plot/"+data_name)

		phoneme_f0_list = AlignF0Phoneme(data_name,"./f0_value")
		for i in range(len(phoneme_f0_list)):
			phoneme = phoneme_f0_list[i]
			SavePlot(phoneme[0],"./phoneme_f0_plot/"+data_name,phoneme[1]+"_"+str(i))
		print("Plots are saved to ./phoneme_f0_plot/"+data_name)
		exit()

		# print(len(phoneme_f0_list))
		# print(dct(phoneme_f0_list[1][0],n=4))
		# Plot(phoneme_f0_list[5][0])

	elif mode == "plot_syllable_f0":
		data_name = args.data_name
		os.system("mkdir ./syllable_f0_plot/")
		os.system("rm -r ./syllable_f0_plot/"+data_name)
		os.system("mkdir ./syllable_f0_plot/"+data_name)
		data_syllable_dic = ReadTxtDoneData("../txt.done.data")
		phoneme_f0_list = AlignF0Phoneme(data_name,"./f0_value")
		syllable_f0_list = AlignF0Syllable(phoneme_f0_list,data_syllable_dic[data_name])
		for i in range(len(syllable_f0_list)):
			syllable = syllable_f0_list[i]
			SavePlot(syllable[1],"./syllable_f0_plot/"+data_name,syllable[0]+"_"+str(i))
		print("Plots are saved to ./syllable_f0_plot/"+data_name)
		exit()
	elif mode=="f0_subsampling":
		sample_num = args.number
		syllable_f0_file = args.syllable_f0_file
		out_file = args.out_file

		syllable_f0_list = LoadSyllableF0List(syllable_f0_file)
		# Plot(syllable_f0_list[1000][1])

		#Deal with the sample less than the sample number
		syllable_f0_list = ExtendSampleF0(syllable_f0_list,sample_num)
		subsample_f0_list = SubsampleF0(syllable_f0_list,sample_num)
		SaveSyllableF0List(subsample_f0_list,out_file)
		# Plot(subsample_f0_list[1000][1])

		# print(len([tup for tup in syllable_f0_list if len(tup[1])<sample_num]))
		# print([tup for tup in syllable_f0_list if len(tup[1])<sample_num])
	elif mode=="dct_representation":
		dct_cof_num = args.number
		syllable_f0_list = LoadSyllableF0List("./subsample_f0.save")
		syllable_f0_dct_list = []
		for syllable,f0_value in syllable_f0_list:
			syllable_f0_dct_list.append([syllable,dct(f0_value,n=dct_cof_num)])
		SaveSyllableF0List(syllable_f0_dct_list,"./syllable_f0_dct_representation.save")
	elif mode=="pca_dct_representation":
		dct_f0_list = LoadSyllableF0List("./syllable_f0_dct_representation.save")
		data = []
		for one_sample in dct_f0_list:
			data.append(one_sample[1])
		data = np.array(data)
		# To getter a better understanding of interaction of the dimensions
		# plot the first three PCA dimensions
		# data = PCA(n_components=2).fit_transform(data)
		data = data[:,0:3]
		############################################################################################
		##2D plot
		# plt.scatter(data[:,0],data[:,1],s=1)
		# plt.show()
		############################################################################################
		#3D plot
		fig = plt.figure(1, figsize=(8, 6))
		ax = Axes3D(fig, elev=-150, azim=110)
		ax.scatter(data[:, 0], data[:, 1], data[:, 2],
		           cmap=plt.cm.Set1, edgecolor='k', s=1)
		ax.set_title("First three PCA directions")
		ax.set_xlabel("1st eigenvector")
		ax.w_xaxis.set_ticklabels([])
		ax.set_ylabel("2nd eigenvector")
		ax.w_yaxis.set_ticklabels([])
		ax.set_zlabel("3rd eigenvector")
		ax.w_zaxis.set_ticklabels([])
		plt.show()
		############################################################################################
	elif mode=="cluster":
		dct_f0_list = LoadSyllableF0List("./syllable_f0_dct_representation.save")
		dct_cof = []
		label = []
		for one_sample in dct_f0_list:
			dct_cof.append(one_sample[1])
			label.append(one_sample[0])
		dct_cof = np.array(dct_cof)

		print("Clustering...")
		# linkage='ward', 'average', 'complete'
		clusters = KMeans(n_clusters=10,init="k-means++")
		clusters.fit(dct_cof)
		print(clusters.labels_)

	elif mode=="file_syllable_f0":
		data = []
		label = []
		with open ("subsample_f0.save") as f:
			for line in f:
				arr = line.strip().split(",")
				label.append(arr[0].split("_"))
				data.append([float(val) for val in arr[1:len(arr)]])
		data = np.array(data)
		# print(data.shape)
		# print(type(data[0]))
		begin = 0
		end = 0
		while begin<len(label):
			while end<len(label) and label[begin][1]==label[end][1]:
				# print(end)
				end += 1
			data_name = "_".join(label[begin][0:2])
			print(data_name)
			os.system("mkdir file_syllable_f0/"+data_name)
			with open("file_syllable_f0/"+data_name+"/syllable","w+") as f:
				for i in range(begin,end):
					f.write(label[i][2]+"\n")
			np.savetxt("file_syllable_f0/"+data_name+"/f0",data[begin:end],delimiter=",",fmt="%.6f")
			begin = end
	elif mode=="save_f0_syllable_list":

		############################################################################################
		#Read the syllable,phoneme,f0 data
		data_syllable_dic = ReadTxtDoneData(args.txt_done_data)
		# print(data_syllable_dic)

		# data_dir = "f0_value"
		data_dir = args.f0_dir
		file_list = os.listdir(data_dir)
		syllable_f0_list = []

		for file in file_list:
			if "f0" not in file:
				continue
			data_name = file.split(".")[0]
			# print(data_name)

			# print(data_syllable_dic[data_name])

			tmp_phoneme_f0_list = AlignF0Phoneme(data_name,data_dir)

			tmp_syllable_f0_list = AlignF0Syllable(tmp_phoneme_f0_list,data_syllable_dic[data_name])

			########################################################################################
			for one_syl in tmp_syllable_f0_list:
				one_syl[0] = data_name+"_"+one_syl[0]
			########################################################################################


			syllable_f0_list += tmp_syllable_f0_list
			# print(syllable_f0_list)
			# break
		# print(syllable_f0_list[0])
		############################################################################################

		############################################################################################
		##Save f0 plot
		# for i in range(len(syllable_f0_list)):
		# 	syllable = syllable_f0_list[i]
		# 	SavePlot(syllable[1],"./phoneme_f0_plot",syllable[0]+"_"+str(i))
		############################################################################################

		############################################################################################
		# out_file = "./syllable_f0_list.save"
		out_file = args.out_file
		SaveSyllableF0List(syllable_f0_list,out_file)
		# syllable_f0_list = LoadSyllableF0List("./syllable_f0_list.save")
		# # print(len(syllable_f0_list[0][1]))
		# plt.plot(syllable_f0_list[9][1],color="r",linestyle="dotted")
		# plt.plot(idct(dct(syllable_f0_list[0][1]))/100,color="b",linestyle="dotted")
		# plt.show()
		############################################################################################

		############################################################################################
		#Generate dct representation for f0 value of syllable
		# dct_length = 5
		# syllable_f0_dct_list = []
		# for syllable,f0_value in syllable_f0_list:
		# 	# print("Processing "+syllable+"...")
		# 	syllable_f0_dct_list.append([syllable,dct(f0_value,n=dct_length)])
		# SaveSyllableF0List(syllable_f0_dct_list,"./syllable_f0_dct_representation.save")
		############################################################################################
	elif mode=="tone_experiment":
		file_list = os.listdir("./file_syllable_f0")
		syl_dic = {}
		##{"maa1":[[f0],[f0],[f0],...,[f0]]}
		for file in file_list:
			if file==".DS_Store":
				print(".DS_Store")
				continue
			f0 = np.loadtxt("./file_syllable_f0/"+file+"/f0",delimiter=",")
			syl = [syl.strip() for syl in open("./file_syllable_f0/"+file+"/syllable").readlines()]
			for i in range(len(syl)):
				if syl[i] not in syl_dic:
					syl_dic[syl[i]] = []
				syl_dic[syl[i]].append(f0[i])
		for key in syl_dic.keys():
			syl_dic[key] = np.array(syl_dic[key])
		syl_count = {}
		for syl,l in syl_dic.items():
			if len(l) not in syl_count:
				syl_count[len(l)] = 0
			syl_count[len(l)] += 1


		filter_freq = 0

		##print frequency statistics
		##syl_count[frequency] = syllable_number
		# count_list = sorted([[freq,num] for freq,num in syl_count.items()],key=lambda tup:tup[0])
		# print("[syllable_frequency, syllable_number]")
		# print(count_list)

		##plot the mean in the same syllable
		# sample = syl_dic["lei1"]
		# plt.scatter(range(len(sample)),sample.mean(axis=1))
		# plt.show()

		##plot sample f0 contour
		#save to ./syllable_f0_contour
		# for key in syl_dic.keys():
		# 	sample = syl_dic[key]
		# 	if len(sample)<filter_freq:
		# 		continue
		# 	for i in range(len(sample)):
		# 		plt.plot(sample[i])
		# 	plt.savefig("./syllable_f0_contour/"+key+".png")
		# 	plt.clf()


		##Normalize the sample f0 list
		# norm_dic = {}
		# for syl,l in syl_dic.items():
		# 	if len(l)<filter_freq:
		# 		continue
		# 	norm_dic[syl] = (l-l.mean(axis=1).reshape((-1,1)))/(l.std(axis=1).reshape((-1,1))+0.00001)
		# print(norm_dic.keys())


		#Plot some nomralized sample
		# sample = norm_dic["lei1"]
		# for i in range(min(10,len(sample))):
		# 	plt.plot(sample[i][2:8])
		# plt.show()

		##plot sample f0 contour
		#save to ./syllable_f0_contour
		# for key in norm_dic.keys():
		# 	sample = norm_dic[key]
		# 	if len(sample)<filter_freq:
		# 		continue
		# 	for i in range(len(sample)):
		# 		plt.plot(sample[i])
		# 	plt.savefig("./syllable_f0_contour/"+key+".png")
		# 	plt.clf()

		# 	plt.plot(sample.mean(axis=0))
		# 	plt.savefig("./syllable_f0_contour/"+key+"_mean.png")
		# 	plt.clf()


		# rmse_dic = {}
		# for syl,l in syl_dic.items():
		# 	if len(l)<filter_freq:
		# 		continue
		# 	rmse_dic[syl+"_"+str(len(l))] = l.mean(axis=1).std()
		# print(rmse_dic)

		##print tone specific rmse
		norm_dic = syl_dic
		for i in range(1,7):
			l = [norm_dic[key] for key in norm_dic.keys() if key[-1]==str(i)]
			l_mean = [norm_dic[key].mean(axis=0) for key in norm_dic.keys() if key[-1]==str(i)]
			print("tone "+str(i))
			arr = np.vstack(tuple(l))
			arr_mean = np.vstack(tuple(l_mean))
			print(RMSE(arr))
			plt.plot(arr_mean.mean(axis=0)[1:9],label="tone "+str(i))
		plt.legend()
		plt.savefig("tone.png")

		l = [norm_dic[key] for key in norm_dic.keys()]
		print("all tone")
		print(RMSE(np.vstack(tuple(l))))
	elif mode=="text_syllable_f0_experiment":

		##f0 = [[data_name,syllable,word,f0],...]
		f0 = word_syllable_f0("../../audio_jyutping.txt","./subsample_f0.save")

		word_dic = {}
		for tup in f0:
			word = tup[2]
			if word not in word_dic:
				word_dic[word] = []
			word_dic[word].append(tup)
		freq_filter = 5
		for word in word_dic.keys():
			if len(word_dic[word])<freq_filter:
				continue
			print(word+"("+str(len(word_dic[word]))+")"),
		print("")
		def norm(arr):
			return (arr-arr.mean())/(arr.std()+0.00001)
		# sample_word = u'要'
		sample_word = u'要'
		sample = word_dic[sample_word]
		random.shuffle(sample)
		for i in range(len(sample)):
		# for i in range(min(len(sample),10)):
			arr = np.array(sample[i][3])
			data_name = sample[i][0]
			# plt.plot(norm(np.array(arr)),label=data_name)
			plt.plot(np.array(arr),label=data_name)
			# plt.plot(np.ones(arr.shape)*arr.mean(),label=data_name)
		# plt.locator_params(axis='y', nbins=10)
		# plt.yticks([])
		plt.legend(loc=(1.04,0))
		# plt.show()
		plt.savefig("./test_pic",bbox_inches='tight')

	elif mode=="put_f0_in_file":
		subsample_file = args.subsample_file
		out_dir = args.out_dir
		os.system("mkdir "+out_dir)
		data_dic = ReadSubsampleFile(subsample_file)
		for data_name,f0_list in data_dic.items():
			with open(out_dir+"/"+data_name,"w+") as f:
				for tup in f0_list:
					f.write(" ".join([str(val) for val in tup[1]])+"\n")


	elif mode=="extract_phrase":
		f0_dir = args.f0_dir
		out_dir = args.out_dir
		consonant_vowel_file = args.consonant_vowel_file
		cons_dic = {}
		vowel_dic = {}
		with open(consonant_vowel_file) as f:
			cons_l = f.readline().strip().split(" ")
			for cons in cons_l:
				cons_dic[cons] = True
			vowel_l = f.readline().strip().split(" ")
			for vowel in vowel_l:
				vowel_dic[vowel] = True

		os.system("mkdir "+out_dir)
		file_list = os.listdir(f0_dir)
		file_list = [file_name for file_name in file_list if "phoneme" in file_name]
		for file in file_list:
			with open(f0_dir+"/"+file) as inf, open(out_dir+"/"+file.split(".")[0],"w+") as outf:
				lines = inf.readlines()
				ph_l = [line.strip().split(" ")[1] for line in lines]
				phrase_l = [[]]
				for ph in ph_l:
					if ph=="pau" or ph=="ssil":
						phrase_l.append([])
					else:
						phrase_l[-1].append(ph)
				phrase_l = [phrase for phrase in phrase_l if len(phrase)!=0]
				for phrase in phrase_l:
					cat_phrase = []
					idx = 0
					while idx < len(phrase):
						if phrase[idx] in vowel_dic:
							cat_phrase.append(phrase[idx])
							idx += 1
						else:
							cat_phrase.append(phrase[idx]+phrase[idx+1])
							idx += 2
					outf.write(" ".join(cat_phrase)+"\n")
	elif mode=="phrase_f0_dct":
		phrase_dir = args.phrase_dir
		subsample_file = args.subsample_file
		out_dir = args.out_dir
		num = args.number
		option = args.option

		##read subsample file in a dictionary
		##data_f0["data_00002"] = [["ha",[100,200,...]],...]
		data_f0 = ReadSubsampleFile(subsample_file)

		##map the syllable in phrase with f0 mean value
		os.system("mkdir "+out_dir)
		os.system("mkdir "+out_dir+"/f0")
		os.system("mkdir "+out_dir+"/dct")
		file_list = os.listdir(phrase_dir)
		file_list = [file for file in file_list if "data" in file]
		for phr_file in file_list:
			with open(phrase_dir+"/"+phr_file) as phr_f, open(out_dir+"/f0/"+phr_file,"w+") as out_f,open(out_dir+"/dct/"+phr_file,"w+") as dct_f:
				idx = 0
				file_f0 = data_f0[phr_file]
				if option=="phrase":
					for line in phr_f:
						line = line.strip().split(" ")
						tmp_f0_l = []
						for i in range(len(line)):
							# print(line[i]+" "+file_f0[idx+i][0])
							assert line[i]==file_f0[idx+i][0]
							tmp_f0_l += file_f0[idx+i][1]
						tmp_f0_l = np.array(tmp_f0_l)
						dct_vec = dct(tmp_f0_l)
						dct_vec[num:] = 0
						idct_vec = (idct(dct_vec)/(len(tmp_f0_l)*2)).reshape((-1,10))
						for i in range(len(idct_vec)):
							out_f.write(" ".join(idct_vec[i].astype(np.str).tolist())+"\n")
						dct_f.write(" ".join(dct_vec[0:num].astype(np.str).tolist())+"\n")
						idx += len(line)
				elif option=="utterance":
					utt_f0 = []
					for tup in file_f0:
						utt_f0 += tup[1]
					# print(utt_f0)
					utt_f0 = np.array(utt_f0).flatten()
					dct_vec = dct(utt_f0)
					dct_vec[num:] = 0
					idct_vec = (idct(dct_vec)/(len(dct_vec)*2)).reshape((-1,10))
					for i in range(len(idct_vec)):
						out_f.write(" ".join(idct_vec[i].astype(np.str).tolist())+"\n")
					dct_f.write(" ".join(dct_vec[0:num].astype(np.str).tolist())+"\n")

	elif mode=="generate_residual_dir":
		f0_dir = args.f0_in_file
		phrase_dir = args.phrase_dir
		out_dir = args.out_dir
		os.system("mkdir "+out_dir)
		file_list = [file for file in os.listdir(f0_dir) if "data" in file]
		for file in file_list:
			ip_f0 = np.loadtxt(phrase_dir+"/"+file,delimiter=" ")
			f0 = np.loadtxt(f0_dir+"/"+file,delimiter=" ")
			np.savetxt(out_dir+"/"+file,f0-ip_f0,delimiter=" ",fmt="%.3f")


	elif mode=="test":
		data_f0 = ReadSubsampleFile("./gen_f0/subsample_f0.save")
		data_name = "data_01000"
		phrase_level_f0 = np.loadtxt("./phrase_dir/phrase_f0_dct/"+data_name,delimiter=" ").flatten()
		true_f0 = data_f0[data_name]
		true_f0 = np.array([tup[1] for tup in true_f0]).flatten()
		assert len(true_f0)==len(phrase_level_f0)
		# plt.scatter(np.arange(len(true_f0)),true_f0,label="true_f0",s=1)
		# plt.scatter(np.arange(len(phrase_level_f0)),phrase_level_f0,label="phrase_level_f0",s=1)
		plt.plot(true_f0,label="true_f0")
		plt.plot(phrase_level_f0,label="phrase_level_f0")
		plt.legend()
		plt.show()



















