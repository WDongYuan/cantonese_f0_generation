from scipy.fftpack import dct,idct
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

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

def AlignF0Phoneme(data_name):
	timeline = []
	f0_list = []
	time = 0

	f0_file = "./f0_value/"+data_name+".f0"
	with open(f0_file) as f:
		for line in f:
			line = line.strip()

			f0_value = float(line.split(" ")[0])
			if f0_value == -1:
				break

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


if __name__=="__main__":
	if len(sys.argv)==1:
		print("python run.py plot_file_f0/plot_syllable_f0 data_00002(data_name)")
		print("python run.py test")
		print("python run.py f0_subsampling 10(subsample number)")
		print("python run.py dct_representation 5(dct coefficience number)")
		exit()

	mode = sys.argv[1]
	if mode == "plot_file_f0":
		data_name = sys.argv[2]
		# data_name = "data_00002"
		PlotFileF0(data_name)
		exit()

	elif mode == "plot_phoneme_f0":
		data_name = sys.argv[2]
		# data_name = "data_00002"
		os.system("mkdir ./phoneme_f0_plot/")
		os.system("rm -r ./phoneme_f0_plot/"+data_name)
		os.system("mkdir ./phoneme_f0_plot/"+data_name)

		phoneme_f0_list = AlignF0Phoneme(data_name)
		for i in range(len(phoneme_f0_list)):
			phoneme = phoneme_f0_list[i]
			SavePlot(phoneme[0],"./phoneme_f0_plot/"+data_name,phoneme[1]+"_"+str(i))
		print("Plots are saved to ./phoneme_f0_plot/"+data_name)
		exit()

		# print(len(phoneme_f0_list))
		# print(dct(phoneme_f0_list[1][0],n=4))
		# Plot(phoneme_f0_list[5][0])

	elif mode == "plot_syllable_f0":
		data_name = sys.argv[2]
		os.system("mkdir ./syllable_f0_plot/")
		os.system("rm -r ./syllable_f0_plot/"+data_name)
		os.system("mkdir ./syllable_f0_plot/"+data_name)
		data_syllable_dic = ReadTxtDoneData("../txt.done.data")
		phoneme_f0_list = AlignF0Phoneme(data_name)
		syllable_f0_list = AlignF0Syllable(phoneme_f0_list,data_syllable_dic[data_name])
		for i in range(len(syllable_f0_list)):
			syllable = syllable_f0_list[i]
			SavePlot(syllable[1],"./syllable_f0_plot/"+data_name,syllable[0]+"_"+str(i))
		print("Plots are saved to ./syllable_f0_plot/"+data_name)
		exit()
	elif mode=="f0_subsampling":
		sample_num = int(sys.argv[2])
		syllable_f0_list = LoadSyllableF0List("./syllable_f0_list.save")
		# Plot(syllable_f0_list[1000][1])

		#Deal with the sample less than the sample number
		syllable_f0_list = ExtendSampleF0(syllable_f0_list,sample_num)
		subsample_f0_list = SubsampleF0(syllable_f0_list,sample_num)
		SaveSyllableF0List(subsample_f0_list,"./subsample_f0.save")
		# Plot(subsample_f0_list[1000][1])

		# print(len([tup for tup in syllable_f0_list if len(tup[1])<sample_num]))
		# print([tup for tup in syllable_f0_list if len(tup[1])<sample_num])
	elif mode=="dct_representation":
		dct_cof_num = int(sys.argv[2])
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
	elif mode=="test":

		############################################################################################
		#Read the syllable,phoneme,f0 data
		data_syllable_dic = ReadTxtDoneData("../../txt.done.data")

		data_dir = "f0_value"
		file_list = os.listdir(data_dir)
		syllable_f0_list = []

		for file in file_list:
			if "f0" not in file:
				continue
			data_name = file.split(".")[0]

			# print(data_syllable_dic[data_name])

			tmp_phoneme_f0_list = AlignF0Phoneme(data_name)

			tmp_syllable_f0_list = AlignF0Syllable(tmp_phoneme_f0_list,data_syllable_dic[data_name])

			########################################################################################
			for one_syl in tmp_syllable_f0_list:
				one_syl[0] = data_name+"_"+one_syl[0]
			########################################################################################


			syllable_f0_list += tmp_syllable_f0_list
			# print(syllable_f0_list)
			# break
		print(syllable_f0_list[0])
		############################################################################################

		############################################################################################
		##Save f0 plot
		# for i in range(len(syllable_f0_list)):
		# 	syllable = syllable_f0_list[i]
		# 	SavePlot(syllable[1],"./phoneme_f0_plot",syllable[0]+"_"+str(i))
		############################################################################################

		############################################################################################
		# SaveSyllableF0List(syllable_f0_list,"./syllable_f0_list.save")
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






