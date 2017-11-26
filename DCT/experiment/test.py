import numpy as np
import os
import matplotlib.pyplot as plt

def normalize_row(arr):
	mean = np.mean(arr,axis=1).reshape((-1,1))
	std = np.std(arr,axis=1).reshape((-1,1))+0.001
	norm_arr = (arr-mean)/std
	return norm_arr,mean,std

if __name__=="__main__":
	#####################################################################
	# data = np.genfromtxt("./subsample_f0.save",delimiter=",",dtype=np.str)
	# name = data[:,0]
	# val = data[:,1:data.shape[1]].astype(np.float)
	# tone_dic = {}
	# for i in range(data.shape[0]):
	# 	tone = name[i][-1]
	# 	if tone not in tone_dic:
	# 		tone_dic[tone] = []
	# 	tone_dic[tone].append(val[i])
	# for tone,f0_list in tone_dic.items():
	# 	print("Processing tone "+tone)
	# 	f0_val = np.array(f0_list)
	# 	f0_list,_,_ = normalize_row(f0_val)
	# 	np.savetxt("./tone/"+tone,f0_list,delimiter=",")
	# 	os.system("mkdir ./tone/tone_"+tone)
	# 	np.random.shuffle(f0_val)
	# 	for i in range(50):
	# 		plt.plot(f0_val[i][2:8])
	# 		plt.savefig("./tone/tone_"+tone+"/"+str(i)+".png")
	# 		plt.clf()
	#####################################################################

	#####################################################################
	data = np.genfromtxt("./subsample_f0.save",delimiter=",",dtype=np.str)
	name = data[:,0]
	val = data[:,1:data.shape[1]].astype(np.float)
	syl_dic = {}
	for i in range(data.shape[0]):
		syl = name[i].split("_")[-1][0:-1]
		if syl not in syl_dic:
			syl_dic[syl] = [[],[]]
		syl_dic[syl][0].append(val[i])
		syl_dic[syl][1].append(name[i])
		# print(name[i])
	for syl,f0_list in syl_dic.items():
		print("Processing syllable "+syl)
		f0_val = np.array(f0_list[0])
		f0_val,_,_ = normalize_row(f0_val)
		np.savetxt("./syllable/"+syl,f0_val,delimiter=",")
		os.system("mkdir ./syllable/syllable_"+syl)
		# np.random.shuffle(f0_val)
		for i in range(len(f0_val)):
			plt.plot(f0_val[i])
			plt.savefig("./syllable/syllable_"+syl+"/"+f0_list[1][i].split("_")[-1]+"_"+str(i)+".png")
			plt.clf()
	#####################################################################

	