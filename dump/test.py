import matplotlib.pyplot as plt
import numpy as np
import os

if __name__=="__main__":
	# data = np.loadtxt("./train_data_f0_vector_norm_row",delimiter=" ")
	# np.random.shuffle(data)
	# for i in range(50):
	# 	plt.plot(data[i])
	# 	plt.savefig("./plot/"+str(i)+".png")
	# 	plt.clf()

	map_file = "../decision_tree/train_dev_data_vector/dev_data/syllable_map"
	indir = "../dumpfeats/concat_feature"
	outfile = "./outfile"
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
			assert new_feat.split(" ")[0]==syl
			of.write(new_feat+"\n")



