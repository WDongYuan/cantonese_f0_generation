import matplotlib.pyplot as plt
import sys
import numpy as np
import random
def Jyutping2F0(file,jyu):
	f0_list = []
	with open(file) as f:
		for line in f:
			line_f0 = line.strip().split(",")
			if line_f0[0].split("_")[-1] == jyu:
				f0_list.append([float(line_f0[i]) for i in range(1,len(line_f0))])
	return f0_list






if __name__=="__main__":
	tool = sys.argv[1]
	if tool == "subtract_f0":
		file = sys.argv[2]
		jyu = sys.argv[3]
		plot_num = int(sys.argv[4])

		f0_list = Jyutping2F0(file,jyu)
		f0_list = np.array(f0_list)
		f0_mean = np.mean(f0_list,0)

		# plt.plot(np.std(f0_list-f0_mean,1).flatten())
		# plt.show()

		id_l = range(len(f0_list))
		random.shuffle(id_l)
		for i in range(plot_num):
			# plt.scatter(range(len(f0_mean)),f0_list[id_l[i]]-f0_mean)
			plt.plot(range(len(f0_mean)),f0_list[id_l[i]]-f0_mean)
			plt.savefig("./dump/"+str(id_l[i])+".png")
			plt.clf()