import matplotlib
matplotlib.use('TkAgg')
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":

	##plot mean for every file
	file_list = [file for file in os.listdir("./") if "run" not in file]
	print(file_list)
	for file in file_list:
		arr = np.loadtxt(file,delimiter=" ")
		arr = (arr-arr.mean())/(arr.std()+0.0001)
		plt.plot(arr.mean(axis=0),label=file)
	plt.legend()
	plt.show()

	##plot random sample in one file
	# file = "6"
	# sample_number = 20
	# arr = np.loadtxt(file,delimiter=" ")
	# np.random.shuffle(arr)
	# for i in range(sample_number):
	# 	plt.plot(arr[i])
	# plt.show()