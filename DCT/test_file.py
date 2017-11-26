import matplotlib.pyplot as plt
import numpy as np
import os

if __name__=="__main__":
	# data_name_list = [data_name for data_name in os.listdir("./predict_vs_true_contour/f0_val/") if "data" in data_name]
	# os.system("rm -r ./predict_vs_true_contour/plot/*")
	# for data_name in data_name_list:
	# 	print(data_name)
	# 	predict = np.loadtxt("./predict_vs_true_contour/f0_val/"+data_name,delimiter=",")
	# 	true = [] 
	# 	with open("./f0_value/"+data_name+".f0") as f:
	# 		for line in f:
	# 			arr = line.strip().split(" ")
	# 			if arr[0]=="":
	# 				true.append(float(arr[1]))
	# 			else:
	# 				true.append(float(arr[0]))
	# 	true = np.array(true)
	# 	predict = predict[0:(np.nonzero(predict))[0][-1]+1]
	# 	true = true[0:(np.nonzero(true))[0][-2]+1]
	# 	# print(np.nonzero(predict))
	# 	plt.scatter(range(len(predict)),predict,label="predict",s=1)
	# 	plt.scatter(range(len(true)),true,label="true",s=1)
	# 	plt.legend()
	# 	plt.savefig("./predict_vs_true_contour/plot/"+data_name+".png")
	# 	plt.clf()
	file_list = os.listdir("./predict_vs_true_contour/f0_val")
	print(file_list)
	for file_name in file_list:
		 # arr = np.loadtxt("./predict_vs_true_contour/f0_val/"+file_name)
		 # # arr = arr[0:(np.nonzero(arr))[0][-1]+1]
		 # np.savetxt("./predict_vs_true_contour/f0_val/"+file_name,arr,fmt="%.5f")
		 os.system("cp ./f0_value/"+file_name+".f0 ~/Desktop/predict_vs_true_contour")