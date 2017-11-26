import sys
import os
import matplotlib.pyplot as plt

def ExtractF0():
	os.system("mkdir f0_value")
	os.system("export ESTDIR=/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools")
	f0_path = "../cmu_yue_wdy_normal_build/f0"
	estdir = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools"

	file_list = os.listdir(f0_path)
	for file in file_list:
		if "f0" not in file:
			continue
		# print(estdir+"/bin/ch_track f0/"+f0_path+"/"+file+" > f0_value/"+file.split(".")[1]+".out")
		os.system(estdir+"/bin/ch_track "+f0_path+"/"+file+" > f0_value/"+file.split(".")[0]+".f0")

def ExtractPhoneme():
	file_list = os.listdir("phoneme")
	for file in file_list:
		if "data" not in file:
			continue
		# print(estdir+"/bin/ch_track f0/"+f0_path+"/"+file+" > f0_value/"+file.split(".")[1]+".out")
		tmp_file = open("f0_value/"+file.split(".")[0]+".phoneme","w+")
		with open("phoneme/"+file) as f:
			for line in f:
				if "#" in line:
					continue
				line = line.strip()
				arr = line.split(" ")
				tmp_file.write(arr[0]+" "+arr[2]+"\n")
		tmp_file.close()


def PlotF0(path,f0_filename,pho_filename):
	timeline = []
	f0_list = []
	time = 0

	tmp_f0_list = []
	tmp_timeline = []
	with open(path+f0_filename) as f:
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
		plt.plot(timeline[i],f0_list[i],'r-',lw=0.1)
		# plt.axvline(timeline[i][-1],linestyle="dotted")

	pho_timeline = []
	with open(path+pho_filename) as f:
		for line in f:
			line = line.strip()
			# pho_timeline.append(float(line.split(" ")[0])*1000)
			plt.axvline(float(line.split(" ")[0])*1000,linestyle="dotted",lw=0.1)

	# pho_y = [100 for i in range(len(pho_timeline))]

	# plt.plot(pho_timeline,pho_y,".")



	plt.ylabel("f0 value")
	plt.xlabel("time(ms)")

	# plt.show()
	os.system("mkdir f0_plot")
	plt.savefig("f0_plot/"+f0_filename+".jpg", dpi=1200)
	# plt.savefig("f0_plot/"+f0_filename+".jpg")
	print("Plot saved!")
	plt.clf()



if __name__=="__main__":
	mode = sys.argv[1]
	if mode=="extract_f0":
		ExtractF0()
	elif mode=="plot_f0":
		# PlotF0("f0_value/","data_00002.out")
		# PlotF0("f0_value/","data_00343.out","phoneme/","data_00343.lab")
		# PlotF0("f0_value/","data_01231.f0","data_01231.phoneme")
		PlotF0("f0_value/","data_00343.f0","data_00343.phoneme")
	elif mode=="extract_phoneme":
		ExtractPhoneme()