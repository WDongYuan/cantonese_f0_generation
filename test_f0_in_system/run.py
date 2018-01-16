import matplotlib
matplotlib.use('TkAgg')
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import random
def FileSyllableSegment(phoneme_file_path,syllable_list):
	cur_time = 0.0
	file = open(phoneme_file_path)
	# file.readline()
	sample = file.readline().strip()
	ph_timeline = []
	while sample!="":
		time,ph = sample.split(" ")
		time = float(time)*1000
		ph_timeline.append([ph,[cur_time,time]])
		cur_time = time
		sample = file.readline().strip()
	# print(ph_timeline)

	syl_timeline = []
	ph_i = 0
	ph_str = ""
	begin_time = 0
	for syl_i in range(len(syllable_list)):
		while ph_timeline[ph_i][0] not in syllable_list[syl_i]:
			ph_i += 1
			begin_time = ph_timeline[ph_i][1][0]

		syl = syllable_list[syl_i]
		while (ph_str+ph_timeline[ph_i][0]) in syl:
			ph_str += ph_timeline[ph_i][0]
			ph_i += 1
		end_time = ph_timeline[ph_i-1][1][1]
		syl_timeline.append([syl,[begin_time,end_time]])
		begin_time = ph_timeline[ph_i][1][0]
		ph_str = ""

	return syl_timeline


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', dest='mode')
	parser.add_argument('--awb_synth', dest='awb_synth')
	parser.add_argument('--f0_dir', dest='f0_dir')
	parser.add_argument('--ccoef_dir', dest='ccoef_dir')
	parser.add_argument('--out_dir', dest='out_dir')
	parser.add_argument('--dir1', dest='dir1')
	parser.add_argument('--dir2', dest='dir2')

	parser.add_argument('--predict_dir', dest='predict_dir')
	parser.add_argument('--true_f0_dir', dest='true_f0_dir')
	parser.add_argument('--wav_dir1', dest='wav_dir1',default="")
	parser.add_argument('--wav_dir2', dest='wav_dir2',default="")
	parser.add_argument('--data_name',dest='data_name')
	parser.add_argument('--include_sil',dest='include_sil')
	parser.add_argument('--dir_list',dest='dir_list')
	parser.add_argument('--all_f0',dest='all_f0')
	parser.add_argument('--in_file',dest='in_file')
	parser.add_argument('--number',dest='number',type=int)
	args = parser.parse_args()
	mode = args.mode

	if mode=="how_to_run":
		print("python run.py --mode synthesis_with_f0"+
			" --awb_synth ./my_synth_f0"+
			" --ccoef_dir ../../cmu_yue_wdy_addf0/ccoefs"+
			" --f0_dir ./voice_lib/0/f0_val"+
			" --out_dir ./voice_lib/0/wav")
		print("python run.py --mode compare_wav"+
			" --dir1 ./voice_lib/0/wav"+
			" --dir2 ./voice_lib/1/wav")
		print("python run.py"+
			" --mode plot_two_file_with_syllable"+
			" --true_f0_dir ../DCT/f0_value"+
			" --predict_dir ../decision_tree/wagon/predict_f0_in_file"+
			" --dir1 ./voice_lib/0/f0_val"+
			" --dir2 ./voice_lib/1/f0_val"+
			" --wav_dir1 ./voice_lib/0/wav"+
			" --wav_dir2 ./voice_lib/1/wav")
		print("python run.py"+
			" --mode timeline_rmse"+
			" --include_sil true"
			" --dir1 ./voice_lib/0/f0_val"+
			" --dir2 ./voice_lib/1/f0_val")

		print("python run.py"+
			" --mode plot_dir_list_with_syllable"+
			" --true_f0_dir ../DCT/f0_value"+
			# " --predict_dir ../decision_tree/wagon/predict_f0_in_file"+
			" --all_f0 ../DCT/subsample_f0.save"+
			" --dir_list ./dir_list"+
			" --data_name data_00013")
		print("python run.py"+
			" --mode plot_shuffle_data"+
			" --in_file ./classify_feature/1"+
			" --number 20")

	elif mode=="synthesis_with_f0":
		f0_dir = args.f0_dir
		ccoef_dir = args.ccoef_dir
		awb_synth = args.awb_synth
		out_dir = args.out_dir
		f0_file_l = os.listdir(f0_dir)
		os.system("rm -r "+out_dir)
		os.system("mkdir "+out_dir)
		for file in f0_file_l:
			if "data" not in file:
				continue
			# print(file)
			os.system(awb_synth+" "+f0_dir+"/"+file+" "+ccoef_dir+" "+out_dir)
	
	elif mode=="compare_wav":
		os.environ["ESTDIR"]="/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools"
		os.environ["FESTVOXDIR"]="/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/festvox"
		os.environ["SPTKDIR"]="/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/SPTK"
		# os.system("export FESTVOXDIR=/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/festvox")
		# os.system("export ESTDIR=/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools")
		# os.system("export SPTKDIR=/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/SPTK")
		FESTVOXDIR = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/festvox"
		dir1 = args.dir1
		tmp1 = os.path.abspath(dir1)
		os.system("ln -s "+tmp1+" dir1")

		dir2 = args.dir2
		tmp2 = os.path.abspath(dir2)
		os.system("ln -s "+tmp2+" dir2")
		os.system(FESTVOXDIR+"/src/eval/abtest dir1 dir2")
		os.system("rm -r dir1")
		os.system("rm -r dir2")
	elif mode == "plot_two_file_with_syllable":
		dir1 = args.dir1
		dir2 = args.dir2
		wav_dir1 = args.wav_dir1
		wav_dir2 = args.wav_dir2

		# data_name = args.data_name
		true_f0_dir = args.true_f0_dir
		predict_file_f0_dir = args.predict_dir

		file_list = os.listdir(dir1)
		random.shuffle(file_list)
		print("data name example:")
		for i in range(20):
			print(file_list[i]),
		print("")
		data_name = ""
		while True:
			data_name = raw_input("please input data_name(\"end\" to end): ")
			if data_name=="end":
				break

			phoneme_file = true_f0_dir+"/"+data_name+".phoneme"
			true_f0_file = true_f0_dir+"/"+data_name+".f0"
			syllable_file = predict_file_f0_dir+"/"+data_name
			
			syl_f0 = []
			with open(syllable_file) as f:
				for line in f:
					arr = line.strip().split(" ")
					syl_f0.append([arr[0],np.array(arr[1:len(arr)]).astype(np.float)])
			syl_timeline = FileSyllableSegment(phoneme_file,[sample[0] for sample in syl_f0])
			# print(syl_timeline)

			fig, ax = plt.subplots()

			file1 = dir1+"/"+data_name+".f0"
			file2 = dir2+"/"+data_name+".f0"
			arr1 = np.loadtxt(file1,delimiter=",")
			arr2 = np.loadtxt(file2,delimiter=",")
			ax.scatter(np.arange(len(arr1))*5,arr1,label="file1",s=0.2)
			ax.scatter(np.arange(len(arr2))*5,arr2,label="file2",s=0.2)

			y = np.ones((len(syl_timeline),))*50
			for i in range(len(y)):
				y[i] += 10*(-1)**i

			ax.scatter([tup[1][1] for tup in syl_timeline],y)
			for i, tup in enumerate(syl_timeline):
				ax.annotate(tup[0],(tup[1][1],y[i]),FontSize=5, rotation=45)
				ax.axvline(x=tup[1][1],linestyle='-',linewidth=0.5, color='b')
			plt.legend()
			plt.savefig("./"+data_name+".jpg",dpi=200)
			os.system("open "+data_name+".jpg")

			if wav_dir1 != "":
				my_input = "y"
				while my_input=="y":
					os.system("nohup play "+wav_dir1+"/"+data_name+".wav")
					os.system("nohup play "+wav_dir2+"/"+data_name+".wav")
					my_input = raw_input("replay? (y/n)")

	elif mode=="timeline_rmse":
		dir1 = args.dir1
		dir2 = args.dir2
		include_sil = args.include_sil
		file_list = [file_name for file_name in os.listdir(dir1) if "data" in file_name]
		data1 = []
		data2 = []
		rmse = 0
		cor = 0
		for file in file_list:
			data1 = np.loadtxt(dir1+"/"+file,delimiter=" ")
			data2 = np.loadtxt(dir2+"/"+file,delimiter=" ")
			if include_sil!="true":
				data1 = data1[np.where(data1>0)[0]]
				data2 = data2[np.where(data2>0)[0]]
			rmse += np.sqrt(np.square(data1-data2).mean())
			cor += np.corrcoef(data1,data2)[0][1]
		# print(data1.shape)
		# print(data2.shape)
		rmse /= len(file_list)
		cor /= len(file_list)
		print("rmse: "+str(rmse))
		print("correlation: "+str(cor))

	elif mode == "plot_dir_list_with_syllable":
		dir_list = []
		with open(args.dir_list) as f:
			for line in f:
				dir_list.append(line.strip())
				
		data_name = args.data_name
		true_f0_dir = args.true_f0_dir
		# predict_file_f0_dir = args.predict_dir
		all_f0 = args.all_f0

		phoneme_file = true_f0_dir+"/"+data_name+".phoneme"
		true_f0_file = true_f0_dir+"/"+data_name+".f0"
		# syllable_file = predict_file_f0_dir+"/"+data_name
		
		syl_f0 = []
		# with open(syllable_file) as f:
		# 	for line in f:
		# 		arr = line.strip().split(" ")
		# 		syl_f0.append([arr[0],np.array(arr[1:len(arr)]).astype(np.float)])
		with open(all_f0) as f:
			cont = f.readlines()
			begin = 0
			while cont[begin].split(",")[0][0:len(data_name)]!=data_name:
				begin += 1
			while begin<len(cont) and cont[begin].split(",")[0][0:len(data_name)]==data_name:
				arr = cont[begin].strip().split(",")
				syl_f0.append([arr[0].split("_")[-1],np.array(arr[1:]).astype(np.float)])
				begin += 1

		syl_timeline = FileSyllableSegment(phoneme_file,[sample[0] for sample in syl_f0])
		# print(syl_timeline)

		fig, ax = plt.subplots()

		for i in range(len(dir_list)):
			d = dir_list[i]
			file = d+"/"+data_name+".f0"
			arr = np.loadtxt(file,delimiter=",")
			ax.scatter(np.arange(len(arr))*5,arr,label=d,s=0.2)

		y = np.ones((len(syl_timeline),))*50
		for i in range(len(y)):
			y[i] += 10*(-1)**i

		ax.scatter([tup[1][1] for tup in syl_timeline],y)
		for i, tup in enumerate(syl_timeline):
			ax.annotate(tup[0],(tup[1][1],y[i]),FontSize=5, rotation=45)
			ax.axvline(x=tup[1][1],linestyle='-',linewidth=0.5, color='b')
		plt.legend()
		plt.savefig("./"+data_name+".jpg",dpi=200)
		os.system("open "+data_name+".jpg")
	elif mode=="plot_shuffle_data":
		in_file = args.in_file
		number = args.number
		with open(in_file) as f:
			cont = f.readlines()
			random.shuffle(cont)
			for i in range(number):
				arr = [float(val) for val in cont[i].strip().split(" ")]
				plt.plot(arr)
			plt.show()




