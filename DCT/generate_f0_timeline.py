import matplotlib
matplotlib.use('TkAgg')
import sys
# sys.path.append("/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/gen_audio/src/")
import my_tool
import numpy as np
import matplotlib.pyplot as plt
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
class SyllableF0():
	def __init__(self,time_length,val_list):
		self.time_length = float(time_length)
		self.time_f0 = []
		self.time_f0.append([0.0,val_list[0]])
		self.segment = self.time_length/(len(val_list)-1)
		for i in range(1,len(val_list)):
			self.time_f0.append([i*self.segment,val_list[i]])
		self.time_f0[-1][0] = self.time_length

	def get_f0_at_time(self,time):
		if time==self.time_length:
			return self.time_f0[-1][1]
		seg_i = int(time/self.segment)
		begin_time = self.time_f0[seg_i][0]
		begin_val = self.time_f0[seg_i][1]
		end_time = self.time_f0[seg_i+1][0]
		end_val = self.time_f0[seg_i+1][1]
		return begin_val+(end_val-begin_val)/(end_time-begin_time)*(time-begin_time)

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
	# print(syl_timeline)
	return syl_timeline

def GenerateF0TimeLine(phoneme_file,syllable_file,true_f0_file,out_file):
	syl_f0 = []
	with open(syllable_file) as f:
		for line in f:
			arr = line.strip().split(" ")
			syl_f0.append([arr[0],np.array(arr[1:len(arr)]).astype(np.float)])
	syl_timeline = FileSyllableSegment(phoneme_file,[sample[0] for sample in syl_f0])
	# print(syl_timeline)
	timeline = np.zeros((60000,))

	# if "01021" in syllable_file:
	# 	print(syl_f0)
	for i in range(len(syl_f0)):
		assert syl_f0[i][0]==syl_timeline[i][0]
		syl_begin = syl_timeline[i][1][0]
		syl_end = syl_timeline[i][1][1]
		time_obj = SyllableF0(syl_end-syl_begin,syl_f0[i][1])
		for i in range(int(syl_begin),int(syl_end)+1):
			timeline[i] = time_obj.get_f0_at_time(float(i)-syl_begin)
	with open(out_file,"w+") as f:
		line_counter = 0
		total_line = len(open(true_f0_file).readlines())
		for i in range(timeline.shape[0]):
			if (i+1)%5==0:
				line_counter += 1
				f.write(str(timeline[i])+"\n")
				if line_counter==total_line:
					break



if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', dest='mode')
	parser.add_argument('--predict_dir', dest='predict_dir')
	parser.add_argument('--out_dir', dest='out_dir')
	parser.add_argument('--true_f0_dir', dest='true_f0_dir')
	parser.add_argument('--file', dest='file')
	parser.add_argument('--file_suffix', dest='file_suffix',default="")

	parser.add_argument('--file1', dest='file1')
	parser.add_argument('--file2', dest='file2')

	parser.add_argument('--dir1', dest='dir1')
	parser.add_argument('--dir2', dest='dir2')
	parser.add_argument('--wav_dir1', dest='wav_dir1')
	parser.add_argument('--wav_dir2', dest='wav_dir2')
	parser.add_argument('--data_name',dest='data_name')


	args = parser.parse_args()
	mode = args.mode

	if mode=="how_to_run":
		print("python generate_f0_timeline.py"+
			" --mode generate_f0_timeline"+
			" --true_f0_dir ./f0_value"+
			" --predict_dir ../decision_tree/wagon/predict_f0_in_file"+
			" --out_dir ./predict_vs_true_contour"+
			" --file_suffix .f0")
		print("python generate_f0_timeline.py"+
			" --mode plot_predict_f0"+
			" --predict_dir ./predict_vs_true_contour/f0_val"+
			" --out_dir ./predict_vs_true_contour/plot")
		print("python generate_f0_timeline.py"+
			" --mode concat_move_plot"+
			" --predict_dir ./predict_vs_true_contour/f0_val"+
			" --out_dir ./predict_vs_true_contour/concat_move")
		print("python generate_f0_timeline.py"+
			" --mode slide_concat_f0"+
			" --predict_dir ./predicted_f0_dir/f0_val"+
			" --out_dir ./predicted_f0_dir/slide_concat")
		print("python generate_f0_timeline.py"+
			" --mode plot_two_file"+
			" --file1 ./predicted_f0_dir/f0_val/data_00013.f0"+
			" --file2 ./predicted_f0_dir/slide_concat/data_00013.f0")
		print("python generate_f0_timeline.py"+
			" --mode plot_two_file_with_syllable"+
			" --true_f0_dir ./f0_value"+
			" --predict_dir ../decision_tree/wagon/predict_f0_in_file"+
			" --dir1 ../test_f0_in_system/voice_lib/0/f0_val"+
			" --dir2 ../test_f0_in_system/voice_lib/1/f0_val"+
			" --wav_dir1 ../test_f0_in_system/voice_lib/0/wav"+
			" --wav_dir2 ../test_f0_in_system/voice_lib/1/wav"+
			" --data_name data_00013")

	if mode=="generate_f0_timeline":
		predict_file_f0_dir = args.predict_dir
		out_dir = args.out_dir
		file_suffix = args.file_suffix
		true_f0_dir = args.true_f0_dir
		# data_name = "data_00013"
		data_name_list = [data_name for data_name in os.listdir(predict_file_f0_dir) if "data" in data_name]
		# syl_data = my_tool.TxtDoneData("../../txt.done.data")
		# print(syl_data.keys())
		# print(FileSyllableSegment("./f0_value/data_00002.phoneme",syl_data["data_00002"]))
		os.system("rm -r "+out_dir)
		os.system("mkdir "+out_dir)
		os.system("mkdir "+out_dir+"/f0_val")
		for data_name in data_name_list:
			# print(data_name)
			phoneme_file = true_f0_dir+"/"+data_name+".phoneme"
			true_f0_file = true_f0_dir+"/"+data_name+".f0"
			syllable_file = predict_file_f0_dir+"/"+data_name
			out_file = out_dir+"/f0_val/"+data_name+file_suffix
			# print(out_file)
			GenerateF0TimeLine(phoneme_file,syllable_file,true_f0_file,out_file)
	if mode=="plot_predict_f0":
		predict_dir = args.predict_dir
		out_dir = args.out_dir
		data_name_list = [data_name for data_name in os.listdir(predict_dir) if "data" in data_name]
		os.system("rm -r "+out_dir)
		os.system("mkdir "+out_dir)
		for data_name in data_name_list:
			print(data_name)
			predict = np.loadtxt(predict_dir+"/"+data_name,delimiter=",")
			true = [] 
			with open("./f0_value/"+data_name+".f0") as f:
				for line in f:
					arr = line.strip().split(" ")
					if arr[0]=="":
						true.append(float(arr[1]))
					else:
						true.append(float(arr[0]))
			true = np.array(true)
			predict = predict[0:(np.nonzero(predict))[0][-1]+1]
			true = true[0:(np.nonzero(true))[0][-2]+1]
			# print(np.nonzero(predict))
			plt.scatter(range(len(predict)),predict,label="predict",s=1)
			plt.scatter(range(len(true)),true,label="true",s=1)
			plt.legend()
			plt.savefig(out_dir+"/"+data_name+".png")
			plt.clf()

	if mode=="concat_move_plot":
		predict_dir = args.predict_dir
		out_dir = args.out_dir
		os.system("rm -r "+out_dir)
		os.system("mkdir "+out_dir)
		for data_name in [data_name for data_name in os.listdir(predict_dir) if "data" in data_name]:
			print(data_name)
			f0_val = np.loadtxt(predict_dir+"/"+data_name,delimiter=",")
			# print(f0_val[1000:1020])
			###################################################################
			# plt.scatter(range(f0_val.shape[0]),f0_val,s=0.1,label="predict")
			###################################################################
			nonzero = np.where(f0_val>0)[0]
			segment_list = []
			begin = 0
			for i in range(1,nonzero.shape[0]):
				if nonzero[i]-nonzero[i-1]>3:
					segment_list.append(nonzero[begin:i])
					begin = i
			segment_list.append(nonzero[begin:nonzero.shape[0]])
			# print(segment_list)
			for seg in segment_list:
				seg_val = np.zeros(seg.shape)
				seg_val[0] = f0_val[seg[0]]
				for i in range(1,seg.shape[0]):
					if np.abs(f0_val[seg[i]]-f0_val[seg[i-1]])>2:
						seg_val[i] = seg_val[i-1]
					else:
						seg_val[i] = seg_val[i-1]+f0_val[seg[i]]-f0_val[seg[i-1]]
				for i in range(seg.shape[0]):
					f0_val[seg[i]] = seg_val[i]
			# print(f0_val[1000:1020])
			f0_val = f0_val[0:np.where(f0_val>0)[0][-1]+1]
			plt.scatter(range(f0_val.shape[0]),f0_val,s=1,label="processed_predict")

			original_f0 = []
			with open("./f0_value/"+data_name+".f0") as f:
				for line in f:
					arr = line.strip().split(" ")
					if arr[0]=="":
						original_f0.append(float(arr[1]))
					else:
						original_f0.append(float(arr[0]))
			original_f0 = np.array(original_f0)
			original_f0 = original_f0[0:np.where(original_f0>0)[0][-2]+1]
			plt.scatter(range(original_f0.shape[0]),original_f0,s=1,label="orignal",color="red")
			plt.legend()
			plt.savefig(out_dir+"/"+data_name+".png")
			plt.clf()

	if mode=="slide_concat_f0":
		predict_dir = args.predict_dir
		out_dir = args.out_dir
		os.system("rm -r "+out_dir)
		os.system("mkdir "+out_dir)
		os.system("mkdir "+out_dir+"/f0_val")
		for data_name in [data_name for data_name in os.listdir(predict_dir) if "data" in data_name]:
			# print(data_name)
			f0_val = np.loadtxt(predict_dir+"/"+data_name,delimiter=",")
			# print(f0_val[1000:1020])
			###################################################################
			# plt.scatter(range(f0_val.shape[0]),f0_val,s=0.1,label="predict")
			###################################################################
			nonzero = np.where(f0_val>0)[0]
			segment_list = []
			begin = 0
			for i in range(1,nonzero.shape[0]):
				if nonzero[i]-nonzero[i-1]>3:
					segment_list.append(nonzero[begin:i])
					begin = i
			segment_list.append(nonzero[begin:nonzero.shape[0]])
			# print(segment_list)
			for seg in segment_list:
				seg_val = np.zeros(seg.shape)
				seg_val[0] = f0_val[seg[0]]
				for i in range(1,seg.shape[0]):
					if np.abs(f0_val[seg[i]]-f0_val[seg[i-1]])>2:
						seg_val[i] = seg_val[i-1]
					else:
						seg_val[i] = seg_val[i-1]+f0_val[seg[i]]-f0_val[seg[i-1]]
				for i in range(seg.shape[0]):
					f0_val[seg[i]] = seg_val[i]
			# print(f0_val[1000:1020])
			np.savetxt(out_dir+"/f0_val/"+data_name,f0_val,delimiter=",",fmt="%.3f")

	if mode=="plot_two_file":
		file1 = args.file1
		file2 = args.file2
		arr1 = np.loadtxt(file1,delimiter=",")
		arr2 = np.loadtxt(file2,delimiter=",")
		plt.scatter(np.arange(len(arr1))*5,arr1,label="file1",s=1)
		plt.scatter(np.arange(len(arr2))*5,arr2,label="file2",s=1)
		plt.legend()
		plt.show()

	# if mode == "plot_two_file_with_syllable":
	# 	dir1 = args.dir1
	# 	dir2 = args.dir2
	# 	wav_dir1 = args.wav_dir1
	# 	wav_dir2 = args.wav_dir2

	# 	data_name = args.data_name
	# 	true_f0_dir = args.true_f0_dir
	# 	predict_file_f0_dir = args.predict_dir

	# 	phoneme_file = true_f0_dir+"/"+data_name+".phoneme"
	# 	true_f0_file = true_f0_dir+"/"+data_name+".f0"
	# 	syllable_file = predict_file_f0_dir+"/"+data_name
		
	# 	syl_f0 = []
	# 	with open(syllable_file) as f:
	# 		for line in f:
	# 			arr = line.strip().split(" ")
	# 			syl_f0.append([arr[0],np.array(arr[1:len(arr)]).astype(np.float)])
	# 	syl_timeline = FileSyllableSegment(phoneme_file,[sample[0] for sample in syl_f0])
	# 	# print(syl_timeline)

	# 	fig, ax = plt.subplots()

	# 	file1 = dir1+"/"+data_name+".f0"
	# 	file2 = dir2+"/"+data_name+".f0"
	# 	arr1 = np.loadtxt(file1,delimiter=",")
	# 	arr2 = np.loadtxt(file2,delimiter=",")
	# 	ax.scatter(np.arange(len(arr1))*5,arr1,label="file1",s=0.2)
	# 	ax.scatter(np.arange(len(arr2))*5,arr2,label="file2",s=0.2)

	# 	y = np.ones((len(syl_timeline),))*50
	# 	for i in range(len(y)):
	# 		y[i] += 10*(-1)**i

	# 	ax.scatter([tup[1][1] for tup in syl_timeline],y)
	# 	for i, tup in enumerate(syl_timeline):
	# 		ax.annotate(tup[0],(tup[1][1],y[i]),FontSize=5, rotation=45)
	# 		ax.axvline(x=tup[1][1],linestyle='-',linewidth=0.5, color='b')
	# 	plt.legend()
	# 	plt.savefig("./"+data_name+".jpg",dpi=200)
	# 	os.system("open "+data_name+".jpg")

	# 	my_input = "y"
	# 	while my_input=="y":
	# 		os.system("play "+wav_dir1+"/"+data_name+".wav")
	# 		os.system("play "+wav_dir2+"/"+data_name+".wav")
	# 		my_input = raw_input("reply? (y/n)")
			









