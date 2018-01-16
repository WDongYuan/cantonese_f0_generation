import sys
import os
import re
from sets import Set
import random
import time
import numpy as np
def LoadSyllableF0List(path):
	datafile_f0_dic = {}
	with open(path) as f:
		for line in f:
			line = line.strip()
			tmp_arr = line.split(",")
			tmp_file_name = "_".join(tmp_arr[0].split("_")[0:2])
			if tmp_file_name not in datafile_f0_dic:
				datafile_f0_dic[tmp_file_name] = []
			tmp_syllable = tmp_arr[0].split("_")[2]
			tmp_f0_value = [float(tmp_arr[i]) for i in range(1,len(tmp_arr))]
			datafile_f0_dic[tmp_file_name].append([tmp_syllable,tmp_f0_value])
	return datafile_f0_dic

def LoadFileSyllableFeature(dir):
	file_list = os.listdir(dir)
	datafile_feats_dic = {}
	for file_name in file_list:
		if "feats" not in file_name:
			continue
		tmp_filename = "_".join(re.split(r'_|\.',file_name)[0:2])
		datafile_feats_dic[tmp_filename] = []
		with open(dir+"/"+file_name) as f:
			for line in f:
				line = line.strip()
				arr = line.split(" ")
				datafile_feats_dic[tmp_filename].append([arr[0],arr])
	return datafile_feats_dic

def SaveDataFeature(datafile_f0_dic,datafile_feats_dic):
	os.system("mkdir feature_f0_data")
	os.system("cp ../dumpfeats/my_feature ./feature_f0_data/feature_list")
	for filename,f0_list in datafile_f0_dic.items():
		feats_list = datafile_feats_dic[filename]
		with open("feature_f0_data/"+filename,"w+") as f:
			for i in range(len(f0_list)):
				assert f0_list[i][0]==feats_list[i][0],f0_list[i][0]+" "+feats_list[i][0]
				f.write("feature:"+",".join(feats_list[i][1])+"\n")
				f.write("f0:"+",".join([str(num) for num in f0_list[i][1]])+"\n")

def CreateTrainData(read_dir,train_dir,feature_file,dct_num,word_with_tone):
	os.system("mkdir "+train_dir)
	file_token_list = []
	for i in range(dct_num):
		file_token_list.append(open(train_dir+"/dct_"+str(i),"w+"))

	################################################
	##Remove tone from word
	word_tone_idx = -1
	if not word_with_tone:
		line_counter = -1
		with open(feature_file) as f:
			for line in f:
				line_counter += 1
				if "Word.name" in line:
					word_tone_idx = line_counter
					break
	################################################

	data_file_list = sorted(os.listdir(read_dir))
	for file_name in data_file_list:
		if "data" not in file_name:
			continue
		tmp_file = open(read_dir+"/"+file_name)
		tmp_line = tmp_file.readline().strip()
		while tmp_line!="":
			tmp_feature = tmp_line.split(":")[1].split(",")

			################################################
			##Remove tone from word
			if (not word_with_tone) and word_tone_idx!=-1:
				tmp_feature[word_tone_idx] = tmp_feature[word_tone_idx][0:-1]
			################################################

			tmp_line = tmp_file.readline()
			tmp_dct_list = [float(val) for val in tmp_line.split(":")[1].split(",")]
			for dct_i in range(dct_num):
				file_token_list[dct_i].write(str(tmp_dct_list[dct_i])+" ")
				file_token_list[dct_i].write(" ".join(tmp_feature)+"\n")
			tmp_line = tmp_file.readline().strip()
	for tmp_file in file_token_list:
		tmp_file.close()
	return

def RecordDataOrder(file_list,output_file):
	outfile = open(output_file,"w+")
	for file_path in file_list:
		with open(file_path) as f:
			for line in f:
				if "feature" not in line:
					continue
				feature = line.split(":")[1]
				syllable = feature.split(",")[0]
				outfile.write(file_path.split("/")[-1]+" "+syllable+"\n")
	outfile.close()
	return


def CreateTrainDataVector(read_dir,train_dir,f0_vector_file,feature_file,word_with_tone):
	# os.system("mkdir "+train_dir)
	data_file_list = sorted([file_name for file_name in os.listdir(read_dir) if "data" in file_name])
	line_counter = -1
	RecordDataOrder([read_dir+"/"+file_name for file_name in data_file_list],train_dir+"/syllable_map")
	with open(f0_vector_file,"w+") as f0_f:
		for file_name in data_file_list:
			file_token = open(read_dir+"/"+file_name)
			content = file_token.readlines()
			file_token.close()
			for i in range(len(content)):
				if "f0" in content[i]:
					# print(content[i])
					line_counter += 1
					f0_f.write(" ".join(content[i].strip().split(":")[1].split(","))+"\n")
					content[i] = "f0:"+str(int(line_counter))+"\n"
			file_token = open(read_dir+"/"+file_name,"w+")
			file_token.writelines(content)
			file_token.close()
	CreateTrainData(read_dir,train_dir,feature_file,1,word_with_tone)

	return

def CreateFeatureDesc(train_dir,feature_file,desc_file,my_desc_file,predict_vector):
	known_feature = {}
	with open(my_desc_file) as f:
		for line in f:
			line = line.strip()
			arr = line.split(" ")
			known_feature[arr[0]] = " ".join(arr[1:len(arr)])

	desc_token = open(desc_file,"w+")
	if predict_vector:
		desc_token.write("((dct_coef vector)\n")
	else:
		desc_token.write("((dct_coef float)\n")

	feature_list = []
	with open(feature_file) as f:
		for line in f:
			line = line.strip()
			feature_list.append([line,Set()])

	train_file_list = os.listdir(train_dir)
	for file_name in train_file_list:
		if "dct" not in file_name:
			continue
		with open(train_dir+"/"+file_name) as f:
			for line in f:
				line = line.strip()
				arr = line.split(" ")
				arr = arr[1:len(arr)]
				for i in range(len(feature_list)):
					feature_list[i][1].add(arr[i])

	for feature_name,val_set in feature_list:
		desc_token.write("("+feature_name)
		if feature_name in known_feature:
			desc_token.write(" "+known_feature[feature_name])
		elif IsSetOfNumber(val_set) and len(val_set)>10:
			desc_token.write(" float")
		else:
			for val in val_set:
				desc_token.write(" "+val)
		desc_token.write(")\n")
	desc_token.write(")")

def SampleData(dct_data_dir,train_ratio,output_dir):
	os.system("mkdir "+output_dir)
	os.system("mkdir "+output_dir+"/train")
	os.system("mkdir "+output_dir+"/dev")
	file_list = []
	train_file_list = []
	dev_file_list = []

	file_name_list = os.listdir(dct_data_dir)
	for file_name in file_name_list:
		if "dct" not in file_name:
			continue
		file_list.append(open(dct_data_dir+"/"+file_name))
		train_file_list.append(open(output_dir+"/train/"+file_name,"w+"))
		dev_file_list.append(open(output_dir+"/dev/"+file_name,"w+"))

	random.seed(time.time())
	file_end = False
	while not file_end:
		if random.random()<train_ratio:
			for i in range(len(file_list)):
				line = file_list[i].readline().strip()
				if line=="":
					file_end = True
					break
				train_file_list[i].write(line+"\n")
		else:
			for i in range(len(file_list)):
				line = file_list[i].readline().strip()
				if line=="":
					file_end = True
					break
				dev_file_list[i].write(line+"\n")

	for i in range(len(file_list)):
		file_list[i].close()
		train_file_list[i].close()
		dev_file_list[i].close()

	##Normalize the train_data
	return

def IsSetOfNumber(my_set):
	for val in my_set:
		try:
			float(val)
			return True
		except ValueError:
			return False

def Normalization(arr):
	norm_arr = np.array(arr)
	std_dev = np.std(norm_arr)
	mean = np.mean(norm_arr)
	norm_arr = (norm_arr-mean)/std_dev
	return norm_arr,mean,std_dev

def NormalizeFile(infile,col,outfile):
	content = []
	with open(infile) as f:
		for line in f:
			line = line.strip()
			arr = line.split(" ")
			content.append(arr)

	##Normalize the data put back into list
	record = []
	for tmp_col in col:
		ori_val = np.array([float(sample[tmp_col]) for sample in content])
		norm_val,mean,std_dev = Normalization(ori_val)
		for i in range(len(norm_val)):
			content[i][tmp_col] = str(norm_val[i])
		record.append([tmp_col,mean,std_dev])

	##Rewrite the file after normalization
	with open(infile,"w+") as f:
		for sample in content:
			f.write(" ".join(sample)+"\n")

	##Write the mean and std_dev in file
	with open(outfile,"w+") as f:
		for tmp_record in record:
			f.write("col_"+str(tmp_record[0])+" "+str(tmp_record[1])+" "+str(tmp_record[2])+"\n")
	return




if __name__=="__main__":
	if len(sys.argv)==1:
		print("1. Collect dct coef data and feature data from specific directory.")
		print("python read_data.py collect_data dct/syllable")
		# print("python read_data.py create_train_data")
		# print("python read_data.py sample_train_dev_data ./train_data 0.8 ./sample_data")
		print("python read_data.py create_train_dev_data 0.8(ratio for train data)")
		print("python read_data.py create_train_dev_data_vector 0.8(ratio for train data)")
		print("python read_data.py create_train_dev_data_vector_with_dev_set ./dev_file")
		print("")
		print("2. Modify the description file in ./train_data")
		exit()

	mode = sys.argv[1]
	if mode=="collect_data":
		data_type = sys.argv[2]
		##Collect data from ../DCT and ../dumpfeats, save to ./features_f0_data
		datafile_f0_dic = None
		if data_type=="dct":
			datafile_f0_dic = LoadSyllableF0List("../DCT/syllable_f0_dct_representation.save")
		elif data_type=="syllable":
			datafile_f0_dic = LoadSyllableF0List("../DCT/subsample_f0.save")
		datafile_feats_dic = LoadFileSyllableFeature("../dumpfeats/data_feature")
		os.system("cp ../dumpfeats/my_feature feature_list")
		SaveDataFeature(datafile_f0_dic,datafile_feats_dic)

	if mode=="create_train_dev_data_vector":
		train_ratio = float(sys.argv[2])
		os.system("rm -r train_dev_data_vector")
		os.system("mkdir train_dev_data_vector")
		read_dir = "./feature_f0_data"

		##Select some utt file as train file and dev file
		file_list = [file_name for file_name in os.listdir(read_dir) if "data" in file_name]
		random.shuffle(file_list)
		train_file_list = file_list[0:int(len(file_list)*train_ratio)]
		dev_file_list = file_list[int(len(file_list)*train_ratio):len(file_list)]
		os.system("mkdir train_dev_data_vector/train_file_buffer")
		os.system("mkdir train_dev_data_vector/dev_file_buffer")

		##Copy train utt file into a buffer folder
		for tmp_train_file in train_file_list:
			os.system("cp "+read_dir+"/"+tmp_train_file+" train_dev_data_vector/train_file_buffer")
		##Copy dev utt file into a buffer folder
		for tmp_dev_file in dev_file_list:
			os.system("cp "+read_dir+"/"+tmp_dev_file+" train_dev_data_vector/dev_file_buffer")

		word_with_tone = False
		##Create the train data
		train_dir = "./train_dev_data_vector/train_data"
		feature_file = "./feature_list"
		os.system("mkdir "+train_dir)
		CreateTrainDataVector("train_dev_data_vector/train_file_buffer",train_dir,"train_dev_data_vector/train_data_f0_vector",feature_file,word_with_tone)

		##Create the dev data
		dev_dir = "./train_dev_data_vector/dev_data"
		feature_file = "./feature_list"
		os.system("mkdir "+dev_dir)
		CreateTrainDataVector("train_dev_data_vector/dev_file_buffer",dev_dir,"train_dev_data_vector/dev_data_f0_vector",feature_file,word_with_tone)


		os.system("rm -r train_dev_data_vector/train_file_buffer")
		os.system("rm -r train_dev_data_vector/dev_file_buffer")

		# NormalizeFile("train_dev_data_vector/train_data_f0_vector",range(10),"train_dev_data_vector/train_data_f0_vector_norm")
		# NormalizeFile("train_dev_data_vector/dev_data_f0_vector",range(10),"train_dev_data_vector/dev_data_f0_vector_norm")

		##Create the feature description file for the train data feature
		desc_file = "./train_dev_data_vector/feature_desc"
		my_desc_file = "./my_feature_desc"
		feature_file = "./feature_list"
		predict_vector = True
		CreateFeatureDesc(train_dir,feature_file,desc_file,my_desc_file,predict_vector)



	#################################################################################
	# if mode=="create_train_data":
	# 	##Read the data in ./feature_f0_data and save as data to be train in ./train_data
	# 	read_dir = "./feature_f0_data"
	# 	train_dir = "./train_data"
	# 	dct_num = 10
	# 	word_with_tone = False
	# 	feature_file = "./feature_list"
	# 	CreateTrainData(read_dir,train_dir,feature_file,dct_num,word_with_tone)

	# 	##Create the feature description file for the train data feature
	# 	desc_file = "./train_data/feature_desc"
	# 	my_desc_file = "./train_data/my_feature_desc"
	# 	feature_file = "./feature_list"
	# 	CreateFeatureDesc(train_dir,feature_file,desc_file,my_desc_file)
	#################################################################################

	if mode=="create_train_dev_data":
		train_ratio = float(sys.argv[2])
		os.system("rm -r train_dev_data")
		os.system("mkdir train_dev_data")
		read_dir = "./feature_f0_data"

		##Select some utt file as train file and dev file
		file_list = [file_name for file_name in os.listdir(read_dir) if "data" in file_name]
		random.shuffle(file_list)
		train_file_list = file_list[0:int(len(file_list)*train_ratio)]
		dev_file_list = file_list[int(len(file_list)*train_ratio):len(file_list)]
		os.system("mkdir train_dev_data/train_file_buffer")
		os.system("mkdir train_dev_data/dev_file_buffer")

		##Copy train utt file into a buffer folder
		for tmp_train_file in train_file_list:
			os.system("ln "+read_dir+"/"+tmp_train_file+" train_dev_data/train_file_buffer")
		##Copy dev utt file into a buffer folder
		for tmp_dev_file in dev_file_list:
			os.system("ln "+read_dir+"/"+tmp_dev_file+" train_dev_data/dev_file_buffer")

		dct_num = 10
		word_with_tone = False

		##Create the train data
		train_dir = "./train_dev_data/train_data"
		feature_file = "./feature_list"
		CreateTrainData("train_dev_data/train_file_buffer",train_dir,feature_file,dct_num,word_with_tone)

		##Create the dev data
		dev_dir = "./train_dev_data/dev_data"
		feature_file = "./feature_list"
		CreateTrainData("train_dev_data/dev_file_buffer",dev_dir,feature_file,dct_num,word_with_tone)

		os.system("rm -r train_dev_data/train_file_buffer")
		os.system("rm -r train_dev_data/dev_file_buffer")

		##Create the feature description file for the train data feature
		desc_file = "./train_dev_data/feature_desc"
		my_desc_file = "./my_feature_desc"
		feature_file = "./feature_list"
		predict_vector = False
		CreateFeatureDesc(train_dir,feature_file,desc_file,my_desc_file,predict_vector)
	#################################################################################

	if mode=="sample_train_dev_data":
		dct_data_file = sys.argv[2]
		train_ratio = float(sys.argv[3])
		output_dir = sys.argv[4]
		SampleData(dct_data_file,train_ratio,output_dir)

	if mode=="normalize_test":
		NormalizeFile("./dump/dct_0",[0],"./dump/normalize_file")

	if mode=="create_train_dev_data_vector_with_dev_set":
		print("running...")
		dev_list = sys.argv[2]
		os.system("rm -r train_dev_data_vector")
		os.system("mkdir train_dev_data_vector")
		read_dir = "./feature_f0_data"

		##Select some utt file as train file and dev file
		file_list = [file_name for file_name in os.listdir(read_dir) if "data" in file_name]
		# random.shuffle(file_list)
		# train_file_list = file_list[0:int(len(file_list)*train_ratio)]
		# dev_file_list = file_list[int(len(file_list)*train_ratio):len(file_list)]
		dev_file_list = []
		with open(dev_list) as f:
			for line in f:
				dev_file_list.append(line.strip())
		train_file_list = [file_name for file_name in file_list if file_name not in dev_file_list]
		os.system("mkdir train_dev_data_vector/train_file_buffer")
		os.system("mkdir train_dev_data_vector/dev_file_buffer")

		##Copy train utt file into a buffer folder
		for tmp_train_file in train_file_list:
			os.system("cp "+read_dir+"/"+tmp_train_file+" train_dev_data_vector/train_file_buffer")
		##Copy dev utt file into a buffer folder
		for tmp_dev_file in dev_file_list:
			os.system("cp "+read_dir+"/"+tmp_dev_file+" train_dev_data_vector/dev_file_buffer")

		word_with_tone = False
		##Create the train data
		train_dir = "./train_dev_data_vector/train_data"
		feature_file = "./feature_list"
		os.system("mkdir "+train_dir)
		CreateTrainDataVector("train_dev_data_vector/train_file_buffer",train_dir,"train_dev_data_vector/train_data_f0_vector",feature_file,word_with_tone)

		##Create the dev data
		dev_dir = "./train_dev_data_vector/dev_data"
		feature_file = "./feature_list"
		os.system("mkdir "+dev_dir)
		CreateTrainDataVector("train_dev_data_vector/dev_file_buffer",dev_dir,"train_dev_data_vector/dev_data_f0_vector",feature_file,word_with_tone)


		os.system("rm -r train_dev_data_vector/train_file_buffer")
		os.system("rm -r train_dev_data_vector/dev_file_buffer")

		# NormalizeFile("train_dev_data_vector/train_data_f0_vector",range(10),"train_dev_data_vector/train_data_f0_vector_norm")
		# NormalizeFile("train_dev_data_vector/dev_data_f0_vector",range(10),"train_dev_data_vector/dev_data_f0_vector_norm")

		##Create the feature description file for the train data feature
		desc_file = "./train_dev_data_vector/feature_desc"
		my_desc_file = "./my_feature_desc"
		feature_file = "./feature_list"
		predict_vector = True
		CreateFeatureDesc(train_dir,feature_file,desc_file,my_desc_file,predict_vector)




