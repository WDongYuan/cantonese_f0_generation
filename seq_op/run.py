import os
import argparse
import numpy as np
from predict_func import *
from rf import RandomForest
import warnings
warnings.filterwarnings('error')

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def read_operation(path):
	op = {}
	with open(path) as f:
		for line in f:
			line = line.strip()
			if line=="END":
				break
			line = line.split(" ")
			op_name = line[0]
			op_val = None
			# if line[1]=="0" or line[1]=="1":
			# 	op_val = True if line[1]=="1" else False
			if is_number(line[1]):
				op_val = float(line[1])
			else:
				op_val = line[1]
			op[op_name] = op_val
	return op
def norm_row(in_file,out_dir):
	file_name = in_file.split("/")[-1]
	out_file = out_dir+"/"+file_name+"_norm_row"
	norm_record = out_file+"_record"
	os.system("python ../decision_tree/wagon/data_processing.py"+
		" --mode normalize_row"+
		" --in_file "+in_file+
		" --out_file "+out_file+
		" --record_file "+norm_record)
	return out_file,norm_record

def norm_global(in_file,out_dir):
	file_name = in_file.split("/")[-1]
	out_file = out_dir+"/"+file_name+"_norm_global"
	norm_record = out_file+"_record"
	os.system("python ../decision_tree/wagon/data_processing.py"+
		" --mode normalize_global"+
		" --in_file "+in_file+
		" --out_file "+out_file+
		" --record_file "+norm_record)
	return out_file,norm_record

def dct_row(in_file,out_dir,number):
	file_name = in_file.split("/")[-1]
	out_file = out_dir+"/"+file_name+"_dct"
	os.system("python ../decision_tree/wagon/data_processing.py"+
		" --mode dct_row"+
		" --in_file "+in_file+
		" --out_file "+out_file+
		" --number "+str(number))
	return out_file

def norm_col(in_file,out_dir,col_num=10):
	file_name = in_file.split("/")[-1]
	out_file = out_dir+"/"+file_name+"_norm_col"
	norm_record = out_file+"_record"
	col_list = " ".join([str(i) for i in range(col_num)])
	os.system("python ../decision_tree/wagon/data_processing.py"+
		" --mode normalize_col"+
		" --in_file "+in_file+
		" --out_file "+out_file+
		" --record_file "+norm_record+
		" --col_list "+col_list)
	return out_file,norm_record

def idct_row(in_file,out_dir,number):
	file_name = in_file.split("/")[-1]
	out_file = out_dir+"/"+file_name+"_idct"
	os.system("python ../decision_tree/wagon/data_processing.py"+
		" --mode idct_row"+
		" --in_file "+in_file+
		" --out_file "+out_file+
		" --number "+str(number))
	return out_file

def log_row(in_file,out_dir):
	file_name = in_file.split("/")[-1]
	out_file = out_dir+"/"+file_name+"_log"
	arr = np.loadtxt(in_file,delimiter=" ")
	arr = np.log(arr)
	np.savetxt(out_file,arr,delimiter=" ",fmt="%.5f")
	return out_file

def exp_row(in_file,out_dir):
	file_name = in_file.split("/")[-1]
	out_file = out_dir+"/"+file_name+"_exp"
	arr = np.loadtxt(in_file,delimiter=" ")
	arr = np.exp(arr)
	np.savetxt(out_file,arr,delimiter=" ",fmt="%.5f")
	return out_file


def unnorm_col(in_file,record_file,out_dir):
	file_name = in_file.split("/")[-1]
	out_file = out_dir+"/"+file_name+"_unnrom_col"
	os.system("python ../decision_tree/wagon/data_processing.py"+
		" --mode unnormalize_col"+
		" --in_file "+in_file+
		" --out_file "+out_file+
		" --record_file "+record_file)
	return out_file

def unnorm_global(in_file,record_file,out_dir):
	file_name = in_file.split("/")[-1]
	out_file = out_dir+"/"+file_name+"_unnrom_global"
	os.system("python ../decision_tree/wagon/data_processing.py"+
		" --mode unnormalize_global"+
		" --in_file "+in_file+
		" --out_file "+out_file+
		" --record_file "+record_file)
	return out_file

def row_mean_std(in_file,out_dir):
	file_name = in_file.split("/")[-1]
	out_file = out_dir+"/"+file_name+"_mean_std"
	os.system("python ../decision_tree/wagon/data_processing.py"+
		" --mode row_mean_std"+
		" --in_file "+in_file+
		" --record_file "+out_file)
	return out_file

def unnorm_mean_std(in_file,mean,std,true_mean_std,predict_mean_std,out_dir):
	true_mean_std = np.loadtxt(true_mean_std)
	predict_mean_std = np.loadtxt(predict_mean_std)
	mean_val = true_mean_std[:,0] if mean=="true" else predict_mean_std[:,0]
	std_val = true_mean_std[:,1] if std=="true" else predict_mean_std[:,1]
	mean_std_name = mean+"_mean_"+std+"_std"
	np.savetxt(out_dir+"/"+mean_std_name,np.vstack((mean_val,std_val)))

	file_name = in_file.split("/")[-1]
	out_file = out_dir+"/"+file_name+"_unnrom_"+mean_std_name
	record_file = out_dir+"/"+mean_std_name
	os.system("python ../decision_tree/wagon/data_processing.py"+
		" --mode unnormalize_row"+
		" --in_file "+in_file+
		" --out_file "+out_file+
		" --record_file "+record_file)
	return out_file

def test_statistics(predict_file,true_file,out_dir):
	os.system("python ../decision_tree/wagon/run.py"+
		" --mode predict_statistic"+
		" --file1 "+predict_file+
		" --file2 "+true_file+
		" > "+out_dir+"/"+"test_statistics")
	print("saving test statistics to "+out_dir+"/"+"test_statistics")

def put_predict_f0_in_file(in_file,syllable_map,out_dir,first_syllable_flag=1):
	out_file = out_dir+"/predict_f0_in_file"
	os.system("mkdir "+out_file)
	os.system("python ../decision_tree/wagon/run.py"+
		" --mode put_back_f0_in_file"+
		" --f0_file_map "+syllable_map+
		" --f0_val "+in_file+
		" --first_syllable_flag "+str(first_syllable_flag)+
		" --out_dir "+out_file)
	print("saving result to "+out_file)
	return out_file

def generate_f0_time_line(in_file,out_dir,true_f0_dir):
	out_file = out_dir+"/f0_timeline"
	os.system("mkdir "+out_file)
	os.system("python ../DCT/generate_f0_timeline.py"+
		" --mode generate_f0_timeline"+
		" --true_f0_dir "+true_f0_dir+
		" --predict_dir "+in_file+
		" --out_dir "+out_file+
		" --file_suffix .f0")
	out_file = out_file+"/f0_val"
	print("f0 timeline is saved to "+out_file)
	return out_file

def slide(in_file,out_dir):
	out_file = out_dir+"/slide_f0"
	os.system("mkdir "+out_file)
	os.system("python ../DCT/generate_f0_timeline.py"+
		" --mode slide_concat_f0"+
		" --predict_dir "+in_file+
		" --out_dir "+out_file)
	out_file = out_file+"/f0_val"
	print("slided f0 contour is saved to "+out_file)
	return out_file

def generate_wav(in_file,ccoef_dir,out_dir):
	prefix_name = "/".join(in_file.split("/")[0:-1])
	os.system("python ../test_f0_in_system/run.py"+
		" --mode synthesis_with_f0"+
		" --awb_synth ../test_f0_in_system/my_synth_f0"+
		" --ccoef_dir "+ccoef_dir+
		" --f0_dir "+in_file+
		" --out_dir "+prefix_name+"/wav")
	out_file = prefix_name+"/wav"
	print("waveform files saved to "+out_file)
	return out_file

def only_use_mean(in_file,out_dir):
	file_name = in_file.split("/")[-1]
	out_file = out_dir+"/"+file_name+"_use_mean"
	f0 = np.loadtxt(in_file,delimiter=" ")
	mean = np.multiply(np.ones(f0.shape),f0.mean(axis=1).reshape(-1,1))
	np.savetxt(out_file,mean,delimiter=" ",fmt="%.5f")
	return out_file

def padding(in_file,out_dir):
	file_name = in_file.split("/")[-1]
	data = np.loadtxt(in_file,delimiter=" ")
	new_data = np.zeros((data.shape[0],10))
	new_data[:,0:data.shape[1]] = data
	out_file = out_dir+"/"+file_name+"_pad"
	np.savetxt(out_file,new_data,delimiter=" ")
	pad_num = data.shape[1]
	return out_file,pad_num

def unpadding(in_file,pad_num,out_dir):
	file_name = in_file.split("/")[-1]
	data = np.loadtxt(in_file,delimiter=" ")
	out_file = out_dir+"/"+file_name+"_unpad"
	np.savetxt(out_file,data[:,0:pad_num],delimiter=" ")
	return out_file

# def get_f0_delta(in_file,out_dir,delta_rate):
# 	file_name = in_file.split("/")[-1]
# 	out_file = out_dir+"/"+file_name+"_delta"
# 	delta_num = 9
# 	arr = np.loadtxt(in_file)
# 	row,col = arr.shape
# 	delta = arr[:,1:col]-arr[:,0:col-1]
# 	np.savetxt(out_file,np.hstack((arr,delta*delta_rate)),delimiter=" ",fmt="%.5f")
# 	return out_file,delta_num

def get_f0_delta(in_file,map_file,out_dir,delta_rate):
	#delta and delta-delta feature for the mean of previous, current and next syllable f0 data
	file_name = in_file.split("/")[-1]
	delta_num = -1
	out_file = out_dir+"/"+file_name+"_delta"
	with open(map_file) as mapf:
		map_lines = mapf.readlines()
		data = np.loadtxt(in_file,delimiter=" ")
		delta_num = 2*data.shape[1]
		assert len(map_lines)==len(data)
		delta_arr = np.zeros((len(map_lines),delta_num))
		for i in range(len(map_lines)):
			pre_data_name = "" if i==0 else map_lines[i-1].split(" ")[0]
			data_name = map_lines[i].split(" ")[0]
			next_data_name = "" if i==len(map_lines)-1 else map_lines[i+1].split(" ")[0]

			if data_name == pre_data_name:
				delta_arr[i][0:data.shape[1]] = data[i]-data[i-1]
			if data_name == next_data_name:
				delta_arr[i][data.shape[1]:] = data[i+1]-data[i]-delta_arr[i][0:data.shape[1]]
		np.savetxt(out_file,np.hstack((data,delta_arr*delta_rate)),delimiter=" ",fmt="%.5f")
	return out_file,delta_num

def get_mean_delta(in_file,map_file,out_dir,delta_rate):
	#delta and delta-delta feature for the mean of previous, current and next syllable f0 data
	file_name = in_file.split("/")[-1]
	delta_num = 3
	out_file = out_dir+"/"+file_name+"_delta"
	with open(map_file) as mapf:
		map_lines = mapf.readlines()
		data = np.loadtxt(in_file,delimiter=" ")
		assert len(map_lines)==len(data)
		delta_arr = np.zeros((len(map_lines),delta_num))
		for i in range(len(map_lines)):
			pre_data_name = "" if i==0 else map_lines[i-1].split(" ")[0]
			data_name = map_lines[i].split(" ")[0]
			next_data_name = "" if i==len(map_lines)-1 else map_lines[i+1].split(" ")[0]

			if data_name == pre_data_name:
				delta_arr[i][0] = data[i].mean()-data[i-1].mean()
			if data_name == next_data_name:
				delta_arr[i][1] = data[i+1].mean()-data[i].mean()
			delta_arr[i][2] = delta_arr[i][1]-delta_arr[i][0]
		np.savetxt(out_file,np.hstack((data,delta_arr*delta_rate)),delimiter=" ",fmt="%.5f")
	return out_file,delta_num

def get_dct_delta(in_file,map_file,out_dir,delta_rate):
	#delta and delta-delta feature for the first coefficient of syllable dct
	file_name = in_file.split("/")[-1]
	delta_num = 3
	out_file = out_dir+"/"+file_name+"_delta"
	with open(map_file) as mapf:
		map_lines = mapf.readlines()
		data = np.loadtxt(in_file,delimiter=" ")
		assert len(map_lines)==len(data)
		delta_arr = np.zeros((len(map_lines),delta_num))
		for i in range(len(map_lines)):
			pre_data_name = "" if i==0 else map_lines[i-1].split(" ")[0]
			data_name = map_lines[i].split(" ")[0]
			next_data_name = "" if i==len(map_lines)-1 else map_lines[i+1].split(" ")[0]

			if data_name == pre_data_name:
				delta_arr[i][0] = data[i][0]-data[i-1][0]
			if data_name == next_data_name:
				delta_arr[i][1] = data[i+1][0]-data[i][0]
			delta_arr[i][2] = delta_arr[i][1]-delta_arr[i][0]
		np.savetxt(out_file,np.hstack((data,delta_arr*delta_rate)),delimiter=" ",fmt="%.5f")
	return out_file,delta_num
def get_self_delta(in_file,out_dir,delta_rate):
	file_name = in_file.split("/")[-1]
	out_file = out_dir+"/"+file_name+"_delta"
	data = np.loadtxt(in_file,delimiter=" ")
	delta_num = data.shape[1]-1
	delta = data[:,1:]-data[:,0:-1]
	np.savetxt(out_file,np.hstack((data,delta*delta_rate)),delimiter=" ",fmt="%.5f")
	return out_file,delta_num

def get_mean_std_delta(in_file,map_file,out_dir,delta_rate):
	#delta and delta-delta feature for the first coefficient of syllable dct
	file_name = in_file.split("/")[-1]
	delta_num = 3
	out_file = out_dir+"/"+file_name+"_msdelta"
	with open(map_file) as mapf:
		map_lines = mapf.readlines()
		data = np.loadtxt(in_file,delimiter=" ")
		assert len(map_lines)==len(data)
		delta_arr = np.zeros((len(map_lines),delta_num,2))
		for i in range(len(map_lines)):
			pre_data_name = "" if i==0 else map_lines[i-1].split(" ")[0]
			data_name = map_lines[i].split(" ")[0]
			next_data_name = "" if i==len(map_lines)-1 else map_lines[i+1].split(" ")[0]

			if data_name == pre_data_name:
				delta_arr[i][0] = data[i]-data[i-1]
			if data_name == next_data_name:
				delta_arr[i][1] = data[i+1]-data[i]
			delta_arr[i][2] = delta_arr[i][1]-delta_arr[i][0]
		delta_arr = delta_arr.reshape((len(map_lines),delta_num*2))
		np.savetxt(out_file,np.hstack((data,delta_arr*delta_rate)),delimiter=" ",fmt="%.5f")
	return out_file,delta_num*2

def get_small_data(file_list,ratio,out_dir):
	out_file_list = []
	for file in file_list:
		file_name = file.split("/")[-1]
		out_file = out_dir+"/"+file_name+"_small"
		with open(file) as f:
			lines = f.readlines()
			with open(out_file,"w+") as outf:
				outf.writelines(lines[0:int(ratio*len(lines))])
		out_file_list.append(out_file)
	return out_file_list

def timeline_statistics(predict_f0,true_f0):
	os.system("python ../decision_tree/wagon/data_processing.py"+
		" --mode timeline_predict_statistics"+
		" --true_f0_dir "+true_f0+
		" --predict_f0_dir "+predict_f0)

def idct_phrase(phrase_syllable_dir,dct_dir,out_dir):
	out_dir += "/idct_phrase_f0"
	os.system("mkdir "+out_dir)
	os.system("python ../decision_tree/data_preprocessing.py"+
		" --mode idct_phrase_f0_dir"+
		" --phrase_syl_dir "+phrase_syllable_dir+
		" --dct_dir "+dct_dir+
		" --out_dir "+out_dir)
	return out_dir

def map_f0(f0_dir,map_file,out_file):
	os.system("python ../decision_tree/data_preprocessing.py"+
		" --mode map_to_new_f0_vector"+
		" --map_file "+map_file+
		" --f0_dir "+f0_dir+
		" --out_file "+out_file+
		" --add_index_prefix 0")

def vector_utt_stat(predict_file,true_file,syl_map):
	dic = {}
	predict = np.loadtxt(predict_file,delimiter=" ")
	predict = predict.reshape((predict.shape[0],-1))
	true = np.loadtxt(true_file,delimiter=" ")
	true = true.reshape((true.shape[0],-1))

	smap = []
	with open(syl_map) as f:
		smap = f.readlines()
	assert len(smap)==len(predict)
	assert len(predict)==len(true)
	for i in range(len(smap)):
		data_name = smap[i].strip().split(" ")[0]
		if data_name not in dic:
			dic[data_name] = [[],[]]
		dic[data_name][0].append(true[i])
		dic[data_name][1].append(predict[i])

	stat = []
	for data_name,f0 in dic.items():
		tmp_true = np.hstack(f0[0])
		tmp_predict = np.hstack(f0[1])
		rmse = np.sqrt(np.square(tmp_true-tmp_predict).mean())
		cor = np.corrcoef(tmp_true,tmp_predict)[0][1]
		stat.append([rmse,cor])
	stat = np.array(stat).mean(axis=0)
	print("rmse: "+str(stat[0]))
	print("cor: "+str(stat[1]))
	return

def vector_syl_stat(predict_file,true_file):
	predict = np.loadtxt(predict_file,delimiter=" ")
	true = np.loadtxt(true_file,delimiter=" ")
	print("rmse: "+str(np.sqrt(np.square(true-predict).mean(axis=1)).mean()))
	cor = np.zeros((len(predict),))

	bias = np.zeros((predict.shape[1],))
	bias[0] += 0.00000001

	for row in range(len(predict)):
		try:
			cor[row] = np.corrcoef(predict[row]+bias,true[row]+bias)[0][1]
		except Warning:
			pass
	print("cor: "+str(cor.mean()))


def mean_std_syl_stat(predict_mean_std,true_mean_std):
	predict = np.loadtxt(predict_mean_std,delimiter=" ")
	true = np.loadtxt(true_mean_std,delimiter=" ")
	print("mean rmse: "+str(np.abs(predict[:,0]-true[:,0]).mean()))
	print("std rmse: "+str(np.abs(predict[:,1]-true[:,1]).mean()))

def mean_std_utt_stat(predict_mean_std,true_mean_std,syl_map):
	dic = {}
	predict = np.loadtxt(predict_mean_std,delimiter=" ")
	true = np.loadtxt(true_mean_std,delimiter=" ")

	print("mean:")
	tmp_true = "_my_true"
	np.savetxt(tmp_true,true[:,0].reshape(true.shape[0],1))
	tmp_predict = "_my_predict"
	np.savetxt(tmp_predict,predict[:,0].reshape(predict.shape[0],1))
	vector_utt_stat(tmp_predict,tmp_true,syl_map)
	print("----------------------------------------")
	print("std:")
	tmp_true = "_my_true"
	np.savetxt(tmp_true,true[:,1].reshape(true.shape[0],1))
	tmp_predict = "_my_predict"
	np.savetxt(tmp_predict,predict[:,1].reshape(predict.shape[0],1))
	vector_utt_stat(tmp_predict,tmp_true,syl_map)

	os.system("rm "+tmp_true)
	os.system("rm "+tmp_predict)

	return


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode',dest='mode')
	parser.add_argument('--voice_dir', dest='voice_dir')
	parser.add_argument('--operation_file', dest='op_file',default="./operation.desc")
	parser.add_argument('--train_data',dest='train_data')
	parser.add_argument('--train_label',dest='train_label')
	parser.add_argument('--dev_data',dest='dev_data')
	parser.add_argument('--dev_label',dest='dev_label')
	parser.add_argument('--out_dir',dest='out_dir')
	parser.add_argument('--train_syllable_map',dest='train_syllable_map')
	parser.add_argument('--dev_syllable_map',dest='dev_syllable_map')
	parser.add_argument('--decompose_desc',dest='decompose_desc',default="./decompose_desc")
	parser.add_argument('--vector_feat_desc',dest='vector_feat_desc')
	parser.add_argument('--val_feat_desc',dest='val_feat_desc')
	parser.add_argument('--true_f0_dir',dest='true_f0_dir')
	parser.add_argument('--ccoef_dir',dest='ccoef_dir')
	parser.add_argument('--data_dir',dest='data_dir')
	args = parser.parse_args()
	
	mode = args.mode
	if mode=="how_to_run":
		print("python run.py --mode run"+
			" --voice_dir ../../cmu_yue_wdy_normal_build_new"+
			" --train_syllable_map ../decision_tree/train_dev_data_vector/train_data/syllable_map"+
			" --dev_syllable_map ../decision_tree/train_dev_data_vector/dev_data/syllable_map"+
			" --vector_feat_desc ../decision_tree/feature_desc_dir/feature_desc_vector"+
			" --val_feat_desc ../decision_tree/feature_desc_dir/feature_desc_val"+
			" --decompose_desc ./decompose_desc"+
			" --operation_file ./operation.desc"+
			" --true_f0_dir ../DCT/f0_value"+
			" --ccoef_dir ../../cmu_yue_wdy_addf0/ccoefs"+
			" --train_data ../decision_tree/train_dev_data_vector/train_data/dct_0"+
			" --train_label ../decision_tree/train_dev_data_vector/train_data_f0_vector"+
			" --dev_data ../decision_tree/train_dev_data_vector/dev_data/dct_0"+
			" --dev_label ../decision_tree/train_dev_data_vector/dev_data_f0_vector"+
			" --out_dir ./test_dir")
		print("python run.py --mode run"+
			" --voice_dir"+
			" --train_syllable_map"+
			" --dev_syllable_map"+
			" --vector_feat_desc"+
			" --val_feat_desc"+
			" --decompose_desc"+
			" --operation_file"+
			" --true_f0_dir"+
			" --ccoef_dir"+
			" --train_data"+
			" --train_label"+
			" --dev_data"+
			" --dev_label"+
			" --data_dir"+
			" --out_dir")

		print("python run.py --mode simple_run"+
			" --voice_dir"+
			" --decompose_desc"+
			" --operation_file"+
			" --data_dir"+
			" --out_dir")
	if mode=="run" or mode=="simple_run":
		if mode=="run":
			voice_dir = args.voice_dir
			out_dir = args.out_dir
			op_file = args.op_file
			train_data = args.train_data
			train_label = args.train_label
			dev_data = args.dev_data
			dev_label = args.dev_label
			train_syllable_map = args.train_syllable_map
			dev_syllable_map = args.dev_syllable_map
			true_f0_dir = args.true_f0_dir
			ccoef_dir = args.ccoef_dir
			vector_feat_desc = args.vector_feat_desc
			val_feat_desc = args.val_feat_desc
			decompose_desc = args.decompose_desc
		elif mode=="simple_run":
			voice_dir = args.voice_dir
			out_dir = args.out_dir
			op_file = args.op_file
			data_dir = args.data_dir
			decompose_desc = args.decompose_desc
			train_data = data_dir+"/train_test_data/train_data/train_feat"
			train_label = data_dir+"/train_test_data/train_data/train_f0"
			dev_data = data_dir+"/train_test_data/test_data/test_feat"
			dev_label = data_dir+"/train_test_data/test_data/test_f0"
			train_syllable_map = data_dir+"/train_test_data/train_data/train_syllable_map"
			dev_syllable_map = data_dir+"/train_test_data/test_data/test_syllable_map"
			true_f0_dir = data_dir+"/f0_value"
			ccoef_dir = voice_dir+"/ccoefs"
			vector_feat_desc = data_dir+"/new_feature_desc_vector"
			val_feat_desc = data_dir+"/new_feature_desc_val"

		os.system("mkdir "+out_dir)
		os.system("cp "+op_file+" "+out_dir)

		train_norm_row_record = None
		train_norm_global_record = None
		train_norm_col_record = None

		dev_norm_row_record = None
		dev_norm_global_record = None
		dev_norm_col_record = None


		op = read_operation(op_file)

		for key,val in op.items():
			print(key+" "+str(val))

		######################################################################
		if op["small_data"] == 1:
			print(">>>>>>>>>> create small dataset <<<<<<<<<<")
			train_data,train_label,train_syllable_map = get_small_data([train_data,train_label,train_syllable_map],0.01,out_dir)
		if op["predict"]=="phrase":
			train_data = data_dir+"/phrase_dir/phrase_f0_res_train_test/train_data/train_feat"
			train_label = data_dir+"/phrase_dir/phrase_f0_res_train_test/train_data/train_f0"
			dev_data = data_dir+"/phrase_dir/phrase_f0_res_train_test/test_data/test_feat"
			dev_label = data_dir+"/phrase_dir/phrase_f0_res_train_test/test_data/test_f0"
		######################################################################

		train_mean_std = row_mean_std(train_label,out_dir)
		dev_mean_std = row_mean_std(dev_label,out_dir)

		train_out = train_label
		dev_out = dev_label


		######################################################################
		if op["log"] == 1:
			print("")
			print(">>>>>>>>>> calculate logarithm for the data <<<<<<<<<<")
			train_out = log_row(train_out,out_dir)
			dev_out = log_row(dev_out,out_dir)
		######################################################################

		######################################################################
		pad_num = 10
		if op["padding"] == 1:
			print(">>>>>>>>>> padding <<<<<<<<<<")
			train_out,pad_num = padding(train_out,out_dir)
			dev_out,pad_num = padding(dev_out,out_dir)
		######################################################################

		######################################################################
		if op["norm"] == "row":
			print("")
			print(">>>>>>>>>> normalize row <<<<<<<<<<")
			train_out,train_norm_row_record = norm_row(train_out,out_dir)
			dev_out,dev_norm_row_record = norm_row(dev_out,out_dir)
		elif op["norm"] == "global":
			print("")
			print(">>>>>>>>>> normalize global <<<<<<<<<<")
			train_out,train_norm_global_record = norm_global(train_out,out_dir)
			dev_out,dev_norm_global_record = norm_global(dev_out,out_dir)
		elif op["norm"] == "none":
			# print(">>>>>>>>>> no row normalization <<<<<<<<<<")
			pass
		######################################################################


		######################################################################
		if op["dct"] >= 0:
			print("")
			print(">>>>>>>>>> dct row <<<<<<<<<<")
			train_out = dct_row(train_out,out_dir,int(op["dct"]))
			dev_out = dct_row(dev_out,out_dir,int(op["dct"]))
		elif op["dct"] == -1:
			# print(">>>>>>>>>> no dct <<<<<<<<<<")
			pass
		######################################################################


		######################################################################
		delta_num = 0
		delta_rate = op["delta_rate"]
		if op["delta"]=="mean":
			print("")
			print(">>>>>>>>>> calculate mean delta feature <<<<<<<<<<")
			train_out,delta_num = get_mean_delta(train_out,train_syllable_map,out_dir,delta_rate)
			dev_out,delta_num = get_mean_delta(dev_out,dev_syllable_map,out_dir,delta_rate)
		elif op["delta"]=="dct":
			print("")
			print(">>>>>>>>>> calculate dct delta feature <<<<<<<<<<")
			train_out,delta_num = get_dct_delta(train_out,train_syllable_map,out_dir,delta_rate)
			dev_out,delta_num = get_dct_delta(dev_out,dev_syllable_map,out_dir,delta_rate)
		elif op["delta"]=="f0":
			print("")
			print(">>>>>>>>>> calculate adjacent f0 delta feature <<<<<<<<<<")
			train_out,delta_num = get_f0_delta(train_out,train_syllable_map,out_dir,delta_rate)
			dev_out,delta_num = get_f0_delta(dev_out,dev_syllable_map,out_dir,delta_rate)
		elif op["delta"]=="self_delta":
			print("")
			print(">>>>>>>>>> calculate self delta feature <<<<<<<<<<")
			train_out,delta_num = get_self_delta(train_out,out_dir,delta_rate)
			dev_out,delta_num = get_self_delta(dev_out,out_dir,delta_rate)
		elif op["delta"]=="none":
			pass
		######################################################################


		######################################################################
		if op["norm_col"] == 1:
			print("")
			print(">>>>>>>>>> normalize column <<<<<<<<<<")
			col_num = np.loadtxt(train_out,delimiter=" ").shape[1]
			train_out,train_norm_col_record = norm_col(train_out,out_dir,col_num)
			dev_out,dev_norm_col_record = norm_col(dev_out,out_dir,col_num)
		elif op["norm_col"] == 0:
			# print(">>>>>>>>>> no column normalization <<<<<<<<<<")
			pass
		######################################################################


		######################################################################
		if op["predict"] == "vector":
			print("")
			print(">>>>>>>>>> predict f0 value vector <<<<<<<<<<")
			os.system("mkdir "+out_dir+"/predict_f0_vector")
			predict_f0 = predict_vector(train_data,train_out,dev_data,dev_out,out_dir+"/predict_f0_vector",vector_feat_desc,stop_size=30)

		elif op["predict"]=="multiple_vector":
			print("")
			print(">>>>>>>>>> decompose and predict multiple f0 value vector <<<<<<<<<<")
			os.system("mkdir "+out_dir+"/predict_f0_vector")
			predict_f0 = predict_multiple_vector(train_data,train_out,dev_data,dev_out,out_dir+"/predict_f0_vector",decompose_desc,vector_feat_desc,val_feat_desc)
		
		elif op["predict"] == "each_val":
			print("")
			print(">>>>>>>>>> predict f0 value individually <<<<<<<<<<")
			os.system("mkdir "+out_dir+"/predict_f0_val")
			predict_f0 = predict_each_f0_value(train_data,train_out,dev_data,dev_out,out_dir+"/predict_f0_val",val_feat_desc)
		
		elif op["predict"] == "tone_specific_vector":
			print("")
			print(">>>>>>>>>> predict tone specific f0 value <<<<<<<<<<")
			os.system("mkdir "+out_dir+"/predict_f0_vector")
			predict_f0 = predict_tone_specific(train_data,train_out,train_syllable_map,dev_data,dev_out,
				dev_syllable_map,out_dir+"/predict_f0_vector",vector_feat_desc,
				vector_prediction=True,tone_num = int(op["tone_num"]),stop_size=20)
		
		elif op["predict"] == "tone_specific_each_val":
			print("")
			print(">>>>>>>>>> predict tone specific f0 value <<<<<<<<<<")
			os.system("mkdir "+out_dir+"/predict_f0_val")
			predict_f0 = predict_tone_specific(train_data,train_out,train_syllable_map,dev_data,dev_out,\
				dev_syllable_map,out_dir+"/predict_f0_val",val_feat_desc,vector_prediction=False)
		
		elif op["predict"]=="phrase":
			print("")
			print(">>>>>>>>>> predict phrase dct coefficient <<<<<<<<<<")
			os.system("mkdir "+out_dir+"/phrase_prediction")
			tmp_train_data = data_dir+"/phrase_dir/phrase_f0_train_test/train_data/train_feat"
			tmp_train_label = data_dir+"/phrase_dir/phrase_f0_train_test/train_data/train_f0"
			tmp_test_data = data_dir+"/phrase_dir/phrase_f0_train_test/test_data/test_feat"
			tmp_test_label = data_dir+"/phrase_dir/phrase_f0_train_test/test_data/test_f0"
			tmp_vector_desc = data_dir+"/phrase_dir/feature_desc_vector"
			tmp_test_map = data_dir+"/phrase_dir/phrase_f0_train_test/test_data/test_syllable_map"
			tmp_phrase_syl = data_dir+"/phrase_dir/phrase_syllable"
			phrase_predict = predict_vector(tmp_train_data,tmp_train_label,tmp_test_data,tmp_test_label,out_dir+"/phrase_prediction",tmp_vector_desc,stop_size=10)
			phrase_dct_dir = put_predict_f0_in_file(phrase_predict,tmp_test_map,out_dir+"/phrase_prediction",first_syllable_flag=0)
			idct_phrase_dir = idct_phrase(tmp_phrase_syl,phrase_dct_dir,out_dir+"/phrase_prediction")
			predict_phrase_level_f0 = out_dir+"/phrase_prediction/predict_phrase_level_f0"
			map_f0(idct_phrase_dir,dev_syllable_map,predict_phrase_level_f0)

			print("")
			print(">>>>>>>>>> predict phrase residual f0 value <<<<<<<<<<")
			os.system("mkdir "+out_dir+"/phrase_res_prediction")
			predict_f0 = predict_vector(train_data,train_out,dev_data,dev_out,out_dir+"/phrase_res_prediction",vector_feat_desc,stop_size=30)
			# exit()
		
		elif op["predict"]=="rf_vector":
			print(">>>>>>>>>> random forest prediction <<<<<<<<<<")
			os.system("mkdir "+out_dir+"/rf_dump")
			rf = RandomForest(20,0.7,0.7,train_out,train_data,dev_out,dev_data,vector_feat_desc,out_dir+"/rf_dump",stop_size=30)
			predict_f0 = rf.train()

		elif op["predict"] == "rf_tone_specific_vector":
			print("")
			print(">>>>>>>>>> predict tone specific f0 value using random forest<<<<<<<<<<")
			os.system("mkdir "+out_dir+"/predict_f0_vector")
			predict_f0 = predict_tone_specific(train_data,train_out,train_syllable_map,dev_data,dev_out,\
				dev_syllable_map,out_dir+"/predict_f0_vector",vector_feat_desc,
				vector_prediction=True,tone_num = int(op["tone_num"]),rf_param=[20,0.7,0.7],stop_size=20)

		elif op["predict"] == "none":
			print("")
			print(">>>>>>>>>> true f0 value <<<<<<<<<<")
			predict_f0 = dev_out
		######################################################################


		######################################################################
		predict_mean_std = None
		if op["predict_mean_std"] == "vector":

			# if op["delta"]!="none":
			# 	print(">>>>>>>>>> calculate delta for mean and standard deviation <<<<<<<<<<")
			# 	train_mean_std,msdelta_num = get_mean_std_delta(train_mean_std,train_syllable_map,out_dir,1)
			# 	dev_mean_std,msdelta_num = get_mean_std_delta(dev_mean_std,dev_syllable_map,out_dir,1)

			print("")
			print(">>>>>>>>>> predict f0 mean and standard deviation <<<<<<<<<<")
			os.system("mkdir "+out_dir+"/predict_mean_std")
			predict_mean_std = predict_vector(train_data,train_mean_std,dev_data,dev_mean_std,out_dir+"/predict_mean_std",vector_feat_desc)

			# if op["delta"]!="none":
			# 	np.savetxt(predict_mean_std+"_msdeldel",np.loadtxt(predict_mean_std,delimiter=" ")[:,0:-msdelta_num],delimiter=" ")
			# 	predict_mean_std = predict_mean_std+"_msdeldel"

		elif op["predict_mean_std"] == "tone_specific_vector":

			print("")
			print(">>>>>>>>>> predict f0 mean and standard deviation(tone dependent)<<<<<<<<<<")
			os.system("mkdir "+out_dir+"/predict_mean_std")
			predict_mean_std = predict_tone_specific(train_data,train_mean_std,train_syllable_map,dev_data,dev_mean_std,
				dev_syllable_map,out_dir+"/predict_mean_std",vector_feat_desc,
				vector_prediction=True,tone_num = int(op["tone_num"]),stop_size=20)

		elif op["predict_mean_std"] == "rf_vector":
			print("")
			print(">>>>>>>>>> predict f0 mean and standard deviation <<<<<<<<<<")
			os.system("mkdir "+out_dir+"/predict_mean_std")
			os.system("mkdir "+out_dir+"/predict_mean_std/rf_dump_mean_std")
			rf = RandomForest(20,1,0.7,train_mean_std,train_data,dev_mean_std,dev_data,
				vector_feat_desc,out_dir+"/predict_mean_std/rf_dump_mean_std",stop_size=30)
			predict_mean_std = rf.train()

		elif op["predict_mean_std"] == "rf_tone_specific_vector":
			print("")
			print(">>>>>>>>>> predict f0 mean and standard deviation(tone dependent random forest)<<<<<<<<<<")
			os.system("mkdir "+out_dir+"/predict_mean_std")
			os.system("mkdir "+out_dir+"/predict_mean_std/rf_dump_mean_std")
			predict_mean_std = predict_tone_specific(train_data,train_mean_std,train_syllable_map,dev_data,dev_mean_std,
				dev_syllable_map,out_dir+"/predict_mean_std/rf_dump_mean_std",vector_feat_desc,
				vector_prediction=True,tone_num = int(op["tone_num"]),rf_param=[20,1,0.7],stop_size=20)
			
		elif op["predict_mean_std"] == "none":
			pass

		if op["predict_mean_std"]!="none":
			print(">>>>>>>>>> shape syllable level statistics <<<<<<<<<<")
			vector_syl_stat(predict_f0,dev_out)
			print(">>>>>>>>>> shape utterance level statistics <<<<<<<<<<")
			vector_utt_stat(predict_f0,dev_out,dev_syllable_map)

			print(">>>>>>>>>> mean std syllable level statistics <<<<<<<<<<")
			mean_std_syl_stat(predict_mean_std,dev_mean_std)
			print(">>>>>>>>>> mean std utterance level statistics <<<<<<<<<<")
			mean_std_utt_stat(predict_mean_std,dev_mean_std,dev_syllable_map)
		######################################################################


		######################################################################
		if op["unnorm_col"] == 1:
			print("")
			print(">>>>>>>>>> unnormalize column <<<<<<<<<<")
			predict_f0 = unnorm_col(predict_f0,train_norm_col_record,out_dir)
		elif op["unnorm_col"] == 0:
			# print(">>>>>>>>>> no column unnormalization <<<<<<<<<<")
			pass
		######################################################################

		######################################################################
		if op["delta"] != "none":
			print("")
			print(">>>>>>>>>> delete delta <<<<<<<<<<")
			np.savetxt(predict_f0+"_deldel",np.loadtxt(predict_f0)[:,0:-delta_num])
			predict_f0 = predict_f0+"_deldel"
		else:
			pass
		######################################################################


		######################################################################
		if op["idct"] >= 0:
			print("")
			print(">>>>>>>>>> idct <<<<<<<<<<")
			predict_f0 = idct_row(predict_f0,out_dir,int(op["idct"]))
		elif op["idct"] == -1:
			# print(">>>>>>>>>> no idct <<<<<<<<<<")
			pass
		######################################################################


		######################################################################
		if "true" in op["unnorm"] or "predict" in op["unnorm"]:
			print("")
			print(">>>>>>>>>> unnormalize row <<<<<<<<<<")
			mean = op["unnorm"].split("_")[0]
			std = op["unnorm"].split("_")[1]
			predict_f0 = unnorm_mean_std(predict_f0,mean,std,dev_mean_std,predict_mean_std,out_dir)
		elif op["unnorm"]=="global":
			print("")
			print(">>>>>>>>>> unnormalize global <<<<<<<<<<")
			predict_f0 = unnorm_global(predict_f0,train_norm_global_record,out_dir)
		elif op["unnorm"]=="none":
			# print(">>>>>>>>>> no row unnormalization <<<<<<<<<<")
			pass
		######################################################################


		######################################################################
		if op["only_use_mean"] == 1:
			print("")
			print(">>>>>>>>>> only use the mean of f0 <<<<<<<<<<")
			predict_f0 = only_use_mean(predict_f0,out_dir)
		elif op["only_use_mean"] == 0:
			pass
		######################################################################

		######################################################################
		if op["padding"]==1:
			predict_f0 = unpadding(predict_f0,pad_num,out_dir)
		######################################################################

		######################################################################
		if op["log"] == 1:
			print("")
			print(">>>>>>>>>> calculate exponential for the data <<<<<<<<<<")
			predict_f0 = exp_row(predict_f0,out_dir)
		######################################################################

		######################################################################
		if op["predict"]=="phrase":
			np.savetxt(out_dir+"/predict_f0",np.loadtxt(predict_f0,delimiter=" ")+np.loadtxt(predict_phrase_level_f0,delimiter=" "),delimiter=" ",fmt="%.5f")
			predict_f0 = out_dir+"/predict_f0"
			train_data = data_dir+"/train_test_data/train_data/train_feat"
			train_label = data_dir+"/train_test_data/train_data/train_f0"
			dev_data = data_dir+"/train_test_data/test_data/test_feat"
			dev_label = data_dir+"/train_test_data/test_data/test_f0"
		######################################################################



		print("final prediction for test data: "+predict_f0)



		######################################################################
		if op["test_stat"] == 1:
			print("")
			print(">>>>>>>>>> syllable level test statistics <<<<<<<<<<")
			vector_syl_stat(predict_f0,dev_label)
			# test_statistics(predict_f0,dev_label,out_dir)
			# os.system("cat "+out_dir+"/"+"test_statistics")
			print(">>>>>>>>>> utterance level test statistics no duration <<<<<<<<<<")
			vector_utt_stat(predict_f0,dev_label,dev_syllable_map)
		elif op["test_stat"] == 0:
			# print(">>>>>>>>>> no test statistics <<<<<<<<<<")
			pass
		######################################################################


		######################################################################
		print("")
		print(">>>>>>>>>> put f0 back in file <<<<<<<<<<")
		predict_f0 = put_predict_f0_in_file(predict_f0,dev_syllable_map,out_dir)
		######################################################################


		######################################################################
		print("")
		if op["predict_duration"] == 0:
			print(">>>>>>>>>> generate f0 timeline <<<<<<<<<<")
			predict_f0 = generate_f0_time_line(predict_f0,out_dir,true_f0_dir)
		elif op["predict_duration"] == 1:
			print(">>>>>>>>>> apply f0 in festival prediction <<<<<<<<<<")
			os.system("python ./apply_f0.py"+
				" --mode run"+
				" --voice_dir "+voice_dir+
				" --out_dir "+out_dir+"/wdy_tmp"+
				" --modified_clustergen_scm ./experiment/clustergen.scm"+
				" --test_txt "+voice_dir+"/etc/txt.done.data.test")
			ccoef_dir = out_dir+"/wdy_tmp/ccoefs"
			f0_tag_dir = out_dir+"/wdy_tmp/f0_value"
			predict_f0 = generate_f0_time_line(predict_f0,out_dir,f0_tag_dir)
		######################################################################

		######################################################################
		print("")
		print(">>>>>>>>>> predict statistics for f0 timeline with duration <<<<<<<<<<")
		print(">>>>>>>>>> put subsample f0 in files and generate timeline")
		os.system("mkdir "+out_dir+"/subsample_in_file")
		sub_f0_dir = put_predict_f0_in_file(dev_label,dev_syllable_map,out_dir+"/subsample_in_file")
		sub_f0_dir = generate_f0_time_line(sub_f0_dir,out_dir+"/subsample_in_file",true_f0_dir)
		print(">>>>>>>>>> calculate statistics...")
		predict_f0 = timeline_statistics(predict_f0,sub_f0_dir)
		######################################################################




		######################################################################
		if op["slide"] == 1:
			print("")
			print(">>>>>>>>>> slide f0 contour for concatenation <<<<<<<<<<")
			predict_f0 = slide(predict_f0,out_dir)
		elif op["slide"] == 0:
			pass
		######################################################################


		######################################################################
		if op["generate_wav"] == 1:
			print("")
			print(">>>>>>>>>> generate waveform files from f0 <<<<<<<<<<")
			predict_f0 = generate_wav(predict_f0,ccoef_dir,out_dir)
		elif op["generate_wav"] == 0:
			pass
		######################################################################

