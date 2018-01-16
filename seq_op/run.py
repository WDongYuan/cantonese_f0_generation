import os
import argparse
import numpy as np
from predict_func import *

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
			if line[1]=="0" or line[1]=="1":
				op_val = True if line[1]=="1" else False
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

def dct_row(in_file,out_dir):
	file_name = in_file.split("/")[-1]
	out_file = out_dir+"/"+file_name+"_dct"
	os.system("python ../decision_tree/wagon/data_processing.py"+
		" --mode dct_row"+
		" --in_file "+in_file+
		" --out_file "+out_file)
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

def idct_row(in_file,out_dir):
	file_name = in_file.split("/")[-1]
	out_file = out_dir+"/"+file_name+"_idct"
	os.system("python ../decision_tree/wagon/data_processing.py"+
		" --mode idct_row"+
		" --in_file "+in_file+
		" --out_file "+out_file)
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

def put_predict_f0_in_file(in_file,syllable_map,out_dir):
	out_file = out_dir+"/predict_f0_in_file"
	os.system("mkdir "+out_file)
	os.system("python ../decision_tree/wagon/run.py"+
		" --mode put_back_f0_in_file"+
		" --f0_file_map "+syllable_map+
		" --f0_val "+in_file+
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
			" --out_dir")
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
		os.system("mkdir "+out_dir)
		os.system("cp "+op_file+" "+out_dir)

		train_norm_row_record = None
		train_norm_global_record = None
		train_norm_col_record = None

		dev_norm_row_record = None
		dev_norm_global_record = None
		dev_norm_col_record = None


		op = read_operation(op_file)

		train_mean_std = row_mean_std(train_label,out_dir)
		dev_mean_std = row_mean_std(dev_label,out_dir)

		train_out = train_label
		dev_out = dev_label

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
		if op["dct"] == 1:
			print("")
			print(">>>>>>>>>> dct row <<<<<<<<<<")
			train_out = dct_row(train_out,out_dir)
			dev_out = dct_row(dev_out,out_dir)
		elif op["dct"] == 0:
			# print(">>>>>>>>>> no dct <<<<<<<<<<")
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
		vector_feat_desc = args.vector_feat_desc
		val_feat_desc = args.val_feat_desc
		if op["predict"] == "vector":
			print("")
			print(">>>>>>>>>> predict f0 value vector <<<<<<<<<<")
			os.system("mkdir "+out_dir+"/predict_f0_vector")
			predict_f0 = predict_vector(train_data,train_out,dev_data,dev_out,out_dir+"/predict_f0_vector",vector_feat_desc)
		elif op["predict"]=="multiple_vector":
			print("")
			print(">>>>>>>>>> decompose and predict multiple f0 value vector <<<<<<<<<<")
			os.system("mkdir "+out_dir+"/predict_f0_vector")
			decompose_desc = args.decompose_desc
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
			predict_f0 = predict_tone_specific(train_data,train_out,train_syllable_map,dev_data,dev_out,\
				dev_syllable_map,out_dir+"/predict_f0_vector",vector_feat_desc,vector_prediction=True)
		elif op["predict"] == "tone_specific_each_val":
			print("")
			print(">>>>>>>>>> predict tone specific f0 value <<<<<<<<<<")
			os.system("mkdir "+out_dir+"/predict_f0_val")
			predict_f0 = predict_tone_specific(train_data,train_out,train_syllable_map,dev_data,dev_out,\
				dev_syllable_map,out_dir+"/predict_f0_val",val_feat_desc,vector_prediction=False)
		elif op["predict"] == "none":
			print("")
			print(">>>>>>>>>> true f0 value <<<<<<<<<<")
			predict_f0 = dev_out
		######################################################################


		######################################################################
		predict_mean_std = None
		if op["predict_mean_std"] == 1:
			print("")
			print(">>>>>>>>>> predict f0 mean and standard deviation <<<<<<<<<<")
			os.system("mkdir "+out_dir+"/predict_mean_std")
			predict_mean_std = predict_vector(train_data,train_mean_std,dev_data,dev_mean_std,out_dir+"/predict_mean_std",vector_feat_desc)
		elif op["predict_mean_std"] == 0:
			pass
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
		if op["idct"] == 1:
			print("")
			print(">>>>>>>>>> idct <<<<<<<<<<")
			predict_f0 = idct_row(predict_f0,out_dir)
		elif op["idct"] == 0:
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
		if op["test_stat"] == 1:
			print("")
			print(">>>>>>>>>> test statistics <<<<<<<<<<")
			test_statistics(predict_f0,dev_label,out_dir)
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
			os.system("python ./apply_f0.py "+voice_dir+" "+out_dir+"/wdy_tmp")
			ccoef_dir = out_dir+"/wdy_tmp/ccoefs"
			f0_tag_dir = out_dir+"/wdy_tmp/f0_value"
			predict_f0 = generate_f0_time_line(predict_f0,out_dir,f0_tag_dir)
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

