import os
import numpy as np
from scipy.fftpack import idct, dct
from rf import RandomForest
def predict_multiple_vector(train_data,train_label,dev_data,dev_label,out_dir,decompose_desc,vector_feat_desc,val_feat_desc,log=""):
	dct_flag = False
	individual_flag = False

	dev_name = dev_data.split("/")[-1]
	train_dir = out_dir+"/split_train_f0"
	os.system("mkdir "+train_dir)
	dev_dir = out_dir+"/split_dev_f0"
	os.system("mkdir "+dev_dir)

	train_label = np.loadtxt(train_label,delimiter=" ",dtype=np.str)
	train_data = np.loadtxt(train_data,delimiter=" ",dtype=np.str)
	dev_label = np.loadtxt(dev_label,delimiter=" ",dtype=np.str)
	dev_data = np.loadtxt(dev_data,delimiter=" ",dtype=np.str)

	decompose_list = []
	with open(decompose_desc) as f:
		for line in f:
			decompose_list.append(np.array([]+[int(val) for val in line.strip().split(",")]))
	out_file_list = []
	print("##########################################")
	for i in range(len(decompose_list)):
		tmp_train_label = train_dir+"/vector_label_"+str(i)
		tmp_train_data = train_dir+"/vector_data_"+str(i)
		tmp_dev_label = dev_dir+"/vector_label_"+str(i)
		tmp_dev_data = dev_dir+"/vector_data_"+str(i)
		tmp_predict_dir = out_dir+"/tmp_predict"+str(i)
		os.system("mkdir "+tmp_predict_dir)

		vec = decompose_list[i]
		if len(vec)>1:
			if dct_flag:
				np.savetxt(tmp_train_label,dct(train_label[:,vec],axis=1),delimiter=" ",fmt="%s")
			else:
				np.savetxt(tmp_train_label,train_label[:,vec],delimiter=" ",fmt="%s")
			np.savetxt(tmp_train_data,train_data,delimiter=" ",fmt="%s")
			if dct_flag:
				np.savetxt(tmp_dev_label,dct(dev_label[:,vec],axis=1),delimiter=" ",fmt="%s")
			else:
				np.savetxt(tmp_dev_label,dev_label[:,vec],delimiter=" ",fmt="%s")
			np.savetxt(tmp_dev_data,dev_data,delimiter=" ",fmt="%s")
			tmp_out_file = None
			if individual_flag:
				tmp_out_file = predict_each_f0_value(tmp_train_data,tmp_train_label,tmp_dev_data,tmp_dev_label,tmp_predict_dir,val_feat_desc,log)
			else:
				tmp_out_file = predict_vector(tmp_train_data,tmp_train_label,tmp_dev_data,tmp_dev_label,tmp_predict_dir,vector_feat_desc,log)
			if dct_flag:
				tmp_result = np.loadtxt(tmp_out_file,delimiter=" ")
				tmp_result = idct(tmp_result,axis=1)/(2*tmp_result.shape[1])
				np.savetxt(tmp_out_file,tmp_result,delimiter=" ")
			out_file_list.append(tmp_out_file)
		elif len(vec)==1:
			np.savetxt(tmp_train_data,np.hstack((train_label[:,vec].reshape((-1,1)),train_data[:,1:])),fmt="%s",delimiter=" ")
			np.savetxt(tmp_dev_data,np.hstack((dev_label[:,vec].reshape((-1,1)),dev_data[:,1:])),fmt="%s",delimiter=" ")
			tmp_out_file = predict_one_value(tmp_train_data,tmp_dev_data,tmp_predict_dir,log)
			out_file_list.append(tmp_out_file)
		print("##########################################")
	# print(out_file_list)
	predict_result = np.zeros(dev_label.shape)
	for i in range(len(out_file_list)):
		predict_result[:,decompose_list[i]] = np.loadtxt(out_file_list[i],delimiter=" ").reshape((-1,len(decompose_list[i])))
	out_file = out_dir+"/predict_mv"
	np.savetxt(out_file,predict_result,delimiter=" ")
	return out_file


def predict_one_value(train_file,dev_file,out_dir,feat_desc="../decision_tree/feature_desc_dir/feature_desc_val",log=""):
	dev_name = dev_file.split("/")[-1]
	os.system("python ../decision_tree/wagon/run.py"+
		" --mode train_predict_individual_val"+
		" --train_file "+train_file+
		" --test_file "+dev_file+
		" --desc_file "+feat_desc+
		" --out_dir "+out_dir+
		log)
	out_file = out_dir+"/"+dev_name
	return out_file

def predict_vector(train_data,train_label,dev_data,dev_label,out_dir,feat_desc="../decision_tree/feature_desc_dir/feature_desc_vector",log="",stop_size=30):
	dev_name = dev_data.split("/")[-1]
	os.system("python ../decision_tree/wagon/run.py"+
		" --mode train_predict_vector"+
		" --desc_file "+feat_desc+
		" --train_file "+train_data+
		" --train_label "+train_label+
		" --test_file "+dev_data+
		" --test_label "+dev_label+
		" --stop_size "+str(stop_size)+
		" --out_dir "+out_dir+
		log)
	out_file = out_dir+"/"+dev_name+"_val"
	return out_file

def predict_each_f0_value(train_data,train_label,dev_data,dev_label,out_dir,feat_desc="../decision_tree/feature_desc_dir/feature_desc_val",log=""):
	train_dir = out_dir+"/split_train_f0"
	os.system("mkdir "+train_dir)
	dev_dir = out_dir+"/split_dev_f0"
	os.system("mkdir "+dev_dir)

	train_label = np.loadtxt(train_label,delimiter=" ",dtype=np.str)
	train_data = np.loadtxt(train_data,delimiter=" ",dtype=np.str)
	dev_label = np.loadtxt(dev_label,delimiter=" ",dtype=np.str)
	dev_data = np.loadtxt(dev_data,delimiter=" ",dtype=np.str)

	for i in range(train_label.shape[1]):
		np.savetxt(train_dir+"/val_"+str(i),
			np.hstack((train_label[:,i].reshape((-1,1)),train_data[:,1:])),fmt="%s")
		np.savetxt(dev_dir+"/val_"+str(i),
			np.hstack((dev_label[:,i].reshape((-1,1)),dev_data[:,1:])),fmt="%s")

	os.system("python ../decision_tree/wagon/run.py"+
		" --mode train_predict_dir"+
		" --train_dir "+train_dir+
		" --test_dir "+dev_dir+
		" --desc_file "+feat_desc+
		" --out_dir "+out_dir+"/predict_f0_val"+
		log)
	file_list = os.listdir(out_dir+"/predict_f0_val")
	f0_vec = np.hstack(tuple([np.loadtxt(out_dir+"/predict_f0_val/"+file,delimiter=" ").reshape((-1,1)) 
		for file in file_list if "val" in file]))
	out_file = out_dir+"/predict_f0_vector"
	np.savetxt(out_file,f0_vec,delimiter=" ")
	print("predicted f0 vector after combination is saved to "+out_file)
	os.system("rm -r "+train_dir)
	os.system("rm -r "+dev_dir)
	return out_file

def predict_tone_specific(train_data,train_label,train_map_file,dev_data,dev_label,dev_map_file,
	out_dir,feat_desc,vector_prediction=True,tone_num = 5,rf_param=None,stop_size=20):
	#rf_param=[tree_num,label_ratio,feat_ratio]
	os.system("mkdir "+out_dir+"/tone_dev")
	os.system("mkdir "+out_dir+"/tone_train")
	os.system("mkdir "+out_dir+"/tone_predict")
	os.system("rm "+out_dir+"/log_file")
	print("split data according to tone")
	os.system("python ../decision_tree/wagon/data_processing.py"+
		" --mode split_tone"+
		" --feat_file "+train_data+
		" --f0_file "+train_label+
		" --map_file "+train_map_file+
		" --record_file "+out_dir+"/tone_train/record_file"+
		" --out_dir "+out_dir+"/tone_train")
	os.system("python ../decision_tree/wagon/data_processing.py"+
		" --mode split_tone"+
		" --feat_file "+dev_data+
		" --f0_file "+dev_label+
		" --map_file "+dev_map_file+
		" --record_file "+out_dir+"/tone_dev/record_file"+
		" --out_dir "+out_dir+"/tone_dev")
	log_file = out_dir+"/prediction_log"
	os.system("rm "+log_file)
	for tone in range(1,1+tone_num):
		print("")
		print(">>>>>>>>>> predicting on tone "+str(tone))
		tone_train_data = out_dir+"/tone_train/feat_tone_"+str(tone)
		tone_train_label = out_dir+"/tone_train/f0_tone_"+str(tone)
		tone_dev_data = out_dir+"/tone_dev/feat_tone_"+str(tone)
		tone_dev_label = out_dir+"/tone_dev/f0_tone_"+str(tone)
		tone_predict = None
		if rf_param is not None:
			os.system("mkdir "+out_dir+"/tone_predict/rf_tone"+str(tone))
			tone_rf = RandomForest(rf_param[0],rf_param[1],rf_param[2],
				tone_train_label,tone_train_data,tone_dev_label,tone_dev_data,
				feat_desc,out_dir+"/tone_predict/rf_tone"+str(tone),stop_size=stop_size)
			tone_predict = tone_rf.train()
		elif vector_prediction:
			tone_predict = predict_vector(tone_train_data,tone_train_label,tone_dev_data,tone_dev_label,
				out_dir+"/tone_predict",feat_desc,log=" >> "+log_file,stop_size=stop_size)
		else:
			tone_predict = predict_each_f0_value(tone_train_data,tone_train_label,tone_dev_data,tone_dev_label,
				out_dir+"/tone_predict",feat_desc,log=" >> "+log_file)
		os.system("rm "+out_dir+"/tone_dev/f0_tone_"+str(tone))
		os.system("mv "+tone_predict+" "+out_dir+"/tone_dev/f0_tone_"+str(tone))
	os.system("python ../decision_tree/wagon/data_processing.py"+
			" --mode combine_tone"+
			" --in_dir "+out_dir+"/tone_dev/"+
			" --record_file "+out_dir+"/tone_dev/record_file"+
			" --out_dir "+out_dir)
	os.system("rm -r "+out_dir+"/tone_dev")
	os.system("rm -r "+out_dir+"/tone_train")
	# os.system("rm -r "+out_dir+"/tone_predict")
	out_file = out_dir+"/combine_f0"
	print("log file: "+log_file)
	return out_file