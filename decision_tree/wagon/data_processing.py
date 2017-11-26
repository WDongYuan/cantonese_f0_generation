import numpy as np
import argparse
import os
from scipy.fftpack import idct, dct
def Normalization(arr):
	norm_arr = np.array(arr)
	std_dev = np.std(norm_arr)
	mean = np.mean(norm_arr)
	norm_arr = (norm_arr-mean)/std_dev
	return norm_arr,mean,std_dev

def NormalizeFileCol(infile,outfile,record_file,col):
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

	##Write the file after normalization
	with open(outfile,"w+") as f:
		for sample in content:
			f.write(" ".join(sample)+"\n")

	##Write the mean and std_dev in file
	with open(record_file,"w+") as f:
		for tmp_record in record:
			f.write("col_"+str(tmp_record[0])+" "+str(tmp_record[1])+" "+str(tmp_record[2])+"\n")
	return
def UnNormalizeFileCol(infile,outfile,record_file):
	norm_col = []
	with open(record_file) as f:
		for line in f:
			val = line.strip().split(" ")
			norm_col.append([int(val[0].split("_")[1]),float(val[1]),float(val[2])])

	ori_val = []
	with open(infile) as f:
		for line in f:
			val = line.strip().split(" ")
			val = [float(val[i]) for i in range(len(val))]
			ori_val.append(val)

	unnorm_val = np.array(ori_val)
	for col,mean,std in norm_col:
		unnorm_val[:,col] = unnorm_val[:,col]*std+mean

	unnorm_val = list(unnorm_val)
	with open(outfile,"w+") as f:
		for row in unnorm_val:
			f.write(" ".join([str(val) for val in row])+"\n")
	return

def NormalizeFileRow(infile,outfile,record_file):
	data = np.loadtxt(infile,delimiter=" ")
	std = np.std(data,axis=1)+0.001
	mean = np.mean(data,axis=1)
	norm_data = (data-mean.reshape((-1,1)))/std.reshape((-1,1))
	np.savetxt(outfile,norm_data,delimiter=" ",fmt="%.10f")
	record = np.vstack((mean,std))
	np.savetxt(record_file,record,delimiter=" ")

def UnNormalizeFileRow(infile,outfile,record_file):
	data = np.loadtxt(infile,delimiter=" ")
	record = np.loadtxt(record_file,delimiter=" ")
	mean = record[0]
	std = record[1]
	# print(std.shape)
	unnorm_data = np.multiply(data,std.reshape((-1,1)))+mean.reshape((-1,1))
	np.savetxt(outfile,unnorm_data,delimiter=" ",fmt="%.10f")

def NormalizeFileGlobal(infile,outfile,record_file):
	data = np.loadtxt(infile,delimiter=" ")
	std = np.std(data)+0.001
	mean = np.mean(data)
	norm_data = (data-mean)/std
	np.savetxt(outfile,norm_data,delimiter=" ",fmt="%.10f")
	np.savetxt(record_file,np.array([mean,std]),delimiter=" ")

def UnNormalizeFileGlobal(infile,outfile,record_file):
	data = np.loadtxt(infile,delimiter=" ")
	record = np.loadtxt(record_file,delimiter=" ")
	mean = record[0]
	std = record[1]
	# print(std.shape)
	unnorm_data = data*std+mean
	np.savetxt(outfile,unnorm_data,delimiter=" ",fmt="%.10f")

	

def CombineColFile(file_list,out_file):
	arr = None
	for file_name in file_list:
		# print(file_name)
		col = np.reshape(np.loadtxt(file_name,delimiter=" "),(-1,1))
		if arr is None:
			arr = col
		else:
			arr = np.hstack((arr,col))
	np.savetxt(out_file,arr,delimiter=" ")
	return

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', dest='mode')
	parser.add_argument('--dir', dest='dir')
	parser.add_argument('--in_file', dest='in_file')
	parser.add_argument('--out_file', dest='out_file')
	parser.add_argument('--record_file', dest='record_file')
	parser.add_argument('--file1', dest='file1')
	parser.add_argument('--file2', dest='file2')
	parser.add_argument('--column', dest='column',type=int,default=0)
	parser.add_argument('--col_list', dest='col_list',nargs='+',type=int)
	args = parser.parse_args()

	mode = args.mode

	# if mode=="combine_column_file":
	# 	my_dir = args.dir
	# 	out_file = args.out_file
	# 	file_list = [my_dir+"/"+file_name for file_name in os.listdir(my_dir)]
	# 	CombineColFile(file_list,out_file)
	if mode=="how_to_run":
		print("python data_processing.py"+
			" --mode dct_row/idct_row/transpose_file"+
			" --in_file"+
			" --out_file")
		print("python data_processing.py"+
			" --mode normalize_row/unnormalize_row/unnormalize_col/normalize_global/unnormalize_global"+
			" --in_file"+
			" --out_file"+
			" --record_file")
		print("python data_processing.py"+
			" --mode normalize_col"+
			" --in_file"+
			" --out_file"+
			" --record_file"+
			" --col_list")
		print("python data_processing.py"+
			" --mode row_mean_std"+
			" --in_file"+
			" --record_file")

	elif mode=="combine_file_column":
		my_dir = args.dir
		out_file = args.out_file
		col = args.column
		file_list = [my_dir+"/"+file_name for file_name in os.listdir(my_dir)]
		file_token_list = [open(file_name) for file_name in file_list]
		line = [token.readline().strip() for token in file_token_list]
		data = []
		while line[0]!="":
			row = []
			for tmp_line in line:
				row.append(tmp_line.split(" ")[col])
			data.append(row)
			line = [token.readline().strip() for token in file_token_list]
		for token in file_token_list:
			token.close()
		
		with open(out_file,"w+") as f:
			for row in data:
				f.write(" ".join(row)+"\n")

	elif mode=="dct_row":
		in_file = args.in_file
		out_file = args.out_file
		data = np.loadtxt(in_file,delimiter=" ")
		data = dct(data,axis=1)
		np.savetxt(out_file,data,delimiter=" ",fmt="%.10f")

	elif mode=="idct_row":
		in_file = args.in_file
		out_file = args.out_file
		data = np.loadtxt(in_file,delimiter=" ")
		data = idct(data,axis=1)/(2*data.shape[1])
		np.savetxt(out_file,data,delimiter=" ",fmt="%.10f")

	elif mode=="file_statistic":
		file1 = args.file1
		file2 = args.file2
		out_file = args.out_file

		arr1 = np.loadtxt(file1,delimiter=" ")
		arr2 = np.loadtxt(file2,delimiter=" ")

		out_file = open(out_file,"w+")

		out_file.write("SAMPLE MEAN:"+"\n")
		out_file.write(str(np.mean(arr1,axis=0))+"\n")

		out_file.write("SAMPLE RMSE:"+"\n")
		out_file.write(str(np.sqrt(np.mean(np.square(arr1-arr2),axis=0)))+"\n")

		out_file.write("Column correlation: "+"\n")
		cor = np.zeros((arr1.shape[1],))
		for col in range(arr1.shape[1]):
			cor[col] = np.corrcoef(arr1[:,col],arr2[:,col])[0][1]
		out_file.write(str(cor)+"\n")

		# out_file.write("ROW correlation: "+"\n")
		# cor = np.zeros((arr1.shape[0],))
		# for row in range(arr1.shape[0]):
		# 	cor[row] = np.corrcoef(arr1[row,:],arr2[row,:])[0][1]
		# out_file.write(str(cor)+"\n")

		out_file.write("ABS ERROR:"+"\n")
		out_file.write(str(np.mean(np.abs(arr1-arr2),axis=0))+"\n")

	elif mode=="normalize_row":
		in_file = args.in_file
		out_file = args.out_file
		record_file = args.record_file
		NormalizeFileRow(in_file,out_file,record_file)
	elif mode=="unnormalize_row":
		in_file = args.in_file
		out_file = args.out_file
		record_file = args.record_file
		UnNormalizeFileRow(in_file,out_file,record_file)

	elif mode=="normalize_col":
		in_file = args.in_file
		out_file = args.out_file
		record_file = args.record_file
		col_list = args.col_list
		NormalizeFileCol(in_file,out_file,record_file,col_list)
	elif mode=="unnormalize_col":
		in_file = args.in_file
		out_file = args.out_file
		record_file = args.record_file
		UnNormalizeFileCol(in_file,out_file,record_file)

	elif mode=="normalize_global":
		in_file = args.in_file
		out_file = args.out_file
		record_file = args.record_file
		NormalizeFileGlobal(in_file,out_file,record_file)
	elif mode=="unnormalize_global":
		in_file = args.in_file
		out_file = args.out_file
		record_file = args.record_file
		UnNormalizeFileGlobal(in_file,out_file,record_file)
	elif mode=="transpose_file":
		in_file = args.in_file
		out_file = args.out_file
		m = np.loadtxt(in_file,delimiter=" ")
		np.savetxt(out_file,m.T,delimiter=" ")
	elif mode=="row_mean_std":
		in_file = args.in_file
		record_file = args.record_file
		data = np.loadtxt(in_file,delimiter=" ")
		mean = data.mean(axis=1)
		std = data.std(axis=1)
		np.savetxt(record_file,np.hstack((mean.reshape((-1,1)),std.reshape((-1,1)))),delimiter=" ",fmt="%.10f")
	else:
		print("Mode error!!")







'''
python data_processing.py --mode combine_file_column --column 0 --dir ../train_dev_data/dev_data --out_file dump_1/true_dct_val
python data_processing.py --mode file_statistic --file1 dump_1/true_f0_val --file2 dump_1/predict_f0_val --out_file ./dump_1/info
'''










