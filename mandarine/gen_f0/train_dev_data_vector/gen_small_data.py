from sets import Set
import sys
import os
if __name__=="__main__":
	if sys.argv[1]=="according_to_number":
		dev_num = 10000
		train_num = 61643
		with open("train_data_f0_vector") as tf, open("train_data_f0_vector_small","w+") as stf:
			stf.write("".join(tf.readlines()[0:train_num]))
		with open("./train_data/dct_0") as tf, open("./train_data/dct_0_small","w+") as stf:
			stf.write("".join(tf.readlines()[0:train_num]))
		with open("./train_data/syllable_map") as tf, open("./train_data/syllable_map_small","w+") as stf:
			stf.write("".join(tf.readlines()[0:train_num]))

		# with open("dev_data_f0_vector") as df, open("dev_data_f0_vector_small","w+") as sdf:
		# 	sdf.write("".join(df.readlines()[0:dev_num]))
		# with open("./dev_data/dct_0") as df, open("./dev_data/dct_0_small","w+") as sdf:
		# 	sdf.write("".join(df.readlines()[0:dev_num]))
		# with open("./dev_data/syllable_map") as df, open("./dev_data/syllable_map_small","w+") as sdf:
		# 	sdf.write("".join(df.readlines()[0:dev_num]))

		# myset = Set({})
		# with open("./train_data/syllable_map") as f:
		# 	for line in f:
		# 		myset.add(line.strip().split(" ")[1][0:-1])
		# with open("./dev_data/syllable_map") as f:
		# 	for line in f:
		# 		myset.add(line.strip().split(" ")[1][0:-1])
		# print(" ".join(list(myset)))
	elif sys.argv[1]=="txt_done_data":
		txt_file = sys.argv[2]
		small_set = Set([])
		with open(txt_file) as f:
			for line in f:
				data_id = line.split(" ")[1]
				small_set.add(data_id)
		# print(small_set)
		with open("train_data_f0_vector") as tf, open("train_data_f0_vector_small","w+") as stf,open("./train_data/syllable_map") as mf:
			tf = tf.readlines()
			mf = mf.readlines()
			for i in range(len(mf)):
				data_id = mf[i].split(" ")[0]
				if data_id in small_set:
					stf.write(tf[i])
			
		with open("./train_data/dct_0") as tf, open("./train_data/dct_0_small","w+") as stf,open("./train_data/syllable_map") as mf:
			tf = tf.readlines()
			mf = mf.readlines()
			count = 0
			for i in range(len(mf)):
				data_id = mf[i].split(" ")[0]
				if data_id in small_set:
					stf.write(" ".join([str(count)]+tf[i].split(" ")[1:]))
					count += 1

		with open("./train_data/syllable_map") as tf, open("./train_data/syllable_map_small","w+") as stf,open("./train_data/syllable_map") as mf:
			tf = tf.readlines()
			mf = mf.readlines()
			for i in range(len(mf)):
				data_id = mf[i].split(" ")[0]
				if data_id in small_set:
					stf.write(tf[i])





		with open("dev_data_f0_vector") as tf, open("dev_data_f0_vector_small","w+") as stf,open("./dev_data/syllable_map") as mf:
			tf = tf.readlines()
			mf = mf.readlines()
			for i in range(len(mf)):
				data_id = mf[i].split(" ")[0]
				if data_id in small_set:
					stf.write(tf[i])
			
		with open("./dev_data/dct_0") as tf, open("./dev_data/dct_0_small","w+") as stf,open("./dev_data/syllable_map") as mf:
			tf = tf.readlines()
			mf = mf.readlines()
			count = 0
			for i in range(len(mf)):
				data_id = mf[i].split(" ")[0]
				if data_id in small_set:
					stf.write(" ".join([str(count)]+tf[i].split(" ")[1:]))
					count += 1

		with open("./dev_data/syllable_map") as tf, open("./dev_data/syllable_map_small","w+") as stf,open("./dev_data/syllable_map") as mf:
			tf = tf.readlines()
			mf = mf.readlines()
			for i in range(len(mf)):
				data_id = mf[i].split(" ")[0]
				if data_id in small_set:
					stf.write(tf[i])








