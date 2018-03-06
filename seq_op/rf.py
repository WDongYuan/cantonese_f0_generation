import numpy as np
import os
class RandomForest:
	def __init__(self,tree_num,label_ratio,feat_ratio,train_label,train_feat,test_label,test_feat,desc_file,out_dir,stop_size=30):
		self.tree_num = tree_num
		self.out_dir = out_dir
		self.train_label = train_label
		self.train_feat = train_feat
		self.test_label = test_label
		self.test_feat = test_feat
		self.desc_file = desc_file
		self.feat_num = np.loadtxt(self.test_feat,delimiter=" ",dtype=str).shape[1]
		self.label_num = np.loadtxt(self.test_label,delimiter=" ",dtype=str).shape[1]
		self.test_n = np.loadtxt(self.test_label,delimiter=" ",dtype=str).shape[0]
		self.stop_size = stop_size


		self.feat_ratio = feat_ratio
		self.select_feat = int(self.feat_ratio*(self.feat_num-1))
		self.label_ratio = label_ratio
		self.select_label = int(self.label_ratio*self.label_num)

		self.tree_feat = []
		for count in range(tree_num):
			tmp_feat_idx = np.arange(self.select_feat+1)
			tmp_feat_idx[1:] = np.random.choice(np.arange(1,self.feat_num),self.select_feat,replace=False)
			self.tree_feat.append(tmp_feat_idx)
		self.tree_label = []
		for count in range(tree_num):
			self.tree_label.append(np.random.choice(self.label_num,self.select_label,replace=False))
		
		self.dump = out_dir+"/rf_dump"
		os.system("mkdir "+self.dump)

		self.tree_predict = []
	def predict_vector(self,train_data,train_label,dev_data,dev_label,out_dir,feat_desc,log="",stop_size=30):
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

	def sample_data(self,label_idx,feat_idx,out_dir):
		tmp_data = np.loadtxt(self.train_label,delimiter=" ",dtype=str)
		np.savetxt(out_dir+"/rf_train_label",tmp_data[:,label_idx],delimiter=" ",fmt="%s")

		tmp_data = np.loadtxt(self.train_feat,delimiter=" ",dtype=str)
		np.savetxt(out_dir+"/rf_train_feat",tmp_data[:,feat_idx],delimiter=" ",fmt="%s")

		tmp_data = np.loadtxt(self.test_label,delimiter=" ",dtype=str)
		np.savetxt(out_dir+"/rf_test_label",tmp_data[:,label_idx],delimiter=" ",fmt="%s")

		tmp_data = np.loadtxt(self.test_feat,delimiter=" ",dtype=str)
		np.savetxt(out_dir+"/rf_test_feat",tmp_data[:,feat_idx],delimiter=" ",fmt="%s")


		with open(self.desc_file) as inf, open(out_dir+"/rf_desc","w+") as outf:
			desc = inf.readlines()
			for i in range(len(feat_idx)):
				outf.write(desc[feat_idx[i]])
			outf.write(desc[-1])

	def train(self):
		for tree_i in range(self.tree_num):
			print(">>>>>>>>>> processing tree "+str(tree_i))
			tree_dir = self.out_dir+"/tree"+str(tree_i)
			os.system("mkdir "+tree_dir)
			self.sample_data(self.tree_label[tree_i],self.tree_feat[tree_i],tree_dir)

			os.system("mkdir "+tree_dir+"/tree_predict")
			self.tree_predict.append(
				self.predict_vector(
					tree_dir+"/rf_train_feat",
					tree_dir+"/rf_train_label",
					tree_dir+"/rf_test_feat",
					tree_dir+"/rf_test_label",
					tree_dir+"/tree_predict",
					tree_dir+"/rf_desc",
					stop_size=self.stop_size
					))
		#combine the prediction from different trees
		comb_predict = np.zeros((self.test_n,self.label_num))
		count_arr = np.zeros((self.label_num,))
		for tree_i in range(self.tree_num):
			count_arr[self.tree_label[tree_i]] += 1
			comb_predict[:,self.tree_label[tree_i]] += np.loadtxt(self.tree_predict[tree_i],delimiter=" ")
		comb_predict /= count_arr

		out_file = self.out_dir+"/rf_predict_val"
		np.savetxt(out_file,comb_predict,fmt="%.5f")
		return out_file




if __name__=="__main__":
	rf = RandomForest(
		2,
		"/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/gen_audio/src/mandarine/dnn_data_dir/train_test_data/test_data/test_f0",
		"/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/gen_audio/src/mandarine/dnn_data_dir/train_test_data/test_data/test_feat",
		"/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/gen_audio/src/mandarine/dnn_data_dir/train_test_data/test_data/test_f0",
		"/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/gen_audio/src/mandarine/dnn_data_dir/train_test_data/test_data/test_feat",
		"/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/gen_audio/src/mandarine/dnn_data_dir/feature_desc_vector",
		"./rf_dump")
	rf.train()

