import sys
import os
import numpy as np
import os.path

if __name__=="__main__":
	if len(sys.argv)==1:
		print("python drop_feature.py train_feat dev_feat val_desc vector_desc")
		print("python drop_feature.py ./train_data/dct_0 ./dev_data/dct_0 ./feature_desc_val ./feature_desc_vector")
		exit()
	train_feat = sys.argv[1]
	dev_feat = sys.argv[2]
	val_desc = sys.argv[3]
	vec_desc = sys.argv[4]

	assert not os.path.exists(train_feat+"_old")
	assert not os.path.exists(dev_feat+"_old")
	assert not os.path.exists(val_desc+"_old")
	assert not os.path.exists(vec_desc+"_old")
	# exit()

	feat_list = ["syl_endpitch","syl_startpitch","p.syl_endpitch","n.syl_startpitch"]
	feat_idx = []

	desc = None
	with open(val_desc) as f:
		desc = f.readlines()

	for feat_name in feat_list:
		for i in range(len(desc)):
			if feat_name in desc[i]:
				feat_idx.append(i)
				break
	print("drop feature index:")
	print(feat_idx)

	##new desc
	desc = None
	with open(val_desc) as f:
		desc = f.readlines()
	os.system("mv "+val_desc+" "+val_desc+"_old")
	with open(val_desc,"w+") as f:
		f.writelines([desc[i] for i in range(len(desc)) if i not in feat_idx])

	desc = None
	with open(vec_desc) as f:
		desc = f.readlines()
	os.system("mv "+vec_desc+" "+vec_desc+"_old")
	with open(vec_desc,"w+") as f:
		f.writelines([desc[i] for i in range(len(desc)) if i not in feat_idx])

	##new feature
	tmp_arr = np.loadtxt(train_feat,delimiter=" ",dtype=np.str)
	os.system("mv "+train_feat+" "+train_feat+"_old")
	tmp_arr = np.delete(tmp_arr,feat_idx,1)
	np.savetxt(train_feat,tmp_arr,delimiter=" ",fmt="%s")

	tmp_arr = np.loadtxt(dev_feat,delimiter=" ",dtype=np.str)
	os.system("mv "+dev_feat+" "+dev_feat+"_old")
	tmp_arr = np.delete(tmp_arr,feat_idx,1)
	np.savetxt(dev_feat,tmp_arr,delimiter=" ",fmt="%s")






