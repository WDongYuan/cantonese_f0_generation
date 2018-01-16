import os
utt_path = "../mandarine/cmu_yue_wdy_cnsmall/festival/utts"
utt_path = "../dumpfeats/utts"
file_list = os.listdir(utt_path)
with open("./my_file_list","w+") as f:
	for file in file_list:
		print(file)
		f.write(utt_path+"/"+file+"\n")
