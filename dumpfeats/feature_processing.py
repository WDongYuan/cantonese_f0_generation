##Extract the previous f0 and next f0 for a syllable

import argparse
import os
import sys
import numpy as np
class Word:
	def __init__(self,feat_name_list,feat_val_list):
		self.feat_dic = {}
		for i in range(len(feat_name_list)):
			self.feat_dic[feat_name_list[i]] = feat_val_list[i]
	def get_feat(self,feat):
		##the val return will be string
		assert feat in self.feat_dic
		return self.feat_dic[feat]

class Phrase:
	def __init__(self,word_list):
		self.word_list = word_list
		self.num_word = len(self.word_list)

class Utt:
	def __init__(self,phrase_list):
		self.phrase_list = phrase_list
		self.num_phrase = len(self.phrase_list)
		self.num_word = sum([phrase.num_word for phrase in self.phrase_list])


def ReadTxtDoneData(path):
	dic = {}
	with open(path) as f:
		for line in f:
			line = line.strip()
			syllable_list = line.split("\"")[1].split(" ")[1:-1]
			syllable_list = [(syllable[0:-1],syllable[-1]) for syllable in syllable_list]

			data_name = line.split(" ")[1]

			dic[data_name] = syllable_list
	return dic

def AlignF0Syllable(phoneme_f0_list,syllable_list):
	syllable_f0_list = []
	# print(phoneme_f0_list)

	phoneme_index = 0
	tmp_syllable_f0 = [[],""]
	for syllable,tone in syllable_list:
		if phoneme_f0_list[phoneme_index][1]=="pau" or phoneme_f0_list[phoneme_index][1]=="ssil":
			while phoneme_f0_list[phoneme_index][1]=="pau" or phoneme_f0_list[phoneme_index][1]=="ssil":
				phoneme_index += 1
			syllable_f0_list.append([phoneme_f0_list[phoneme_index-1][1],[0]])

		while tmp_syllable_f0[1]!=syllable:
			tmp_syllable_f0[0] += phoneme_f0_list[phoneme_index][0]
			tmp_syllable_f0[1] += phoneme_f0_list[phoneme_index][1]
			phoneme_index += 1
			# print(tmp_syllable_f0)

		tmp_syllable_f0[1] += tone
		syllable_f0_list.append([tmp_syllable_f0[1],tmp_syllable_f0[0]])
		tmp_syllable_f0 = [[],""]
		# print(syllable_f0_list)

	##The format will be [syllable_label,f0_value]
	return syllable_f0_list

def AlignF0Phoneme(data_name,f0_dir):
	timeline = []
	f0_list = []
	time = 0

	f0_file = f0_dir+"/"+data_name+".f0"
	with open(f0_file) as f:
		for line in f:
			line = line.strip()

			f0_value = max(float(line.split(" ")[0]),0)
			# if f0_value == -1:
			# 	break

			f0_list.append(f0_value)
			timeline.append(time)
			time += 1
	# print(f0_list)
	# print(timeline)

	pre_end = 0
	phoneme_list = []

	##Align the f0 value and the phoneme
	phoneme_file = f0_dir+"/"+data_name+".phoneme"
	with open(phoneme_file) as f:
		for line in f:
			line = line.strip()

			arr = line.split(" ")
			tmp_time = float(arr[0])*1000
			tmp_time = int(tmp_time/5)
			tmp_label = arr[1]

			phoneme_list.append([f0_list[pre_end:tmp_time+1],tmp_label])
			pre_end = tmp_time
	# print(phoneme_list)

	##Clean the f0 value
	for i in range(len(phoneme_list)):
		tmp_phoneme = phoneme_list[i][0]
		##The majority is 0
		if float(len([1 for pho in tmp_phoneme if pho==0]))/len(tmp_phoneme)>0.8:
			while tmp_phoneme[0]!=0:
				phoneme_list[i-1][0].append(tmp_phoneme.pop(0))
			while tmp_phoneme[-1]!=0:
				phoneme_list[i+1][0] = [tmp_phoneme.pop(-1)]+phoneme_list[i+1][0]
		##The majority is non zero
		elif float(len([1 for pho in tmp_phoneme if pho==0]))/len(tmp_phoneme)<0.2:
			while tmp_phoneme[0]==0:
				phoneme_list[i-1][0].append(tmp_phoneme.pop(0))
			while tmp_phoneme[-1]==0:
				phoneme_list[i+1][0] = [tmp_phoneme.pop(-1)]+phoneme_list[i+1][0]

	##The format of every pari will be [f0_value,phoneme_label]
	return phoneme_list

def PosInPhrase(idx,syl_list):
	begin = idx
	end = idx
	##Find the begin of the phrase
	while begin>0 and syl_list[begin][0].split("_")[-1]!="pau" and syl_list[begin][0].split("_")[-1]!="ssil":
		begin -= 1
	while end<len(syl_list) and syl_list[end][0].split("_")[-1]!="pau" and syl_list[end][0].split("_")[-1]!="ssil":
		end += 1
	length = end-begin+1
	in_percent = float(idx-begin)/length
	in_pos = idx-begin
	out_pos = end-idx
	return in_pos,out_pos,in_percent

def read_syllable_f0_list(file):
	dic = {}
	pre_data_name = ""
	with open(file) as f:
		for line in f:
			line = line.strip().split(",")
			data_name = "_".join(line[0].split("_")[0:2])
			if data_name not in dic:
				dic[data_name] = []
			dic[data_name].append(line[1:])
	return dic



if __name__=="__main__":


	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', dest='mode')
	parser.add_argument('--f0_dir', dest='f0_dir')
	parser.add_argument('--txt_done_data', dest='txt_done_data')
	parser.add_argument('--in_dir_1', dest='in_dir_1')
	parser.add_argument('--in_dir_2', dest='in_dir_2')
	parser.add_argument('--out_dir', dest='out_dir')
	parser.add_argument('--phrase_dir', dest='phrase_dir')
	parser.add_argument('--feat_dir', dest='feat_dir')
	parser.add_argument('--feat_desc', dest='feat_desc')
	parser.add_argument('--syllable_f0_list', dest='syllable_f0_list')
	parser.add_argument('--remove_tone_from_word_idx', dest='word_idx',type=int, default=-1)
	args = parser.parse_args()
	mode = args.mode
	if mode=="how_to_run":
		print("######################################################")
		print("python feature_processing.py --mode extract_feature"+
			" --f0_dir ../DCT/f0_value"+
			" --syllable_f0_list syllable_f0_list.save"+
			" --txt_done_data ../../txt.done.data"+
			" --out_dir ./extract_feature")
		print("######################################################")
		print("python feature_processing.py --mode concat_feature"+
			" --in_dir_1 ./data_feature"+
			" --in_dir_2 ./extract_feature"+
			" --out_dir ./concat_feature"+
			" --remove_tone_from_word_idx 0")
		print("######################################################")
		print("python feature_processing.py --mode remove_tone_in_word"+
			" --in_dir_1 ./data_feature"+
			" --out_dir ./new_data_feature"+
			" --remove_tone_from_word_idx 0")
		print("######################################################")
		print("python feature_processing.py --mode extrac_phrase_feature"+
			" --phrase_dir ../mandarine/gen_f0/phrase_dir/phrase_syllable"+
			" --feat_dir ../mandarine/gen_f0/feature/concat_feature"+
			" --feat_desc ../mandarine/gen_f0/feature_list"+
			" --out_dir ../mandarine/gen_f0/phrase_dir/phrase_feat")
		print("######################################################")
	if mode=="extract_feature":
		# selected_feature = ["last_f0","next_f0","last_mean","next_mean","last_std","next_std"]
		# selected_feature = ["last_f0","last_mean","last_std"]
		selected_feature = ["in_pos","out_pos","in_percent","tone","duration"]

		txt_done_data = args.txt_done_data
		f0_dir = args.f0_dir
		out_dir = args.out_dir

		############################################################################################
		#Read the syllable,phoneme,f0 data
		data_syllable_dic = ReadTxtDoneData(txt_done_data)
		data_dir = f0_dir
		file_list = os.listdir(data_dir)
		syllable_f0_list = []
		'''
		syllable_f0_list = 
		[
		[['data_00002_zung3', [f0 value list]],
		['data_00002_man4', [f0 value list]]],

		[['data_00003_zung3', [f0 value list]],
		['data_00003_man4', [f0 value list]]],

		......
		]
		'''
		for file in file_list:
			if "f0" not in file:
				continue
			data_name = file.split(".")[0]
			tmp_phoneme_f0_list = AlignF0Phoneme(data_name,data_dir)
			tmp_syllable_f0_list = AlignF0Syllable(tmp_phoneme_f0_list,data_syllable_dic[data_name])
			########################################################################################
			for one_syl in tmp_syllable_f0_list:
				one_syl[0] = data_name+"_"+one_syl[0]
			########################################################################################
			syllable_f0_list.append(tmp_syllable_f0_list)
		############################################################################################
		

		##This is the list to record the collected features in order
		feature_in_order = [True]

		os.system("mkdir "+out_dir)
		for syl_l in syllable_f0_list:
			## syl_l contains all the syllable-f0_value pairs in one data file
			## every pair is in format like ['data_00002_zung3', [f0 value list]]
			data_name = "_".join(syl_l[0][0].split("_")[0:2])
			with open(out_dir+"/"+data_name+".feats","w+") as f:
				feature_list = []
				for i in range(len(syl_l)):
					one_syl = syl_l[i]
					if one_syl[0].split("_")[-1]=="ssil" or one_syl[0].split("_")[-1]=="pau":
						continue
					one_syl_feature = [syl_l[i][0]]
					if "last_f0" in selected_feature:
						if feature_in_order[0]:
							feature_in_order.append("last_f0")
						if i==0:
							one_syl_feature.append("0")
						else:
							one_syl_feature.append(str(syl_l[i-1][1][-1]))
					if "next_f0" in selected_feature:
						if feature_in_order[0]:
							feature_in_order.append("next_f0")
						if i==len(syl_l)-1:
							one_syl_feature.append("0")
						else:
							one_syl_feature.append(str(syl_l[i+1][1][0]))
					if "last_mean" in selected_feature:
						if feature_in_order[0]:
							feature_in_order.append("last_mean")
						if i==0:
							one_syl_feature.append("0")
						else:
							one_syl_feature.append(str(np.array(syl_l[i-1][1]).mean()))

					if "next_mean" in selected_feature:
						if feature_in_order[0]:
							feature_in_order.append("next_mean")
						if i==len(syl_l)-1:
							one_syl_feature.append("0")
						else:
							one_syl_feature.append(str(np.array(syl_l[i+1][1]).mean()))
					if "last_std" in selected_feature:
						if feature_in_order[0]:
							feature_in_order.append("last_std")
						if i==0:
							one_syl_feature.append("0")
						else:
							one_syl_feature.append(str(np.array(syl_l[i-1][1]).std()))

					if "next_std" in selected_feature:
						if feature_in_order[0]:
							feature_in_order.append("next_std")
						if i==len(syl_l)-1:
							one_syl_feature.append("0")
						else:
							one_syl_feature.append(str(np.array(syl_l[i+1][1]).std()))

					in_pos,out_pos,in_percent = PosInPhrase(i,syl_l)
					if "in_pos" in selected_feature:
						if feature_in_order[0]:
							feature_in_order.append("in_pos")
						one_syl_feature.append(str(in_pos))
					if "out_pos" in selected_feature:
						if feature_in_order[0]:
							feature_in_order.append("out_pos")
						one_syl_feature.append(str(out_pos))
					if "in_percent" in selected_feature:
						if feature_in_order[0]:
							feature_in_order.append("in_percent")
						one_syl_feature.append(str(round(in_percent,3)))
					if "tone" in selected_feature:
						if feature_in_order[0]:
							feature_in_order.append("tone")
						one_syl_feature.append(syl_l[i][0][-1])
					if "duration" in selected_feature:
						if feature_in_order[0]:
							feature_in_order.append("duration")
						one_syl_feature.append(str(len(syl_l[i][1])))


					feature_list.append(one_syl_feature)
					feature_in_order[0] = False

				for feature in feature_list:
					f.write(" ".join(feature)+"\n")
		print("The features collected in order:")
		feature_in_order.pop(0)
		print(feature_in_order)

	if mode=="concat_feature":
		in_dir_1 = args.in_dir_1
		in_dir_2 = args.in_dir_2
		out_dir = args.out_dir
		word_idx = args.word_idx
		file_list_1 = [file_name for file_name in os.listdir(in_dir_1) if "data" in file_name]
		file_list_2 = [file_name for file_name in os.listdir(in_dir_2) if "data" in file_name]
		os.system("mkdir "+out_dir)

		if len(file_list_1)!=len(file_list_2):
			print("Some files are missing. Use the shorter file list.")
			if len(file_list_1)>len(file_list_2):
				file_list_1 = [file_name for file_name in file_list_2]
			else:
				file_list_2 = [file_name for file_name in file_list_1]


		file_list_1 = sorted(file_list_1)
		file_list_2 = sorted(file_list_2)
		for i in range(len(file_list_1)):
			file1 = file_list_1[i]
			file2 = file_list_2[i]
			assert file1==file2, file1 +" does not equal to " + file2

			with open(in_dir_1+"/"+file1) as f1, open(in_dir_2+"/"+file2) as f2, open(out_dir+"/"+file1,"w+") as of:
				for l1,l2 in zip(f1,f2):
					l1 = l1.strip().split(" ")
					l2 = l2.strip().split(" ")
					# assert l1[0].split("_")[-1]==l2[0].split("_")[-1]
					if word_idx!=-1:
						l1[word_idx] = l1[word_idx][0:-1]
					l1 += l2[1:len(l2)]
					of.write(" ".join(l1)+"\n")
		print("please add feature in the feature description if you want to predict on this feature")

	if mode=="remove_tone_in_word":
		in_dir_1 = args.in_dir_1
		out_dir = args.out_dir
		word_idx = args.word_idx
		file_list_1 = [file_name for file_name in os.listdir(in_dir_1) if "data" in file_name]
		os.system("mkdir "+out_dir)

		for i in range(len(file_list_1)):
			file1 = file_list_1[i]
			with open(in_dir_1+"/"+file1) as f1, open(out_dir+"/"+file1,"w+") as of:
				for l1 in f1:
					l1 = l1.strip().split(" ")
					l1[word_idx] = l1[word_idx][0:-1]
					of.write(" ".join(l1)+"\n")
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print("remember to check the feature description if you want to predict on this feature")
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

	if mode=="extrac_phrase_feature":
		phrase_dir = args.phrase_dir
		feat_dir = args.feat_dir
		out_dir = args.out_dir
		feat_desc = args.feat_desc
		os.system("mkdir "+out_dir)

		feat_name_list = []
		with open(feat_desc) as f:
			for line in f:
				feat_name_list.append(line.strip())
		feat_file_list = os.listdir(feat_dir)
		file_list = os.listdir(phrase_dir)
		file_list = [file for file in file_list if "data" in file]
		extract_feature = ["pre_num_word","num_word","next_num_word","utt_num_phrase","utt_num_word","phrase_pos","phrase_pos_ratio",
		"first_pos","last_pos","first_tone","last_tone","pre_tone","next_tone","duration"]
		print("extracted features:")
		print("\n".join(extract_feature))
		with open(out_dir+"/0_phrase_feature","w+") as f:
			f.write("\n".join(extract_feature)+"\n")

		for file in file_list:
			utt = None
			if file+".feats" not in feat_file_list:
				continue
			with open(phrase_dir+"/"+file) as phf, open(feat_dir+"/"+file+".feats") as featf, open(out_dir+"/"+file+".feats","w+") as outf:
				phrase_list = []
				for line in phf:
					line = line.strip().split(" ")
					word_list = []
					for i in range(len(line)):
						word_list.append(Word(feat_name_list,featf.readline().strip().split(" ")))
					phrase_list.append(Phrase(word_list))
				utt = Utt(phrase_list)

				for i in range(len(utt.phrase_list)):
					feat_val_list = []
					##should follow the order in the list!!
					#pre_num_word
					if i==0:
						feat_val_list.append(0)
					else:
						feat_val_list.append(utt.phrase_list[i-1].num_word)
					#num_word
					feat_val_list.append(utt.phrase_list[i].num_word)

					#next_num_word
					if i==len(utt.phrase_list)-1:
						feat_val_list.append(0)
					else:
						feat_val_list.append(utt.phrase_list[i+1].num_word)

					#utt_num_phrase
					feat_val_list.append(utt.num_phrase)

					#utt_num_word
					feat_val_list.append(utt.num_word)

					#phrase_pos
					feat_val_list.append(i)

					#phrase_pos_ratio
					feat_val_list.append(round(float(i)/utt.num_phrase,3))

					#first_pos
					feat_val_list.append(sum([tmp_phrase.num_word for tmp_phrase in utt.phrase_list[0:i]]))

					#last_pos
					feat_val_list.append(sum([tmp_phrase.num_word for tmp_phrase in utt.phrase_list[0:i+1]])-1)

					#first_tone
					feat_val_list.append(utt.phrase_list[i].word_list[0].get_feat("tone"))

					#last_tone
					feat_val_list.append(utt.phrase_list[i].word_list[-1].get_feat("tone"))

					#pre_tone
					if i==0:
						feat_val_list.append(0)
					else:
						feat_val_list.append(utt.phrase_list[i-1].word_list[0].get_feat("tone"))

					#next_tone"
					if i==len(utt.phrase_list)-1:
						feat_val_list.append(0)
					else:
						feat_val_list.append(utt.phrase_list[i+1].word_list[0].get_feat("tone"))

					#duration
					feat_val_list.append(sum([int(tmp_word.get_feat("duration")) for tmp_word in utt.phrase_list[i].word_list]))

					outf.write(" ".join([str(feat_val) for feat_val in feat_val_list])+"\n")

































