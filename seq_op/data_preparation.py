import argparse
import os
import numpy as np
from func import *


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode',dest='mode')
	parser.add_argument('--out_dir',dest='out_dir')
	parser.add_argument('--voice_dir',dest='voice_dir')
	parser.add_argument('--voice_dir_f0',dest='voice_dir_f0')
	parser.add_argument('--voice_dir_lab',dest='voice_dir_lab')
	parser.add_argument('--voice_dir_utt',dest='voice_dir_utt')
	parser.add_argument('--txt_done_data',dest='txt_done_data')
	parser.add_argument('--feature_list',dest='feature_list')
	parser.add_argument('--consonant_vowel_file',dest='consonant_vowel_file')
	parser.add_argument('--data_pos_file',dest='data_pos_file')
	parser.add_argument('--pos_dic',dest='pos_dic',default="")
	parser.add_argument('--word_dic',dest='word_dic',default="")
	args = parser.parse_args()

	mode = args.mode
	if mode=="how_to_run":
		print("python data_preparation.py"+
			" --mode run"+
			" --voice_dir ../mandarine/cmu_yue_wdy_zhall"+
			" --voice_dir_f0 ../mandarine/cmu_yue_wdy_zhall/f0"+
			" --voice_dir_lab ../mandarine/cmu_yue_wdy_zhall/lab"+
			" --voice_dir_utt ../mandarine/cmu_yue_wdy_zhall/festival/utts"+
			" --txt_done_data ../mandarine/txt.done.data"+
			" --feature_list ./feature_list"+
			" --consonant_vowel_file ../mandarine/consonant_vowel"+
			" --data_pos_file ../dnn/lstm_data/refine_txt_token_pos"+
			" --pos_dic(optional default=\"\")"+
			" --word_dic(optional default=\"\")"+
			" --out_dir ./my_data")
	elif mode=="run":
		out_dir = args.out_dir
		voice_dir = args.voice_dir
		voice_dir_f0 = args.voice_dir_f0
		voice_dir_lab = args.voice_dir_lab
		voice_dir_utt = args.voice_dir_utt
		txt_done_data = args.txt_done_data
		feature_list = args.feature_list
		consonant_vowel_file = args.consonant_vowel_file
		data_pos_file = args.data_pos_file

		os.system("mkdir "+out_dir)
		os.system("mkdir "+out_dir+"/f0_value")

		# print("")
		# print(">>>>>>>>>>extract f0 into f0_value<<<<<<<<<<")
		# os.system("python ../DCT/run.py"+
		# 	" --mode extract_f0"+
		# 	" --in_dir "+voice_dir_f0+
		# 	" --out_dir "+out_dir+"/f0_value")

		# print("")
		# print(">>>>>>>>>>extract phoneme into f0_value<<<<<<<<<<")
		# os.system("python ../DCT/run.py"+
		# 	" --mode extract_phoneme"+
		# 	" --in_dir "+voice_dir_lab+
		# 	" --out_dir "+out_dir+"/f0_value")

		print("")
		print(">>>>>>>>>>align f0 and phoneme<<<<<<<<<<")
		os.system("python ../DCT/run.py"+
			" --mode save_f0_syllable_list"+
			" --txt_done_data "+txt_done_data+
			" --f0_dir "+out_dir+"/f0_value"+
			" --out_file "+out_dir+"/syllable_f0_list.save")

		print("")
		print(">>>>>>>>>>subsample f0 value<<<<<<<<<<")
		os.system("python ../DCT/run.py"+
			" --mode f0_subsampling"+
			" --syllable_f0_file "+out_dir+"/syllable_f0_list.save"+
			" --out_file "+out_dir+"/subsample_f0.save"
			" --number 10")

		print("")
		print(">>>>>>>>>>put f0 in files<<<<<<<<<<")
		os.system("python ../DCT/run.py"+
			" --mode put_f0_in_file"+
			" --subsample_file "+out_dir+"/subsample_f0.save"+
			" --out_dir "+out_dir+"/f0_in_file")

		print("")
		print(">>>>>>>>>>put syllable in files<<<<<<<<<<")
		os.system("python ../DCT/run.py"+
			" --mode put_syllable_in_file"+
			" --txt_done_data "+txt_done_data+
			" --out_dir "+out_dir+"/syllable_in_file")

		print("")
		print(">>>>>>>>>>dump feature<<<<<<<<<<")
		print("----->create file list")
		file_list = os.listdir(voice_dir_utt)
		file_list = [file for file in file_list if "utt" in file]
		with open(out_dir+"/utt_file_list","w+") as f:
			for file in file_list:
				f.write(voice_dir_utt+"/"+file+"\n")
		print("----->extract feature from festival")
		os.system("nohup python ../dumpfeats/run.py "+feature_list+" "+out_dir+"/utt_file_list "+out_dir+"/data_feature "+voice_dir)

		print("")
		print(">>>>>>>>>>self extract features<<<<<<<<<<")
		os.system("python ../dumpfeats/feature_processing.py --mode extract_feature"+
			" --f0_dir "+out_dir+"/f0_value"+
			" --txt_done_data "+txt_done_data+
			" --out_dir "+out_dir+"/self_extract_feature")
		
		print("")
		#################################################
		#for feature with no word name
		##############print(">>>>>>>>>>concat features<<<<<<<<<<")
		##############os.system("python ../dumpfeats/feature_processing.py --mode concat_feature"+
		##############	" --in_dir_1 "+out_dir+"/data_feature"+
		##############	" --in_dir_2 "+out_dir+"/self_extract_feature"+
		##############	" --out_dir "+out_dir+"/concat_feature"+
		##############	" --remove_tone_from_word_idx -1")
		##################################################
		print(">>>>>>>>>>concat features<<<<<<<<<<")
		os.system("python ../dumpfeats/feature_processing.py --mode concat_feature"+
			" --in_dir_1 "+out_dir+"/data_feature"+
			" --in_dir_2 "+out_dir+"/self_extract_feature"+
			" --out_dir "+out_dir+"/concat_feature"+
			" --remove_tone_from_word_idx 0")
		##################################################

		print("")
		print(">>>>>>>>>>create new concat feature list<<<<<<<<<<")
		# ['in_pos', 'out_pos', 'in_percent', 'tone']
		feat_l = None
		with open(feature_list) as f:
			feat_l = f.readlines()
		feat_l = [line.strip() for line in feat_l]
		feat_l.append("in_pos")
		feat_l.append("out_pos")
		feat_l.append("in_percent")
		feat_l.append("tone")
		feat_l.append("duration")
		with open(out_dir+"/concat_feature_list","w+") as f:
			f.write("\n".join(feat_l)+"\n")

		print("")
		print(">>>>>>>>>>create train test file list<<<<<<<<<<")
		os.system("python ../decision_tree/data_preprocessing.py"+
			" --mode create_train_test_file_list"+
			" --f0_in_file "+out_dir+"/f0_in_file"+
			" --train_ratio 0.8"+
			# " --dev_ratio(option) "+
			" --out_file "+out_dir+"/train_test_list")

		print("")
		print(">>>>>>>>>>create feature description file for decision tree<<<<<<<<<<")		
		os.system("python ../decision_tree/data_preprocessing.py"+
			" --mode create_train_dev_vector_data"+
			" --syl_in_file "+out_dir+"/syllable_in_file"+
			" --f0_in_file "+out_dir+"/f0_in_file"+
			" --feat_in_file "+out_dir+"/concat_feature"+
			" --train_test_file_list "+out_dir+"/train_test_list"+
			" --out_dir "+out_dir+"/train_test_data")
		os.system("python ../decision_tree/data_preprocessing.py"+
			" --mode create_feature_desc"+
			" --in_file "+out_dir+"/train_test_data/train_data/train_feat"
			" --feature_file "+out_dir+"/concat_feature_list"
			" --out_file "+out_dir+"/feature_desc_vector")
		print("modify description file feature type:")
		os.system("open "+out_dir+"/feature_desc_vector")
		finish = raw_input("finish?(y/n)")
		while finish!="y":
			finish = raw_input("finish?(y/n)")
		os.system("cp "+out_dir+"/feature_desc_vector "+out_dir+"/feature_desc_val")
		val_desc = None
		with open(out_dir+"/feature_desc_val") as f:
			val_desc = f.readlines()
		val_desc[0] = "((dct_coef float)\n"
		with open(out_dir+"/feature_desc_val","w+") as f:
			f.writelines(val_desc)
		os.system("rm -r "+out_dir+"/train_test_data")


		################################################
		print("")
		# for Chinese data
		print(">>>>>>>>>>extract phrase<<<<<<<<<<")
		os.system("mkdir "+out_dir+"/phrase_dir")
		os.system("python ../DCT/run.py"+
			" --mode extract_phrase"+
			" --f0_dir "+out_dir+"/f0_value"+
			" --out_dir "+out_dir+"/phrase_dir/phrase_syllable"+
			" --consonant_vowel_file "+consonant_vowel_file)
		#################################################
		# # for Cantonese data from Festival
		# print(">>>>>>>>>>extract phrase<<<<<<<<<<")
		# os.system("mkdir "+out_dir+"/phrase_dir")
		# os.system("python ../DCT/run.py"+
		# 	" --mode extract_can_phrase"+
		# 	" --f0_dir "+out_dir+"/f0_value"+
		# 	" --out_dir "+out_dir+"/phrase_dir/phrase_syllable"+
		# 	" --txt_done_data "+txt_done_data+
		# 	" --consonant_vowel_file "+consonant_vowel_file)
		#################################################
		# # for Cantonese data from cuprosody
		# print(">>>>>>>>>>extract phrase<<<<<<<<<<")
		# os.system("mkdir "+out_dir+"/phrase_dir")
		# os.system("python ../cantonese/tool.py"+
		# 	" --mode extract_phrase"+
		# 	" --alignment_file ../CUPROSODY/disk1/Alignment/result_1300_new.mlf"+
		# 	" --out_dir "+out_dir+"/phrase_dir/phrase_syllable")
		#################################################


		###############################################################################################################################
		print("")
		print(">>>>>>>>>>prepare to update the feature description file<<<<<<<<<<")
		vec_desc = None
		with open(out_dir+"/feature_desc_vector") as f:
			vec_desc = f.readlines()
		vec_desc.pop(-1)
		val_desc = None
		with open(out_dir+"/feature_desc_val") as f:
			val_desc = f.readlines()
		val_desc.pop(-1)

		print(">>>>>>>>>>extract phrase feature<<<<<<<<<<")
		append_phrase_to_feature(out_dir+"/concat_feature",out_dir+"/phrase_dir/phrase_syllable")
		vec_desc.append("(phr_pos float)\n")
		vec_desc.append("(phr_pos_percent float)\n")
		vec_desc.append("(phr_num float)\n")
		vec_desc.append("(syl_pos_in_phr float)\n")
		vec_desc.append("(syl_percent_in_phr float)\n")
		vec_desc.append("(syl_num_in_phr float)\n")
		val_desc.append("(phr_pos float)\n")
		val_desc.append("(phr_pos_percent float)\n")
		val_desc.append("(phr_num float)\n")
		val_desc.append("(syl_pos_in_phr float)\n")
		val_desc.append("(syl_percent_in_phr float)\n")
		val_desc.append("(syl_num_in_phr float)\n")
	
		
		print("")
		print(">>>>>>>>>>extract consonant and vowel<<<<<<<<<<")
		consonant_dic,vowel_dic = get_syl_dic(consonant_vowel_file)
		append_syl_to_feature(out_dir+"/concat_feature",txt_done_data,consonant_dic,vowel_dic)
		vec_desc.append("(consonant "+" ".join([str(idx) for idx in range(len(consonant_dic))])+")\n")
		vec_desc.append("(vowel "+" ".join([str(idx) for idx in range(len(vowel_dic))])+")\n")
		val_desc.append("(consonant "+" ".join([str(idx) for idx in range(len(consonant_dic))])+")\n")
		val_desc.append("(vowel "+" ".join([str(idx) for idx in range(len(vowel_dic))])+")\n")

		print("")
		print(">>>>>>>>>>extract part of speech features and word index<<<<<<<<<<")
		if args.word_dic!="":
			print("load word dictionary.")
			word_dic = load_dic(args.dic)
		else:
			# word_dic = get_word_dic("../dnn/lstm_data/refine_txt_token_pos",0)
			word_dic = get_word_dic(data_pos_file,0)
			save_dic(word_dic,out_dir+"/word_dic")
		if args.pos_dic!="":
			print("load pos dictionary.")
			pos_dic = load_dic(args.pos_dic)
		else:
			# pos_dic = get_pos_dic("../dnn/lstm_data/refine_txt_token_pos")
			pos_dic = get_pos_dic(data_pos_file)
			save_dic(pos_dic,out_dir+"/pos_dic")
		print(pos_dic)
		append_pos_to_feature(out_dir+"/concat_feature",data_pos_file,pos_dic,word_dic)
		vec_desc.append("(part_of_speech "+" ".join([str(idx) for idx in range(len(pos_dic)+1)])+")\n")
		vec_desc.append("(pre_part_of_speech "+" ".join([str(idx) for idx in range(len(pos_dic)+1)])+")\n")
		vec_desc.append("(next_part_of_speech "+" ".join([str(idx) for idx in range(len(pos_dic)+1)])+")\n")
		vec_desc.append("(token_pos_in_utt float)\n")
		vec_desc.append("(char_pos_in_token float)\n")
		vec_desc.append("(word_index "+" ".join([str(idx) for idx in range(len(word_dic)+1)])+")\n")

		val_desc.append("(part_of_speech "+" ".join([str(idx) for idx in range(len(pos_dic)+1)])+")\n")
		val_desc.append("(pre_part_of_speech "+" ".join([str(idx) for idx in range(len(pos_dic)+1)])+")\n")
		val_desc.append("(next_part_of_speech "+" ".join([str(idx) for idx in range(len(pos_dic)+1)])+")\n")
		val_desc.append("(token_pos_in_utt float)\n")
		val_desc.append("(char_pos_in_token float)\n")
		val_desc.append("(word_index "+" ".join([str(idx) for idx in range(len(word_dic)+1)])+")\n")

		print("")
		print(">>>>>>>>>>update feature description file<<<<<<<<<<")
		vec_desc.append(")\n")
		val_desc.append(")\n")
		with open(out_dir+"/new_feature_desc_vector","w+") as f:
			f.writelines(vec_desc)
		with open(out_dir+"/new_feature_desc_val","w+") as f:
			f.writelines(val_desc)
		###############################################################################################################################



		print("")
		print(">>>>>>>>>>calculate phrase level dct<<<<<<<<<<")
		os.system("python ../DCT/run.py --mode phrase_f0_dct"+
			" --subsample_file "+out_dir+"/subsample_f0.save"+
			" --phrase_dir "+out_dir+"/phrase_dir/phrase_syllable"+
			" --out_dir "+out_dir+"/phrase_dir/phrase_f0_dct"+
			" --option phrase"
			" --number 2")

		print("")
		print(">>>>>>>>>>calculate phrase residual value directory<<<<<<<<<<")
		os.system("python ../decision_tree/data_preprocessing.py"+
			" --mode dir_minus"+
			" --dir1 "+out_dir+"/f0_in_file"
			" --dir2 "+out_dir+"/phrase_dir/phrase_f0_dct/f0"+
			" --begin1 0"+
			" --begin2 0"+
			" --out_dir "+out_dir+"/phrase_dir/phrase_f0_dct/f0_res")

		print("")
		print(">>>>>>>>>>extract phrase level feature<<<<<<<<<<")
		os.system("python ../dumpfeats/feature_processing.py --mode extrac_phrase_feature"+
			" --phrase_dir "+out_dir+"/phrase_dir/phrase_syllable"+
			" --feat_dir "+out_dir+"/concat_feature"+
			" --feat_desc "+out_dir+"/concat_feature_list"+
			" --out_dir "+out_dir+"/phrase_dir/phrase_feat")
		


		print("")
		print(">>>>>>>>>>create train data and test data<<<<<<<<<<")
		os.system("python ../decision_tree/data_preprocessing.py"+
			" --mode create_train_dev_vector_data"+
			" --syl_in_file "+out_dir+"/syllable_in_file"+
			" --f0_in_file "+out_dir+"/f0_in_file"+
			" --feat_in_file "+out_dir+"/concat_feature"+
			" --train_test_file_list "+out_dir+"/train_test_list"+
			" --out_dir "+out_dir+"/train_test_data")

		print("")
		print(">>>>>>>>>>create phrase f0 train and test data<<<<<<<<<<")
		os.system("python ../decision_tree/data_preprocessing.py"+
			" --mode create_general_train_dev_vector_data"+
			" --f0_in_file "+out_dir+"/phrase_dir/phrase_f0_dct/dct"+
			" --feat_in_file "+out_dir+"/phrase_dir/phrase_feat"
			" --train_test_file_list "+out_dir+"/train_test_list"+
			" --out_dir "+out_dir+"/phrase_dir/phrase_f0_train_test")

		print("")
		print(">>>>>>>>>>create phrase residual f0 train and test data<<<<<<<<<<")
		os.system("python ../decision_tree/data_preprocessing.py"+
			" --mode create_train_dev_vector_data"+
			" --syl_in_file "+out_dir+"/syllable_in_file"+
			" --f0_in_file "+out_dir+"/phrase_dir/phrase_f0_dct/f0_res"+
			" --feat_in_file "+out_dir+"/concat_feature"+
			" --train_test_file_list "+out_dir+"/train_test_list"+
			" --out_dir "+out_dir+"/phrase_dir/phrase_f0_res_train_test")

		print("")
		print(">>>>>>>>>>create phrase feature description file<<<<<<<<<<")
		os.system("python ../decision_tree/data_preprocessing.py"+
			" --mode create_feature_desc"+
			" --in_file "+out_dir+"/phrase_dir/phrase_f0_train_test/train_data/train_feat"+
			" --feature_file "+out_dir+"/phrase_dir/phrase_feat/0_phrase_feature"+
			" --out_file "+out_dir+"/phrase_dir/feature_desc_vector")

		
















