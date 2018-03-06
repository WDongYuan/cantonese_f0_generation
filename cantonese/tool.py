# -*- coding: utf-8 -*-
import string
import matplotlib
matplotlib.use('TkAgg')
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sets import Set
import re
def get_consonant_vowel(cv_file):
	cons = None
	vowel = None
	with open(cv_file) as f:
		cons = Set(f.readline().strip().split(" "))
		vowel = Set(f.readline().strip().split(" "))
	return cons,vowel

def decompose_syllable(cons_set,vowel_set,syl):
	cons = ""
	vowel = ""
	p = 1
	while syl[0:p] in cons_set:
		p += 1
	cons = syl[0:p-1]
	vowel = syl[p-1:-1]
	tone = syl[-1]
	return cons,vowel,tone


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', dest='mode')
	parser.add_argument('--word_file', dest='word_file')
	parser.add_argument('--syllable_file', dest='syllable_file')
	parser.add_argument('--out_dir', dest='out_dir')
	parser.add_argument('--out_file', dest='out_file')
	parser.add_argument('--alignment_file', dest='alignment_file')
	parser.add_argument('--consonant_vowel', dest='consonant_vowel')
	parser.add_argument('--txt_done_data', dest='txt_done_data')
	parser.add_argument('--macro_prosody', dest='macro_prosody')
	args = parser.parse_args()
	mode = args.mode

	if mode=="how_to_run":
		print("python tool.py"+
			" --mode gen_txt_done_data"+
			" --word_file ../CUPROSODY/disk1/Transcription/chinese_with_segment"+
			" --syllable_file ../CUPROSODY/disk1/Transcription/LSHK_with_segment"+
			" --out_dir ./")
		print("python tool.py"+
			" --mode extract_consonant_vowel"+
			" --alignment_file ../CUPROSODY/disk1/Alignment/result_1300.mlf"+
			" --out_file ./consonant_vowel")
		print("python tool.py"+
			" --mode gen_addenda_file"+
			" --consonant_vowel ./consonant_vowel"+
			" --txt_done_data ./txt.done.data"
			" --out_file ./cmu_yue_addenda.scm")
		print("python tool.py"+
			" --mode gen_pos_file"+
			" --macro_prosody ../CUPROSODY/disk1/ProsodicData/macro_prosody"+
			" --txt_done_data ./txt.done.data"
			" --out_file ./txt_token_pos")
		print("python tool.py"+
			" --mode extract_phrase"+
			" --alignment_file ../CUPROSODY/disk1/Alignment/result_1300.mlf"+
			" --out_dir ./phrase_dir/phrase_syllable")
	elif mode=="gen_txt_done_data":
		word_file = args.word_file
		syl_file = args.syllable_file
		out_dir = args.out_dir
		syl_dic = {}
		with open(syl_file) as sylf, open(out_dir+"/txt.done.data","w+") as txt:
			for line in sylf:
				line = re.split(r'\s|\t',line.strip())
				index = line[0]
				for i in range(len(index),5):
					index = "0"+index
				index = "data_"+index
				syl_l = []
				for phrase in line[1:]:
					syl_l += phrase.split("-")
				syl_dic[index] = syl_l
				txt.write("( "+index+" \" "+" ".join(syl_l)+" \" )"+"\n")
		print("---------->file saved to "+out_dir+"/txt.done.data")

		with open(word_file) as wordf, open(out_dir+"/txt.done.data.word","w+") as txt:
			for line in wordf:
				line = re.split(r'\s|\t',line.strip())
				index = line[0]
				for i in range(len(index),5):
					index = "0"+index
				index = "data_"+index

				sentence = "".join(line[1:])
				sentence = string.replace(sentence,"『","")
				sentence = string.replace(sentence,"』","")
				sentence = string.replace(sentence,"﹖","")
				sentence = string.replace(sentence,"？","")
				sentence = string.replace(sentence,"卅","卅啊")
				sentence = string.replace(sentence,"𧙗","佑")
				if len(sentence.decode("utf-8"))!=len(syl_dic[index]):
					print(syl_dic[index])
					print(len(syl_dic[index]))
					print(sentence)
					print(len(sentence.decode("utf-8")))
					for tmp_word in sentence.decode("utf-8"):
						print(tmp_word+"-")

				txt.write("( "+index+" \""+sentence+".\" )"+"\n")
		print("---------->file saved to "+out_dir+"/txt.done.data.word")

	elif mode=="extract_consonant_vowel":
		align_file = args.alignment_file
		out_file = args.out_file

		cons_set = Set([])
		vowel_set = Set([])
		with open(align_file) as af:
			for line in af:
				if "_" not in line:
					continue
				line = line.split(" ")
				tup = line[2].split("_")
				if tup[0]=="I":
					cons_set.add(tup[1])
				elif tup[0]=="F":
					vowel_set.add(tup[1])
				else:
					raise Exception("extraction fault !!!")
		with open(out_file,"w+") as f:
			f.write(" ".join(list(cons_set))+"\n")
			f.write(" ".join(list(vowel_set))+"\n")
		print("---------->file saved to "+out_file)

	elif mode=="gen_addenda_file":
		#(lex.add.entry '("hua4" nn (((h ua) 4)))

		##consonant vowel file
		cv_file = args.consonant_vowel
		txt = args.txt_done_data
		out_file = args.out_file

		cons_set,vowel_set = get_consonant_vowel(cv_file)

		syl_set = Set([])
		with open(txt) as f:
			for line in f:
				syl = line.strip().split(" ")[3:-2]
				for tmp in syl:
					syl_set.add(tmp)

		with open(out_file,"w+") as f:
			for syl in syl_set:
				cons,vowel,tone = decompose_syllable(cons_set,vowel_set,syl)
				f.write("(lex.add.entry '(\""+syl+"\" nn ((("+cons+" "+vowel+") "+tone+"))))\n")
		print("---------->file saved to "+out_file)

	elif mode=="test":
		# old_phone = Set([])
		# with open("cmu_yue_wdy_phoneset.scm") as f:
		# 	lines = f.readlines()[62:101]
		# 	lines = [line.strip() for line in lines]
		# 	for line in lines:
		# 		if "(" not in line:
		# 			continue
		# 		old_phone.add(line.split(" ")[1])
		# # print(old_phone)

		# cons = None
		# vowel = None
		# with open("consonant_vowel") as f:
		# 	cons = Set(f.readline().strip().split(" "))
		# 	vowel = Set(f.readline().strip().split(" "))

		# new_phone = Set([])
		# for phone in cons:
		# 	if phone not in old_phone:
		# 		new_phone.add(phone)
		# for phone in vowel:
		# 	if phone not in old_phone:
		# 		new_phone.add(phone)
		# for phone in new_phone:
		# 	print(phone+"".join([" " for i in range(4-len(phone))]))

		# file_list = os.listdir("./wav")
		# for file in file_list:
		# 	if "wav" not in file:
		# 		continue
		# 	# new_file_name = "".join(["0" for i in range(9-len(file))])+file
		# 	new_file_name = "data_"+file
		# 	os.system("mv ./wav/"+file+" ./wav/"+new_file_name)
		# file_list = os.listdir("")
		# with open("txt.done.data") as f, open("txt.done.data.trunc","w+") as outf:
		pass

	elif mode=="gen_pos_file":
		macro_prosody = args.macro_prosody
		out_file = args.out_file
		data = []
		with open(macro_prosody) as f:
			lines = f.readlines()
			row = 0
			while row<len(lines):
				data_name = re.split(r'-|\s',lines[row].strip())[-1]
				data_name_len = len(data_name)
				for i in range(5-data_name_len):
					data_name = "0"+data_name
				data_name = "data_"+data_name
				row += 1
				token_l = []
				token_len = 0
				while row<len(lines) and "---" not in lines[row]:
					token = lines[row].split(" ")[0]
					# if len(token_l)==0 or token!=token_l[-1]:
					# 	token = string.replace(token,"『","")
					# 	token = string.replace(token,"』","")
					# 	token = string.replace(token,"﹖","")
					# 	token = string.replace(token,"？","")
					# 	token = string.replace(token,"卅","卅啊")
					# 	token = string.replace(token,"𧙗","佑")

					# 	token_l.append(token)
					if token_len == 0:
						token = string.replace(token,"『","")
						token = string.replace(token,"』","")
						token = string.replace(token,"﹖","")
						token = string.replace(token,"？","")
						token = string.replace(token,"𧙗","佑")
						token = string.replace(token,"卅","卅啊")
						token_l.append(token)
						token_len += 1
					else:
						token = string.replace(token,"『","")
						token = string.replace(token,"』","")
						token = string.replace(token,"﹖","")
						token = string.replace(token,"？","")
						token = string.replace(token,"𧙗","佑")
						token = string.replace(token,"卅","卅啊")
						token_len += 1
					if token_len==len(token.decode("utf-8")):
						token_len = 0
					row += 1
				token_l.append(".")
				data.append([data_name,token_l])

		with open(out_file,"w+") as f:
			for data_name,token_l in data:
				f.write(data_name+"\n")
				f.write(" ".join(token_l)+"\n")
				f.write(" ".join(["NN" for i in range(len(token_l))])+"\n")
				f.write("\n")

	elif mode=="extract_phrase":
		align_file = args.alignment_file
		out_dir = args.out_dir
		with open(align_file) as f:
			lines = f.readlines()
			row = 1
			data_list = []
			while row<len(lines):
				data_name = re.split(r'\.|\/',lines[row])[-2]
				for i in range(5-len(data_name)):
					data_name = "0"+data_name
				data_name = "data_"+data_name
				# data_list.append(data_name)
				row += 1
				phrase_list = []
				phrase = []
				while row<len(lines) and lines[row].strip()!=".":
					if "sil" in lines[row]:
						phrase_list.append(phrase)
						phrase = []
					elif len(lines[row].strip().split(" "))==5:
						tup = lines[row].strip().split(" ")
						syllable = tup[-1]
						phrase.append(syllable)
					elif "sp" in lines[row]:
						phrase_list.append(phrase)
						phrase = []
					row += 1
				phrase_list.append(phrase)
				phrase_list = [phrase for phrase in phrase_list if len(phrase)!=0]
				row += 1
				data_list.append([data_name,phrase_list])
		
		os.system("mkdir "+out_dir)
		for data_name,phrase_list in data_list:
			with open(out_dir+"/"+data_name,"w+") as f:
				for phrase in phrase_list:
					f.write(" ".join(phrase)+"\n")

		# for data_name,phrase_list in data_list:
		# 	l = []
		# 	with open("./my_can_data/syllable_in_file/"+data_name) as f:
		# 		l = [tmp.strip() for tmp in f.readlines()]
		# 	l2 = []
		# 	for phrase in phrase_list:
		# 		l2 += phrase
		# 	if len(l)!=len(l2):
		# 		print(data_name)
		# 		print(len(l))
		# 		print(len(l2))
		# 		for i in range(len(l2)):
		# 			if l2[i] not in l[i]:
		# 				print(l2[i])
		# 				print(i)
		# 				break




				













		