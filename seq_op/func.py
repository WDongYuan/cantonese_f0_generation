import os
import numpy as np
	
def get_pos_dic(pos_file):
	pos_dic = {}
	with open(pos_file) as f:
		cont = f.readlines()
		for i in range(len(cont)):
			if i%4!=2:
				continue
			pos_l = cont[i].strip().split(" ")
			pos_l.pop(-1)
			for pos in pos_l:
				if pos not in pos_dic:
					pos_dic[pos] = len(pos_dic)+1
	return pos_dic

def get_word_dic(word_file,vocab_size):
	w_dic = {}
	with open(word_file) as f:
		cont = f.readlines()
		for i in range(len(cont)):
			if i%4!=1:
				continue
			w_l = cont[i].strip().split(" ")
			w_l.pop(-1)
			w_l = "".join(w_l).decode("utf-8")
			for w in w_l:
				w = w.encode("utf-8")
				if w not in w_dic:
					w_dic[w] = 0
				w_dic[w] += 1
	w_list = [[w,count] for w,count in w_dic.items()]
	w_list = sorted(w_list,key=lambda tup:tup[1],reverse=True)
	w_list = [tup[0] for tup in w_list]
	w_list = w_list[0:vocab_size]
	w_list.append("unk")
	w_dic = {}
	for w in w_list:
		w_dic[w] = len(w_dic)+1
	return w_dic

def append_pos_to_feature(feat_dir,pos_file,pos_dic,word_dic):
	##read pos tag
	data_dic = {}

	with open(pos_file) as f:
		sents = f.readlines()
		row = 0
		while row < len(sents):
			data_name = sents[row].strip()
			row += 1
			token = sents[row].strip().split(" ")
			row += 1
			pos = sents[row].strip().split(" ")
			row += 1
			row += 1
			token.pop(-1)##remove punctuation
			pos.pop(-1)##remove punctuation
			pos_list = []
			assert len(token)==len(pos)
			for i in range(len(token)):
				for j in range(len(token[i].decode("utf-8"))):
					one_word = token[i].decode("utf-8")[j]

					tmp_feat = []

					##current pos
					tmp_feat.append(str(pos_dic[pos[i]]))

					##previous pos
					if i==0:
						tmp_feat.append("0")
					else:
						tmp_feat.append(str(pos_dic[pos[i-1]]))

					##next pos
					if i==len(token)-1:
						tmp_feat.append("0")
					else:
						tmp_feat.append(str(pos_dic[pos[i+1]]))

					##pos tag position in utterance
					tmp_feat.append(str(i))

					##word position in pos tag
					tmp_feat.append(str(j))

					##chinese character(word) index
					if one_word.encode("utf-8") not in word_dic:
						tmp_feat.append(str(word_dic["unk"]))
					else:
						tmp_feat.append(str(word_dic[one_word.encode("utf-8")]))

					pos_list.append(tmp_feat)
					# pos_list.append([pos[i],i,j])##i=pos tag postion in utterance, j=word postion in its pos tag
			data_dic[data_name] = pos_list

	file_list = os.listdir(feat_dir)
	file_list = [tmp_name for tmp_name in file_list if "data" in tmp_name]

	feature_before = 0
	with open(feat_dir+"/"+file_list[0]) as f:
		feature_before = len(f.readline().strip().split(" "))

	for file_name in file_list:
		pos = data_dic[file_name.split(".")[0]]
		file_sents = None
		with open(feat_dir+"/"+file_name) as f:
			file_sents = f.readlines()
		# print(pos)
		# print(file_sents)
		assert len(file_sents)==len(pos),file_name
		# file_sents = [file_sents[i].strip()+" "+str(pos_dic[pos[i][0]])+" "+str(pos[i][1])+" "+str(pos[i][2])+"\n" for i in range(len(file_sents))]
		file_sents = [file_sents[i].strip()+" "+" ".join(pos[i])+"\n" for i in range(len(file_sents))]
		with open(feat_dir+"/"+file_name,"w+") as f:
			f.writelines(file_sents)

	# print("append 5 pos features: pos tag, pre pos tag, next pos tag, pos position in utterance, word position in pos")
	print("pos features "+str(feature_before)+" "+str(feature_before+6-1))
	return


def get_syl_dic(consonant_vowel_file):
	c_dic = {}
	v_dic = {}
	with open(consonant_vowel_file) as f:
		line = f.readline().strip().split(" ")
		for c in line:
			c_dic[c] = len(c_dic)

		line = f.readline().strip().split(" ")
		for v in line:
			v_dic[v] = len(v_dic)
	return c_dic,v_dic

def decompose_zh_syl(syl_l,c_dic,v_dic):
	result = []
	for syl in syl_l:
		if syl in v_dic:
			result.append([0,v_dic[syl]])
		else:
			p = 0
			while syl[0:p+1] in c_dic:
				p += 1
			# print(syl[0:p]+" "+syl[p:])
			result.append([c_dic[syl[0:p]],v_dic[syl[p:]]])
	result = np.array(result).astype(np.int32)
	return result

def append_syl_to_feature(feat_dir,txt_done_data,c_dic,v_dic):
	data = {}
	with open(txt_done_data) as f:
		for line in f:
			line = line.strip().split(" ")
			data_name = line[1]
			syl_list = line[3:-2]
			data[data_name] = []
			for syl in syl_list:
				data[data_name].append(syl[0:-1])
	file_list = os.listdir(feat_dir)
	file_list = [file for file in file_list if "data" in file]

	feature_before = 0
	with open(feat_dir+"/"+file_list[0]) as f:
		feature_before = len(f.readline().strip().split(" "))

	for data_name in file_list:
		if "data" not in data_name:
			continue
		syl_l = data[data_name.split(".")[0]]
		cvl = decompose_zh_syl(syl_l,c_dic,v_dic)
		feat_cont = None
		with open(feat_dir+"/"+data_name) as f:
			feat_cont = f.readlines()
		feat_cont = [line.strip() for line in feat_cont]
		assert len(feat_cont)==len(cvl)
		with open(feat_dir+"/"+data_name,"w+") as f:
			for i in range(len(feat_cont)):
				feat_cont[i] = feat_cont[i]+" "+str(cvl[i][0])+" "+str(cvl[i][1])+"\n"
			f.writelines(feat_cont)
	print("syllable features "+str(feature_before)+" "+str(feature_before+2-1))


def append_phrase_to_feature(feat_dir,phrase_syl_dir):
	file_list = os.listdir(feat_dir)
	file_list = [file for file in file_list if "data" in file]

	feature_before = 0
	with open(feat_dir+"/"+file_list[0]) as f:
		feature_before = len(f.readline().strip().split(" "))

	for file in file_list:
		if "data" not in file:
			continue
		with open(phrase_syl_dir+"/"+file.split(".")[0]) as f:
			utt = f.readlines()
			for i in range(len(utt)):
				utt[i] = utt[i].strip().split(" ")

			phrase_feat = []
			for i in range(len(utt)):
				phrase = utt[i]
				for j in range(len(phrase)):
					word_feat = []
					#phrase position in utt
					word_feat.append(i)

					#phrase percent in utt
					word_feat.append(float(i)/len(utt))

					#phrase number in utt
					word_feat.append(len(utt))

					#syllable position in phrase
					word_feat.append(j)

					#syllable percent in phrase
					word_feat.append(float(j)/len(phrase))

					#syllable number in phrase
					word_feat.append(len(phrase))

					phrase_feat.append(word_feat)
			phrase_feat = np.array(phrase_feat)

			ori_feat = None
			with open(feat_dir+"/"+file) as featf:
				ori_feat = featf.readlines()
				assert len(phrase_feat)==len(ori_feat),file+": "+str(len(phrase_feat))+" doesn't equal to "+str(len(ori_feat))
			with open(feat_dir+"/"+file,"w+") as outf:
				for i in range(len(ori_feat)):
					outf.write(ori_feat[i].strip()+" "+" ".join(phrase_feat[i].astype(np.str).tolist())+"\n")
	# print("append 6 phrase features")
	print("phrase features "+str(feature_before)+" "+str(feature_before+6-1))

def load_dic(file):
	dic = {}
	with open(file) as f:
		for line in f:
			line = line.strip().split(" ")
			dic[line[0]] = int(line[1])
	return dic

def save_dic(dic,out_file):
	with open(out_file,"w+") as f:
		for key,idx in dic.items():
			f.write(str(key)+" "+str(idx)+"\n")