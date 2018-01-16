import os
import string
from sets import Set
import sys


if __name__=="__main__":
	if sys.argv[1]=="all_version":
		utt_dir = "./casia_data/prompt-utt-pyt/"
		word_txt = "./casia_data/txt.done.data-all"
		syl_txt = "txt.done.data"
		dic = {}
		with open(word_txt) as wf, open(syl_txt,"w+") as sf:
			for line in wf:
				line = line.strip().split(" ")
				data_name = line[1]
				text = line[2].decode("utf-8")[1:-2]
				# print(text)
				utt_name = data_name+".utt"
				syl_list = []
				with open(utt_dir+"/"+utt_name) as utt:
					for line in utt:
						if "dur_factor" not in line:
							continue
						else:
							syl = line.split(" ")[5]
							if syl=="pau" or syl=="ssil":
								continue
							else:
								if "ve" in syl:
									syl = syl.replace("ve","ue")
								syl_list.append(syl)
				assert len(syl_list)==len(text)
				for i in range(len(syl_list)):
					word = text[i].encode("utf-8")
					if word not in dic:
						dic[word] = {}
						dic[word][syl_list[i]] = 0
					else:
						if syl_list[i] not in dic[word]:
							dic[word][syl_list[i]] = 0
					dic[word][syl_list[i]] += 1
				sf.write(" ".join(["(","data_"+data_name.split("_")[1],"\""]+syl_list+["\"",")"])+"\n")
		# for word, pro_l in dic.items():
		# 	print(word),
		# 	for pro,count in pro_l.items():
		# 		print(pro+":"+str(count)),
		# 	print("")
		cons = ["b", "d", "t", "g", "j", "k", "p", "q", "f", "h", "sh", "s", "ch", "c", "x", \
			"zh", "z", "m", "n", "l", "r", "w", "y"]
		# vowel = Set([])
		#(lex.add.entry '("bat1" nn (((b a t) 1))))#
		with open("cmu_yue_addenda.scm","w+") as af:
			for word, pro_l in dic.items():
				for pro,count in pro_l.items():
					for i in range(len(cons)):
						tmp = cons[i]
						if pro.startswith(tmp):
							tmp_vowel = pro[len(tmp):-1]
							af.write("(lex.add.entry '(\""+pro+"\" nn ((("+tmp+" "+tmp_vowel+") "+pro[-1]+"))))\n")
							break
						if i==len(cons)-1:
							af.write("(lex.add.entry '(\""+pro+"\" nn ((("+pro[0:-1]+") "+pro[-1]+"))))\n")
	elif sys.argv[1]=="small_version":
		all_syl_txt = "./txt.done.data"
		small_word_txt = "./word_txt.done.data_small"
		small_syl_txt = "./syl_txt.done.data_small"
		dic = {}
		with open(all_syl_txt) as f:
			for line in f:
				line = line.strip()
				data_id = line.split(" ")[1].split("_")[1]
				dic[data_id] = line
		with open(small_word_txt) as wt, open(small_syl_txt,"w+") as st:
			for line in wt:
				data_id = line.split(" ")[1].split("_")[1]
				st.write(dic[data_id]+"\n")







