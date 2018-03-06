import matplotlib
matplotlib.use('TkAgg')
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stanfordcorenlp import StanfordCoreNLP
def parse_txt_file_pos(txt_file,out_file):
	parser = StanfordCoreNLP(r'/Users/weidong/Downloads/stanford-corenlp-full-2016-10-31',lang='zh')
	with open(txt_file) as txtf, open(out_file,"w+") as outf:
		for line in txtf:
			line = line.strip().split(" ")
			data_name = line[1]
			text = line[2].decode("utf-8")[1:-1].encode("utf-8")
			tokens = parser.word_tokenize(text)
			pos = parser.pos_tag(text)
			outf.write(data_name+"\n")
			outf.write(" ".join(tokens).encode("utf-8")+"\n")
			outf.write(" ".join([tup[1] for tup in pos]).encode("utf-8")+"\n")
			outf.write("\n")
	return

def pos_refine(convert_map,pos_file,refine_pos_file):
	pos_map = {}
	with open(convert_map) as f:
		for line in f:
			line = line.strip().split(" ")
			pos_map[line[0]] = line[1]
	with open(pos_file) as old_f, open(refine_pos_file,"w+") as new_f:
		old_lines = old_f.readlines()
		for i in range(len(old_lines)):
			if i%4!=2:
				new_f.write(old_lines[i])
			else:
				line = old_lines[i].strip().split(" ")
				line = [pos_map[pos] for pos in line]
				new_f.write(" ".join(line)+"\n")
	return

def generate_f0_tag(ph_file,out_file):
	def my_round(num):
		return num/5*5 if num%5<3 else (num/5+1)*5
	timeline = []
	with open(ph_file) as f:
		for line in f:
			line = line.strip().split(" ")
			# print(line)
			ph_end = int(float(line[0])*1000)
			ph_end = my_round(ph_end)
			ph = line[1]
			timeline.append([ph,ph_end])
	# print(timeline)
	count = 0
	with open(out_file,"w+") as f:
		f.write(" ")
		for tup in timeline:
			ph = tup[0]
			end = tup[1]
			while count<end:
				if ph=="pau" or ph=="ssil":
					f.write("0.0 1.0\n")
				else:
					f.write("100.0 1.0\n")
				count += 5
		f.write("-1.0 1.0\n")
	return

def syllable_time_series(ph_file,out_file):
	##return the time series for the phoneme in the file.
	##every phoneme is represent by [phone,begin_idx,end_idx](index puls one means plus 5 ms)(for the consistent with the f0 file)
	def my_round(num):
		return num/5*5 if num%5<3 else (num/5+1)*5
	timeline = []
	pre_end = 0
	with open(ph_file) as f:
		for line in f:
			line = line.strip().split(" ")
			# print(line)
			ph_end = int(float(line[0])*1000)
			ph_end = my_round(ph_end)
			ph = line[1]
			timeline.append([ph,pre_end/5+1,ph_end/5])
			pre_end=ph_end
	return timeline


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--voice_dir', dest='voice_dir')
	parser.add_argument('--out_dir',dest='out_dir')
	parser.add_argument('--test_txt',dest='test_txt')
	parser.add_argument('--mode',dest='mode')
	parser.add_argument('--modified_clustergen_scm',dest='modified_clustergen_scm')
	parser.add_argument('--txt_file',dest='txt_file')
	parser.add_argument('--out_file',dest='out_file')
	parser.add_argument('--pos_convert_map',dest='pos_convert_map')
	args = parser.parse_args()
	mode = args.mode
	if mode=="how_to_run":
		print("python apply_f0.py"+
			" --mode run"+
			" --voice_dir ./cmu_yue_wdy_normal_build_new(voice_directory)"+
			" --test_txt ./etc/txt.done.data.test"+
			" --modified_clustergen_scm ./clustergen.scm"+
			" --out_dir wdy_tmp(out_dir)")
		print("python apply_f0.py"+
			" --mode tune_timeline"+
			" --voice_dir ./cmu_yue_wdy_normal_build_new(voice_directory)"+
			" --test_txt ./etc/txt.done.data.test"+
			" --modified_clustergen_scm ./clustergen.scm"+
			" --out_dir wdy_tmp(out_dir)")
		print("python apply_f0.py"+
			" --mode parse_txt"+
			" --txt_file"+
			" --pos_convert_map"
			" --out_file")
		exit()
	elif mode=="run":
	# ori_dir = os.getcwd()
		os.environ["FESTVOXDIR"] = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/festvox"
		os.environ["SPTKDIR"] = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/SPTK"
		os.environ["ESTDIR"] = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools"
		# voice_dir = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/gen_audio/cmu_yue_wdy_normal_build_new"
		voice_dir = args.voice_dir
		out_dir = args.out_dir
		tmp_dir = os.path.abspath(out_dir)
		test_txt = args.test_txt
		test_txt = os.path.abspath(test_txt)
		modified_clustergen_scm = os.path.abspath(args.modified_clustergen_scm)


		os.chdir(voice_dir)
		#modify the clustergen.scm and save the intermeddiate result
		os.system("mv ./festvox/clustergen.scm ./festvox/clustergen_original.scm")
		os.system("cp "+modified_clustergen_scm+" ./festvox")

		# os.system("mkdir "+ori_dir+"/apply_f0_wav")
		os.system("mkdir "+tmp_dir)
		os.system("mkdir "+tmp_dir+"/ccoefs")
		os.system("mkdir "+tmp_dir+"/f0_value")
		os.system("mkdir "+tmp_dir+"/festival")
		os.system("mkdir "+tmp_dir+"/festvox")
		os.system("mkdir "+tmp_dir+"/festival/utts")
		with open(test_txt) as f:
			for line in f:
				data_name = line.split(" ")[1]
				print(data_name)
				with open("line.data","w+") as line_file:
					line_file.write(line)
				os.system("$FESTVOXDIR/src/clustergen/cg_test tts aaa line.data")
				with open("utt.utt") as utt_file,open(tmp_dir+"/f0_value/"+data_name+".phoneme","w+") as ph_file:
					for line in utt_file:
						line = line.strip()
						if "end" in line:
							line = line.split(" ")
							if len(line)==10:
								ph_file.write(line[8]+" "+line[5]+"\n")

				generate_f0_tag(tmp_dir+"/f0_value/"+data_name+".phoneme",tmp_dir+"/f0_value/"+data_name+".f0")
				os.system("mv param.track "+tmp_dir+"/ccoefs/"+data_name+".mcep")
				os.system("cp utt.utt "+tmp_dir+"/festival/utts/"+data_name+".utt")


		os.system("mv ./festvox/clustergen_original.scm ./festvox/clustergen.scm")
		os.system("cp ./festvox/clustergen.scm "+tmp_dir+"/festvox")


				# os.system("./my_synth_f0 ../src/test_f0_in_system/voice_lib/0/f0_val/"+data_name+".f0 param.track "+ori_dir+"/apply_f0_wav")
		# os.system("head -3 etc/txt.done.data.test >3.data")
		# os.system("mv utt.utt FILEID1.utt")
		# os.system("mv param.track FILEID1.mcep")
	elif mode=="tune_timeline":
		os.environ["FESTVOXDIR"] = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/festvox"
		os.environ["SPTKDIR"] = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/SPTK"
		os.environ["ESTDIR"] = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools"
		# voice_dir = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/gen_audio/cmu_yue_wdy_normal_build_new"
		voice_dir = args.voice_dir
		out_dir = args.out_dir
		tmp_dir = os.path.abspath(out_dir)
		test_txt = args.test_txt
		test_txt = os.path.abspath(test_txt)
		modified_clustergen_scm = os.path.abspath(args.modified_clustergen_scm)


		os.chdir(voice_dir)
		#modify the clustergen.scm and save the intermeddiate result
		os.system("mv ./festvox/clustergen.scm ./festvox/clustergen_original.scm")
		os.system("cp "+modified_clustergen_scm+" ./festvox")

		# os.system("mkdir "+ori_dir+"/apply_f0_wav")
		os.system("mkdir "+tmp_dir)
		os.system("mkdir "+tmp_dir+"/ccoefs")
		os.system("mkdir "+tmp_dir+"/f0_value")
		os.system("mkdir "+tmp_dir+"/gen_wav")
		tmp_gen_voice_dir = "aaa"
		with open(test_txt) as f:
			for line in f:
				data_name = line.split(" ")[1]
				print(data_name)
				with open("line.data","w+") as line_file:
					line_file.write(line)
				os.system("$FESTVOXDIR/src/clustergen/cg_test tts "+tmp_gen_voice_dir+" line.data")
				with open("utt.utt") as utt_file,open(tmp_dir+"/f0_value/"+data_name+".phoneme","w+") as ph_file:
					for line in utt_file:
						line = line.strip()
						if "end" in line:
							line = line.split(" ")
							if len(line)==10:
								ph_file.write(line[8]+" "+line[5]+"\n")
				syl_timeline = syllable_time_series(tmp_dir+"/f0_value/"+data_name+".phoneme",tmp_dir+"/f0_value/"+data_name+".f0")
				os.system("mv param.track "+tmp_dir+"/ccoefs/"+data_name+".mcep")
				os.system("/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools/bin/ch_track "+
					tmp_dir+"/ccoefs/"+data_name+".mcep >x1.out")
				with open("x1.out") as mcepf:
					data_mcep = []
					for line in mcepf:
						line = line.strip().split(" ")
						line = [ss for ss in line if ss!=""]
						data_mcep.append(line)
					# print(data_mcep[0])
					data_mcep = np.array(data_mcep).astype(np.float)
				syl_timeline[-1][2] = data_mcep.shape[1]
				#################################################
				## Modify the f0 value here
				# plt.plot(data_mcep[:,0][syl_timeline[17][1]-1:syl_timeline[20][2]],label=data_name)
				# print(data_mcep[:,0][syl_timeline[17][1]-1:syl_timeline[20][2]])
				for tup in syl_timeline:
					print(tup[0])
					print(data_mcep[:,0][tup[1]-1:tup[2]])
				#################################################
				np.savetxt("x1.out",data_mcep,delimiter=" ",fmt="%.5f")
				tmp_mcep = "tmp.mcep"
				os.system("/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools/bin/ch_track "+
					" x1.out -itype ascii -otype est_binary -s 0.005 -o "+tmp_mcep)
				os.system("/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools/../festival/bin/festival"+
					" -b '(begin(wave.save (mlsa_resynthesis (track.load \""+tmp_mcep+"\") nil nil) \"'"+tmp_dir+"/"+data_name+".wav'\"))'")
		# plt.legend()
		# plt.show()



		os.system("mv ./festvox/clustergen_original.scm ./festvox/clustergen.scm")
	elif mode=="parse_txt":
		txt_file = args.txt_file
		out_file = args.out_file
		convert_map = args.pos_convert_map
		tmp_pos_file = "_tmp_pos"
		parse_txt_file_pos(txt_file,tmp_pos_file)
		pos_refine(convert_map,tmp_pos_file,out_file)
		os.system("rm "+tmp_pos_file)

'''
export FESTVOXDIR=/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/festvox
export SPTKDIR=/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/SPTK
export ESTDIR=/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools
head -3 etc/txt.done.data.test >3.data
../../build/festvox/src/clustergen/cg_test tts aaa 3.data
mv utt.utt FILEID1.utt
mv param.track FILEID1.mcep
./my_synth_f0 ../src/test_f0_in_system/voice_lib/0/f0_val/data_00141.f0 data_00141.mcep ./
'''