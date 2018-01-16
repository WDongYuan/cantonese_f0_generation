import os
import sys
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

if __name__=="__main__":
	if len(sys.argv)==1:
		print("python apply_f0.py ./cmu_yue_wdy_normal_build_new(voice_directory) wdy_tmp(out_dir)")
		exit()
	# ori_dir = os.getcwd()
	os.environ["FESTVOXDIR"] = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/festvox"
	os.environ["SPTKDIR"] = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/SPTK"
	os.environ["ESTDIR"] = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools"
	# voice_dir = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/gen_audio/cmu_yue_wdy_normal_build_new"
	voice_dir = sys.argv[1]
	out_dir = sys.argv[2]
	tmp_dir = os.path.abspath(out_dir)
	os.chdir(voice_dir)

	# os.system("mkdir "+ori_dir+"/apply_f0_wav")
	os.system("mkdir "+tmp_dir)
	os.system("mkdir "+tmp_dir+"/ccoefs")
	os.system("mkdir "+tmp_dir+"/f0_value")
	with open("etc/txt.done.data.test") as f:
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


			# os.system("./my_synth_f0 ../src/test_f0_in_system/voice_lib/0/f0_val/"+data_name+".f0 param.track "+ori_dir+"/apply_f0_wav")
	# os.system("head -3 etc/txt.done.data.test >3.data")
	# os.system("mv utt.utt FILEID1.utt")
	# os.system("mv param.track FILEID1.mcep")
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