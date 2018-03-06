import os
import re
os.environ["FESTVOXDIR"] = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/festvox"
os.environ["ESTDIR"] = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools"
os.environ["SPTKDIR"] = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/SPTK"
# export FESTVOXDIR=/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/festvox
# export ESTDIR=/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools
# export SPTKDIR=/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/SPTK
# FESTVOXDIR="/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/festvox"
def gen_wav(in_file,out_file):
	os.system("/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools/bin/ch_track"+
		" "+in_file+" -itype ascii -otype est_binary -s 0.005 -o x1.mcep")
	os.system("/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools/../festival/bin/festival"+
		" -b '(begin(wave.save (mlsa_resynthesis (track.load \"x1.mcep\") nil nil) \"'"+out_file+"'\"))'")
def main(voice_dir,odir):
	os.system("mkdir "+odir)
	os.system("mkdir "+odir+"/utt")
	os.system("mkdir "+odir+"/mcep")
	os.system("mkdir "+odir+"/wav")
	test_file = voice_dir+"/etc/txt.done.data.test"
	with open(test_file) as f:
		for line in f:
			with open(odir+"/tmp_test_file","w+") as tmpf:
				tmpf.write(line)
			os.system("$FESTVOXDIR/src/clustergen/cg_test tts tts_base "+odir+"/tmp_test_file")
			data_name = ""
			with open("utt.utt") as uttf:
				lines = uttf.readlines()
				data_name = "data_"+re.findall(r"\D(\d{5})\D",lines[4])[0]
			print(data_name)
			os.system("cp utt.utt "+odir+"/utt/"+data_name+".utt")
			with open("param.track") as trackf, open(odir+"/mcep/"+data_name+".mcep","w+") as mcepf:
				lines = trackf.readlines()[34:]
				lines = [re.split(r"\s|\\t",line)[3:] for line in lines]
				lines = [" ".join([ss for ss in line if ss!=""])+"\n" for line in lines]
				mcepf.writelines(lines)
			# os.system("cp param.track "+odir+"/mcep/"+data_name+".mcep")
			gen_wav(odir+"/mcep/"+data_name+".mcep",odir+"/wav/"+data_name+".wav")
			

if __name__=="__main__":
	voice_dir = "/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/gen_audio/cmu_yue_wdy_normal_build"
	odir = "my_dump"
	main(voice_dir,odir)

