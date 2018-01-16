import sys
import os
if __name__=="__main__":
	if len(sys.argv)==1:
		print("python run.py ./my_feature(feature_file)"+
			" ./my_file_list(the file containing the utt file list)"+
			" ./data_feature(output dir)"+
			" ../../cmu_yue_wdy_normal_build(voice directory)")
		exit()
	feature_file = sys.argv[1]
	file_list = sys.argv[2]
	save_dir = sys.argv[3]
	voice_dir = sys.argv[4]
	os.system("mkdir "+save_dir)
	# print("export FESTVOXDIR=/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/festvox")
	# print("export ESTDIR=/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools")
	# print("export SPTKDIR=/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/SPTK")
	ESTDIR="/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/build/speech_tools"
	os.system(ESTDIR+"/../festival/examples/dumpfeats -feats "+feature_file+
		" -relation Segment -output "+save_dir+"/%s.feats -from_file "+file_list+
		" -eval "+voice_dir+"/festvox/clustergen.scm")