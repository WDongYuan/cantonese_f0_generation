import argparse
import os

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', dest='mode')
	parser.add_argument('--awb_synth', dest='awb_synth')
	parser.add_argument('--f0_dir', dest='f0_dir')
	parser.add_argument('--ccoef_dir', dest='ccoef_dir')
	parser.add_argument('--out_dir', dest='out_dir')
	args = parser.parse_args()
	mode = args.mode

	if mode=="how_to_run":
		print("python run.py --mode synthesis_with_f0"+
			" --awb_synth ./my_synth_f0"+
			" --ccoef_dir ../../cmu_yue_wdy_addf0/ccoefs"+
			" --f0_dir ./voice_lib/0/f0_val"+
			" --out_dir ./voice_lib/0/wav")

	if mode=="synthesis_with_f0":
		f0_dir = args.f0_dir
		ccoef_dir = args.ccoef_dir
		awb_synth = args.awb_synth
		out_dir = args.out_dir
		f0_file_l = os.listdir(f0_dir)
		os.system("rm -r "+out_dir)
		os.system("mkdir "+out_dir)
		for file in f0_file_l:
			if "data" not in file:
				continue
			# print(file)
			os.system(awb_synth+" "+f0_dir+"/"+file+" "+ccoef_dir+" "+out_dir)