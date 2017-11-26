import os
import sys
if __name__=="__main__":
	out_dir = sys.argv[1]
	file_list = os.listdir("./"+out_dir)
	print("Generating wav files...")
	for file in file_list:
		if "data" in file:
			os.system("./bin/synth_f0 "+out_dir+"/"+file)
	print("Done.")