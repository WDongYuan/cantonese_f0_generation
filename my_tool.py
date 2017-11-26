import os
import sys
import random
def TxtDoneData(path):
	dic = {}
	with open(path) as f:
		for line in f:
			arr = line.strip().split(" ")
			data_name = arr[1]
			syllable_list = arr[3:-2]
			dic[data_name] = syllable_list
	return dic
class MyDir():
	def __init__(self,path):
		os.system("mkdir "+path)
		self.dir = path
		self.file_list = []
		self.token_list = []

	def add_file(self,file_name):
		os.system("mkdir "+self.dir+"/"+file_name)
		self.file_list.append(file_name)
		return self.dir+"/"+file_name

	def remove_file(self,file_name):
		os.system("rm "+self.dir+"/"+file_name)

	def read_file_token(self,file_name):
		file_token = open(self.dir+"/"+file_name)
		self.token.append(file_token)
		return file_token

	def write_file_token(self,file_name):
		file_token = open(self.dir+"/"+file_name,"w+")
		self.token.append(file_token)
		return file_token

	def clear_token(self):
		for file_token in self.token_list:
			file_token.close()
		self.token_list = []
		
	def destroy(self):
		self.clear_token()
		os.system("rm -r "+self.dir)

def SplitFile(ori_file,ratio,to_file_1,to_file_2):
	file_1 = open(to_file_1,"w+")
	file_2 = open(to_file_2,"w+")
	with open(ori_file) as f:
		for line in f:
			if random.random()<ratio:
				file_1.write(line)
			else:
				file_2.write(line)
	file_1.close()
	file_2.close()
	return
if __name__=="__main__":
	if len(sys.argv)==1:
		print("python my_tool.py split_file origin_file ratio_1 ratio_2 to_file1 to_file2")
		exit()
		
	tool = sys.argv[1]
	if tool=="split_file":
		ori_file = sys.argv[2]
		ratio_1 = float(sys.argv[3])
		ratio_2 = float(sys.argv[4])
		to_file_1 = sys.argv[5]
		to_file_2 = sys.argv[6]
		SplitFile(ori_file,ratio_1,to_file_1,to_file_2)