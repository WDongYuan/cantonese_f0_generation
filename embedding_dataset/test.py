with open("wiki.yue/wiki.zh_yue.vec") as inf, open("vocab","w+") as outf:
	for line in inf:
		token = line.split(" ")[0]
		if len(token.decode("utf-8"))==1:
			outf.write(line)