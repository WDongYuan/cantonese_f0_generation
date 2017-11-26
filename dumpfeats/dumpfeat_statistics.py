


feature_dic = {}
if __name__=="__main__":
	feature_map = []
	with open("./accent.feats") as f:
		for line in f:
			line = line.strip()
			feature_dic[line] = {}
			feature_map.append(line)

	with open("./accent.data") as f:
		for line in f:
			arr = line.strip().split(" ")
			for i in range(len(arr)):
				tmp_feature = feature_map[i]
				if arr[i] not in feature_dic[tmp_feature]:
					feature_dic[tmp_feature][arr[i]] = 0
				feature_dic[tmp_feature][arr[i]] += 1
	for feature in feature_map:
		print(feature)
		print(feature_dic[feature])
		print("###############################")
