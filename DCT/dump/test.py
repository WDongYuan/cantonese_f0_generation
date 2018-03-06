import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.fftpack import idct, dct
import numpy as np

if __name__=="__main__":
	# arr = [280.979,308.489,263.422,251.818,256.948,187.620,209.607,273.195,286.956,295.195,168.038]
	# tmp = []
	# for val in arr:
	# 	tmp += [val]*10
	# arr = tmp
	# a = dct(arr)
	# a[3:] = 0
	# b = idct(a)/(2*len(arr))
	# plt.plot(arr,label="original")
	# plt.plot(b,label="idct")
	# plt.legend()
	# plt.show()

	data = np.loadtxt("/Users/weidong/GoogleDrive/CMU/NLP/Can2Ch_Speech/my_festival/gen_audio/src/mandarine/dt_data_dir/train_test_data/train_data/train_f0",delimiter=" ")
	dct_data = dct(data,axis=1)
	dct_data[:,5:] = 0
	idct_data = idct(dct_data,axis=1)/(2*data.shape[1])
	print(np.sqrt(np.square(data-idct_data).mean(axis=1)).mean())
