import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.fftpack import idct, dct

if __name__=="__main__":
	arr = [280.979,308.489,263.422,251.818,256.948,187.620,209.607,273.195,286.956,295.195,168.038]
	tmp = []
	for val in arr:
		tmp += [val]*10
	arr = tmp
	a = dct(arr)
	a[3:] = 0
	b = idct(a)/(2*len(arr))
	plt.plot(arr,label="original")
	plt.plot(b,label="idct")
	plt.legend()
	plt.show()
