import numpy as np

def retainIf(npArr, predicate):
	newNpArr = np.array([])
	for i in range(0, npArr.shape[0]):
		if predicate(i):
			if newNpArr.shape == np.array([]).shape:
				newNpArr = np.array([npArr[i]])
			else:
				newNpArr = np.concatenate((newNpArr, np.array([npArr[i]])), axis = 0)
	return newNpArr

def divide(npArr, div):
	def predicate(index):
		nonlocal div
		if index % div == 0:
			return True
		return False
	return retainIf(npArr, predicate)