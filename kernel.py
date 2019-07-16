
class Kernel:
	'''
	Class for encapsulating kernel function.
	'''
	def __init__(self, func, func_name=None, name=None):
		self.kernel = func
		self.func_name = func_name
		self.name = name

	def tostring(self, streams=3):
		if type(self.name) is type(list()):
			return '{' + self.name[streams - 2] + ':' + self.func_name + '}'
		return '{' + self.name + ':' + self.func_name + '}'

	def stringigfy(kernels, streams=3):
		string = '{'
		for i in range(len(kernels)):
			if type(kernels[i].name) is type(list()):
				string += kernels[i].name[streams - 2] + ':' + kernels[i].func_name
			else:
				string += kernels[i].name + ':' + kernels[i].func_name
			if i < len(kernels) - 1:
				string += ', '
			else:
				string += '}'
		return string

	def apply(self, X, L=None):
		return self.kernel(X, L)
