from program_config import ProgramConfig
from params import Params
from kernel import Kernel
from sklearn.metrics import pairwise

# Data
RGB_DATASET_ROOT = '/home/inspiros/Desktop/CoHai/CMDFALL Dataset/minhkv/mica_4000/bin' # modify here
OF_DATASET_ROOT = '/home/inspiros/Desktop/CoHai/CMDFALL Dataset/minhkv/merged_flow/bin' # modify here
DEPTH_DATASET_FILE = 'dataset/depth/depth.pkl'

# Annotations
ANNOTATION_FILE = 'dataset/annotation.csv'
MAPPING_FILE = 'dataset/file_mapping.txt'
LABEL_LIST_FILE = 'dataset/ActionIndex.txt'

LAYER = 'fc6'
KINECT = '3'


CONFIGS = []

'''
Linear config
'''
def linear(X, L=None):
	return pairwise.linear_kernel(X, L)

linear_params = Params(name = 'linear',
					assignable_names = ['lin', 'linear'],
					kernel_func_rgb = linear,
					kernel_func_of = linear,
					kernel_func_depth = linear,
					kernel_func_concatenate = linear,
					C_mkl = 0.001,
					C_concatenate = 0.1
					)

CONFIGS.append(linear_params.to_program_config())


'''
RBF config
'''
def rbf_rgb(X, L=None): #0.00001
	return pairwise.rbf_kernel(X, L, gamma=0.00001)
def rbf_of(X, L=None): #0.0001
	return pairwise.rbf_kernel(X, L, gamma=0.0001)
def rbf_depth(X, L=None): #0.0001
	return pairwise.rbf_kernel(X, L, gamma=0.0001)
def rbf_concatenate(X, L=None):
	return pairwise.rbf_kernel(X, L, gamma=0.0001)

rbf_params = Params(name = 'rbf',
					assignable_names = ['rbf', 'gaussian'],
					kernel_func_rgb = rbf_rgb,
					kernel_func_of = rbf_of,
					kernel_func_depth = rbf_depth,
					kernel_func_concatenate = rbf_concatenate,
					C_mkl = 10,
					C_concatenate = 1000
					)

CONFIGS.append(rbf_params.to_program_config())


'''
Laplacian config
'''
def laplacian_rgb(X, L=None): #0.0001
	return pairwise.laplacian_kernel(X, L, gamma=0.0001)
def laplacian_of(X, L=None): #0.001
	return pairwise.laplacian_kernel(X, L, gamma=0.001)
def laplacian_depth(X, L=None): #0.001
	return pairwise.laplacian_kernel(X, L, gamma=0.001)
def laplacian_concatenate(X, L=None):
	return pairwise.laplacian_kernel(X, L, gamma=0.001)

laplacian_params = Params(name = 'laplacian',
					assignable_names = ['lap', 'laplacian'],
					kernel_func_rgb = laplacian_rgb,
					kernel_func_of = laplacian_of,
					kernel_func_depth = laplacian_depth,
					kernel_func_concatenate = laplacian_concatenate,
					C_mkl = 100,
					C_concatenate = 100
					)

CONFIGS.append(laplacian_params.to_program_config())
