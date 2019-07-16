import pickle
import os
from loaddataset import *

PKL_DIRECTORY = 'temp'
from fileutils import assureDir
# Create directiory
assureDir(PKL_DIRECTORY)

def save(obj, file):
	if not os.path.exists(os.path.join(PKL_DIRECTORY, os.path.dirname(file))):
		os.makedirs(os.path.join(PKL_DIRECTORY, os.path.dirname(file)))
	with open(os.path.join(PKL_DIRECTORY, file), 'wb') as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load(file):
	if not os.path.isfile(os.path.join(PKL_DIRECTORY, file)):
		return None
	with open(os.path.join(PKL_DIRECTORY, file), 'rb') as input:
		obj = pickle.load(input)
		return obj

def loadPython2(file):
	if not os.path.isfile(file):
		return None
	with open(file, 'rb') as input:
		u = pickle._Unpickler(input)
		u.encoding = 'latin1'
		p = u.load()
		return p

def remove(file):
	if os.path.isfile(os.path.join(PKL_DIRECTORY, file)):
		os.remove(os.path.join(PKL_DIRECTORY, file))

def isPersisted(pkl):
	return os.path.isfile(os.path.join(PKL_DIRECTORY, pkl))

def persistSingleDataset(origin, pkl, overwrite=False):
	if overwrite or load(pkl) is None:
		dataset = DatasetLoader(origin).dataset
		print('--- Persisting',pkl,'---')
		save(dataset, pkl)
		return dataset
	return load(pkl)

def persistSinglePklDataset(origin, pkl, overwrite=False):
	if overwrite or load(pkl) is None:
		dataset = PklDatasetLoader(origin).dataset
		print('--- Persisting',pkl,'---')
		save(dataset, pkl)
		return dataset
	return load(pkl)

# with open('/home/inspiros/Desktop/CoHai/CMDFALL Dataset/minhkv/depth/depth_dmm_cnn_acc_30_fea_train.pkl', 'rb') as input:
# 	p = pickle.load(input)

# with open('/home/inspiros/Desktop/CoHai/CMDFALL Dataset/minhkv/depth/depth_dmm_cnn_acc_30_fea.pkl', 'rb') as input:
# 	t = pickle.load(input)

# p.update(t)

# with open('/home/inspiros/Desktop/CoHai/CMDFALL Dataset/minhkv/depth/depth.pkl', 'wb') as output:
# 	pickle.dump(p, output, pickle.HIGHEST_PROTOCOL)
