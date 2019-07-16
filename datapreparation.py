import datapersistant
import numpy as np
import os
from annotation import fix_labelset


def toDict(X, y, z):
	dataDict = {}
	for i in range(len(z)):
		dataDict.update({z[i]: {'data': X[i], 'label': y[i]}})
	return dataDict

def mergeDicts(dicts):
	newDicts = [{} for d in dicts]
	for key in dicts[0].keys():
		if not False in [key in d.keys() for d in dicts]:
			for i in range(len(dicts)):
				newDicts[i].update({key: {'data': dicts[i].get(key)['data'], 'label': dicts[i].get(key)['label']}})
	return newDicts

def extractFromDict(dict1, keyword):
	ret = []
	for key in dict1.keys():
		ret.append(dict1.get(key)[keyword])
	return ret


def summary_data(Xtrs, ytr, Xtes, yte):
	print('DATA SUMMARY')
	print('train')
	print('X', *(X.shape for X in Xtrs))
	print('y', ytr, np.unique(ytr).shape[0])
	print('train')
	print('X', *(X.shape for X in Xtes))
	print('y', yte, np.unique(yte).shape[0])
	print('')


def prepare_data(rgb_dataset_root, of_dataset_root, depth_dataset_file, streams=3, override=False):
	rgb_dataset = datapersistant.persistSingleDataset(rgb_dataset_root, 'rgb_data.pkl')
	of_dataset = datapersistant.persistSingleDataset(of_dataset_root, 'of_data.pkl')
	depth_dataset = datapersistant.persistSinglePklDataset(depth_dataset_file, 'depth_data.pkl')
	if not override:
		if (
				streams==2 and
				datapersistant.isPersisted(os.path.join('2 streams', 'Xtr_rgb.pkl')) and
				datapersistant.isPersisted(os.path.join('2 streams', 'Xtr_of.pkl')) and
				datapersistant.isPersisted(os.path.join('2 streams', 'ytr.pkl')) and
				datapersistant.isPersisted(os.path.join('2 streams', 'Xte_rgb.pkl')) and
				datapersistant.isPersisted(os.path.join('2 streams', 'Xte_of.pkl')) and
				datapersistant.isPersisted(os.path.join('2 streams', 'yte.pkl'))
			):
			Xtr_k1x = datapersistant.load(os.path.join('2 streams', 'Xtr_rgb.pkl'))
			Xtr_k2x = datapersistant.load(os.path.join('2 streams', 'Xtr_of.pkl'))
			ytrx = datapersistant.load(os.path.join('2 streams', 'ytr.pkl'))
			Xte_k1x = datapersistant.load(os.path.join('2 streams', 'Xte_rgb.pkl'))
			Xte_k2x = datapersistant.load(os.path.join('2 streams', 'Xte_of.pkl'))
			ytex = datapersistant.load(os.path.join('2 streams', 'yte.pkl'))

			summary_data(*([Xtr_k1x, Xtr_k2x], ytr), *([Xte_k1x, Xte_k2x], yte))
			return (([Xtr_k1x, Xtr_k2x], ytr), ([Xte_k1x, Xte_k2x], yte))
		elif (
				streams==3 and
				datapersistant.isPersisted(os.path.join('3 streams', 'Xtr_rgb.pkl')) and
				datapersistant.isPersisted(os.path.join('3 streams', 'Xtr_of.pkl')) and
				datapersistant.isPersisted(os.path.join('3 streams', 'Xtr_depth.pkl')) and
				datapersistant.isPersisted(os.path.join('3 streams', 'ytr.pkl')) and
				datapersistant.isPersisted(os.path.join('3 streams', 'Xte_rgb.pkl')) and
				datapersistant.isPersisted(os.path.join('3 streams', 'Xte_of.pkl')) and
				datapersistant.isPersisted(os.path.join('3 streams', 'Xte_depth.pkl')) and
				datapersistant.isPersisted(os.path.join('3 streams', 'yte.pkl'))
			):
			Xtr_k1x = datapersistant.load(os.path.join('3 streams', 'Xtr_rgb.pkl'))
			Xtr_k2x = datapersistant.load(os.path.join('3 streams', 'Xtr_of.pkl'))
			Xtr_k3x = datapersistant.load(os.path.join('3 streams', 'Xtr_depth.pkl'))
			ytrx = datapersistant.load(os.path.join('3 streams', 'ytr.pkl'))
			Xte_k1x = datapersistant.load(os.path.join('3 streams', 'Xte_rgb.pkl'))
			Xte_k2x = datapersistant.load(os.path.join('3 streams', 'Xte_of.pkl'))
			Xte_k3x = datapersistant.load(os.path.join('3 streams', 'Xte_depth.pkl'))
			ytex = datapersistant.load(os.path.join('3 streams', 'yte.pkl'))

			summary_data(*([Xtr_k1x, Xtr_k2x, Xtr_k3x], ytrx), *([Xte_k1x, Xte_k2x, Xte_k3x], ytex))
			return (([Xtr_k1x, Xtr_k2x, Xtr_k3x], ytrx), ([Xte_k1x, Xte_k2x, Xte_k3x], ytex))

	Xtr_rgb, ytr_rgb, ztr_rgb = rgb_dataset.getEvenIds()
	Xtr_of, ytr_of, ztr_of = of_dataset.getEvenIds()
	Xte_rgb, yte_rgb, zte_rgb = rgb_dataset.getUnevenIds()
	Xte_of, yte_of, zte_of = of_dataset.getUnevenIds()

	dd_tr1 = toDict(Xtr_rgb, ytr_rgb, ztr_rgb)
	dd_tr2 = toDict(Xtr_of, ytr_of, ztr_of)
	dd_te1 = toDict(Xte_rgb, yte_rgb, zte_rgb)
	dd_te2 = toDict(Xte_of, yte_of, zte_of)


	if streams==2:
		[dd_tr1m, dd_tr2m] = mergeDicts([dd_tr1, dd_tr2])
		[dd_te1m, dd_te2m] = mergeDicts([dd_te1, dd_te2])

		Xtr_k1x = np.array(extractFromDict(dd_tr1mm, 'data'))
		Xtr_k2x = np.array(extractFromDict(dd_tr2mm, 'data'))
		ytrx = fix_labelset(np.array(extractFromDict(dd_tr1mm, 'label')))
		Xte_k1x = np.array(extractFromDict(dd_te1mm, 'data'))
		Xte_k2x = np.array(extractFromDict(dd_te2mm, 'data'))
		ytex = fix_labelset(np.array(extractFromDict(dd_te1mm, 'label')))

		# save 2 streams
		datapersistant.save(Xtr_k1x, os.path.join('2 streams', 'Xtr_rgb.pkl'))
		datapersistant.save(Xtr_k2x, os.path.join('2 streams', 'Xtr_of.pkl'))
		datapersistant.save(ytrx, os.path.join('2 streams', 'ytr.pkl'))

		datapersistant.save(Xte_k1x, os.path.join('2 streams', 'Xte_rgb.pkl'))
		datapersistant.save(Xte_k2x, os.path.join('2 streams', 'Xte_of.pkl'))
		datapersistant.save(ytex, os.path.join('2 streams', 'yte.pkl'))

		summary_data(*([Xtr_k1x, Xtr_k2x], ytr), *([Xte_k1x, Xte_k2x], yte))
		return (([Xtr_k1x, Xtr_k2x], ytrx), ([Xte_k1x, Xte_k2x], ytex))


	# get depth stream
	Xtr_de, ytr_de, ztr_de = depth_dataset.extract(list(dd_tr1.keys()))
	Xte_de, yte_de, zte_de = depth_dataset.extract(list(dd_te1.keys()))
	dd_tr3 = toDict(Xtr_de, ytr_de, ztr_de)
	dd_te3 = toDict(Xte_de, yte_de, zte_de)

	[dd_tr1m, dd_tr2m, dd_tr3m] = mergeDicts([dd_tr1, dd_tr2, dd_tr3])
	[dd_te1m, dd_te2m, dd_te3m] = mergeDicts([dd_te1, dd_te2, dd_te3])

	Xtr_k1x = np.array(extractFromDict(dd_tr1m, 'data'))
	Xtr_k2x = np.array(extractFromDict(dd_tr2m, 'data'))
	Xtr_k3x = np.array(extractFromDict(dd_tr3m, 'data'))
	ytrx = fix_labelset(np.array(extractFromDict(dd_tr3m, 'label')))
	Xte_k1x = np.array(extractFromDict(dd_te1m, 'data'))
	Xte_k2x = np.array(extractFromDict(dd_te2m, 'data'))
	Xte_k3x = np.array(extractFromDict(dd_te3m, 'data'))
	ytex = fix_labelset(np.array(extractFromDict(dd_te3m, 'label')))

	# save 3 streams
	datapersistant.save(Xtr_k1x, os.path.join('3 streams', 'Xtr_rgb.pkl'))
	datapersistant.save(Xtr_k2x, os.path.join('3 streams', 'Xtr_of.pkl'))
	datapersistant.save(Xtr_k3x, os.path.join('3 streams', 'Xtr_depth.pkl'))
	datapersistant.save(ytrx, os.path.join('3 streams', 'ytr.pkl'))

	datapersistant.save(Xte_k1x, os.path.join('3 streams', 'Xte_rgb.pkl'))
	datapersistant.save(Xte_k2x, os.path.join('3 streams', 'Xte_of.pkl'))
	datapersistant.save(Xte_k3x, os.path.join('3 streams', 'Xte_depth.pkl'))
	datapersistant.save(ytex, os.path.join('3 streams', 'yte.pkl'))

	summary_data(*([Xtr_k1x, Xtr_k2x, Xtr_k3x], ytrx), *([Xte_k1x, Xte_k2x, Xte_k3x], ytex))
	return (([Xtr_k1x, Xtr_k2x, Xtr_k3x], ytrx), ([Xte_k1x, Xte_k2x, Xte_k3x], ytex))
