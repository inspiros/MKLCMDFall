# MKL evaluation on CMDFall
Dedicated to run on RGB and OF FC features pre-extracted in server:
- RGB: `/home/minhkv/datasets/feature/minhkv/mica_4000/bin`
- OF: `/home/minhkv/feature/mica/merged_flow/bin`
Download these folders to local computer or push this code folder to server.
### Input
- FC RGB features folder
- FC OF features folder
- Depth features file
### Output

Confusion matrices, classification reports and f1-score csv of
- SVM on single modality
- SVM on concatenated normalized features
- MKL

In the following format:
```python
result
└───confusion #contains confusion matrices
│	└───3 streams '''[streams]'''
│		└───laplacian '''[kernels]'''
│			└───20 classes '''[kernels]'''
│			│	└─── '''<Confusion matrix images and csv files>'''
│			└───6 classes '''[kernels]'''
│			│	└─── '''<Confusion matrix images and csv files>'''
│			└───2 classes '''[kernels]'''
│				└─── '''<Confusion matrix images and csv files>'''
└───report #contain classification reports
	└───3 streams
│		└───laplacian '''[kernels]'''
│			└───20 classes '''[kernels]'''
│			│	└─── '''<classification report csv>'''
│			└───6 classes '''[kernels]'''
│			│	└─── '''<classification report csv>'''
│			└───2 classes '''[kernels]'''
│				└─── '''<classification report csv>'''
```
## How to use
### Dependencies
Embedded MKL codes, no need to download library. But possibly requires `sklearn`, `pandas`, `matplotlib`
### Configure parameters
Firstly, set the dataset folders in file `configs.py`:
```python
RGB_DATASET_ROOT = '/home/path/to/folder/containing/RGB/features'
OF_DATASET_ROOT = '/home/path/to/folder/containing/OF/features'
DEPTH_DATASET_FILE = '/home/path/to/Depth/file'
```
[Optional] Modify kernels, SVM, MKL parameters in file `configs.py`. There are predefined **linear**, **rbf** and **laplacian** configurations which I used.

To add a new kernels configuration, append that file with the same format as predefined configs. For example:
```python
'''
Polynomial config
'''
from sklearn.metrics import pairwise # Optional

def polynomial_rgb(X, L=None): # Kernel computing call-back for rgb, always with these input params
	# Return kernel function from package sklearn.metrics.pairwise
	# Add hyperparameters if needed here, in this case, "degree=3"
	return pairwise.polynomial_kernel(X, L, degree=3)

def polynomial_of(X, L=None): # Kernel computing call-back for depth
	return pairwise.polynomial_kernel(X, L, degree=4)

def polynomial_depth(X, L=None): # Kernel computing call-back for depth
	return pairwise.polynomial_kernel(X, L, degree=6)

def polynomial_concatenate(X, L=None): # Kernel computing call-back for concatenated features
	return pairwise.polynomial_kernel(X, L, degree=4)

polynomial_params = Params(name = 'polynomial', # name of the kernels configuration
					assignable_names = ['poly', 'polynomial'], # accepted names when you run command, eg: --kernels=poly
					kernel_func_rgb = polynomial_rgb,
					kernel_func_of = polynomial_of,
					kernel_func_depth = polynomial_depth,
					kernel_func_concatenate = polynomial_concatenate,
					C_mkl = 0.25, # C of base learner of MKL
					C_rgb = None, # (Optional) C of SVM on rgb, equals C_mkl if None
					C_of = None, # (Optional) C of SVM on of, equals C_mkl if None
					C_depth = None, # (Optional) C of SVM on depth, equals C_mkl if None
					C_concatenate = 10, # (Optional) C of SVM on concatenated features, equals C_mkl if None
					lam_mkl = None # (Optional) lamda [0,1] of EasyMKL, 0.0 if None
					)

CONFIGS.append(polynomial_params.to_program_config())
```
### Run
Execute file `run_cmdfall_classification.py`:
- Run command manually from terminal with the following format:
`python3 run_gesture_classification.py --streams=3 --kernels=linear`

> Currently supported arguments:

> |Argument|Meaning|eg.|
> |---|---|---|
> |`confusion_matrix`|Export confusion matrix or not, default True|True|
> |`classification_report`|Export classification report or not, default True|True|
> |`streams`|Number of modalities: 2 for RGB-OF, 3 (default) for RGB-OF-D|3|
> |`num_classes`|Number of classes: 20/6/2, None (default) for all|6|
> |`kernels`|Kernels configurations set in `configs.py`, accepting only keywords in `assignable_names`|linear|


- Run multiple commands sequentially:
-- Modify `evaluation_procedure` file
-- Execute it from terminal `./evaluation_procedure`
> Run `chmod 777 evaluation_procedure` if not working
