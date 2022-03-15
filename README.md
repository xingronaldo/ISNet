# Code for ISNet & ISNet-lw.
---------------------------------------------
Here I provide PyTorch implementations for ISNet and ISNet-lw.


## * Requirements
>TITAN X<br>
>python 3.6.5<br>
>PyTorch 1.7.0

## * Installation
Clone this repo:

```shell
git clone https://github.com/xingronaldo/ISNet.git
cd ISNet/ISNet
```

* Install DCNv2

```shell
cd DCNv2
python setup.py build develop
cd ..
```
Attention: GTX/RTX series GPUs may fail to compile DCNv2. TITAN/Tesla series GPUs are recommended.


* Install other dependencies

All other dependencies can be installed via 'pip'.

## * Dataset Preparation
Download data and add them to `./datasets`. 

The data structure for the Season-Varying dataset has been already given in that folder. 

The LEVIR-CD dataset and the SYSU-CD dataset share the same data structure. 

Note that the instances in original LEVIR-CD dataset are cropped from 1024×1024 to 256×256.


## * Test
You can download our pretrained models for Season-Varying and LEVIR-CD from [Baidu Netdisk, code: tgrs](https://pan.baidu.com/s/1rux9Zxjc8yGsga28CSD0kg) and [Baidu Netdisk, code: tgrs](https://pan.baidu.com/s/1DTazE7I3lhELPRZr5oyniQ), respectively. 

Then put them in `./checkpoints/SV/trained_models` and `./checkpoints/LEVIR-CD/trained_models`, separately.

* Test on the Season-Varying dataset

```python
python test.py --dataset SV --name SV --load_pretrain True --which_epoch 194
```

* Test on the LEVIR-CD dataset

```python
python test.py --dataset LEVIR-CD --name LEVIR-CD --load_pretrain True --which_epoch 255
```

## * Train & Validation
```python
python trainval.py --dataset SV --name SV 
```
All the hyperparameters can be adjusted in `./config`.

During training, the occupied GPU memory is around 3357MB when batch size is 8, and around 4101MB when batch size is 16, on single TITAN X. 







