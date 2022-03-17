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
**Attention:** GTX/RTX series GPUs may fail to compile DCNv2. TITAN/Tesla series GPUs are recommended.


* Install other dependencies

All other dependencies can be installed via 'pip'.

## * Dataset Preparation
Download data and add them to `./datasets`. 

The data structure for the Season-Varying dataset has been already given in that folder. 

The LEVIR-CD dataset and the SYSU-CD dataset share the same data structure. 

Note that the instances in original LEVIR-CD dataset are cropped from 1024×1024 to 256×256.


## * Test
You can download our pretrained models for Season-Varying, LEVIR-CD and SYSU-CD from [Baidu Netdisk, code: tgrs](https://pan.baidu.com/s/1rux9Zxjc8yGsga28CSD0kg), [Baidu Netdisk, code: tgrs](https://pan.baidu.com/s/1DTazE7I3lhELPRZr5oyniQ) and [Baidu Netdisk, code: tgrs](https://pan.baidu.com/s/1CDkcUUpdd0w9tz4fe7no0A), respectively. 


Then put them in `./checkpoints/SV/trained_models`, `./checkpoints/LEVIR-CD/trained_models` and `./checkpoints/SYSU-CD/trained_models`, separately.

* Test on the Season-Varying dataset

```python
python test.py --dataset SV --name SV --load_pretrain True --which_epoch 194
```

* Test on the LEVIR-CD dataset

```python
python test.py --dataset LEVIR-CD --name LEVIR-CD --load_pretrain True --which_epoch 255
```

* Test on the SYSU-CD dataset

```python
python test.py --dataset SYSU-CD --name SYSU-CD --load_pretrain True --which_epoch 57
```

## * Train & Validation
```python
python trainval.py --dataset SV --name SV 
```
All the hyperparameters can be adjusted in `./config`.

During training, the occupied GPU memory is around **3357MB** when batch size is 8, and around **4101MB** when batch size is 16, on single TITAN X. 


## * Supplement
You can download all predictions (shown as left, below) of our ISNet for Season-Varying, LEVIR-CD and SYSU-CD test sets from [Baidu Netdisk, code: tgrs](https://pan.baidu.com/s/194O19U0I3Pq766cggjmQTQ), [Baidu Netdisk, code: tgrs](https://pan.baidu.com/s/11QsyHkzwlaYGEmlysQL6Uw) and [Baidu Netdisk, code: tgrs](https://pan.baidu.com/s/1Wl4Iq_tee3Lhx6pa3FqnXA), respectively. 

![](https://github.com/xingronaldo/ISNet/tree/main/ISNet/predictions/ISNet.png "raw_prediction") ![](https://github.com/xingronaldo/ISNet/tree/main/ISNet/predictions/ISNet_marked.png "marked_prediction")

## * Contact
Don't hesitate to contact me if you have any questions.

Email: guangxingwang@mail.nwpu.edu.cn



