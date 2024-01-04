# FacadeNet

## FacadeNet's code is built based on the swapping-autoencoder [repo](https://github.com/taesungp/swapping-autoencoder-pytorch) code structure

## Conda Enviroment
#### Create new conda enviroment
```conda env create -f environment.yml```

## Datasets
#### Structure 
Root
  - Facades(Rectified facade images downloaded from [lsaa]( dataset)
  - horizontal maps
  - vertical maps
  - DINO features(Extract features using [DINOv2](https://github.com/facebookresearch/dinov2))
  - Depth
#### Click on the link to download the data: [Link](https://github.com/ZPdesu/lsaa-dataset)

## Training
#### Change the file paths paths in config file experiments/facades_vec_pc_lr_launcher.py
#### Run 
```python -m experiments facades_vec_pc_lr train facadenet```
