# ICON
Code release for "Make the U in UDA Matter: Invariant Consistency Learning for Unsupervised Domain Adaptation" (NeurIPS 2023). Paper is available [here](https://arxiv.org/pdf/2309.12742.pdf).

## Prerequisites
- torch>=1.7.0
- torchvision
- qpsolvers
- numpy
- prettytable
- tqdm
- scikit-learn
- webcolors
- matplotlib


## Training

Replace {data_dir} with the dataset directory. Missing datasets will be downloaded automatically. Replace {log_dir} with the logging directory (for storing model checkpoints, tensorboard logs and console logs). For Office-Home, source (-s) and target domain (-t) takes values from {'Ar', 'Cl', 'Rw', 'Pr'}.

VisDA-2017
```
CUDA_VISIBLE_DEVICES=0 python run_icon.py {data_dir} -d VisDA2017 -s Synthetic -t Real -a resnet50 --epochs 50 --lr 0.002 --per-class-eval --temperature 3.0 --center-crop --w-transfer 0.08 --w-st 1.0 --threshold 0.97 --log-root {log_dir} --batch-size 28 --optim sgd --con-start-epoch 5 --con-mode sim --w-inv 0.25 --inv-start-epoch 5 --back-cluster-start-epoch 9 --topk 3 --dim-reduction umap --reduced-dim 50 --eqinv --exp-name visda_reproduce --seed 0
```

Office Home
```
CUDA_VISIBLE_DEVICES=0 python run_icon.py {data_dir} -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 50 --lr 0.005 --temperature 2.5 --bottleneck-dim 2048  --w-transfer 0.015 --w-st 0.5 --threshold 0.97 --log-root {log_dir} --batch-size 28  --con-start-epoch 0 --con-mode stats --back-cluster-start-epoch 0 --topk 5 --seed 0 --w-inv 0.1 --inv-start-epoch 10 --exp-name Ar2Cl --optim sam
```

## Acknowledgement
This code is implemented based on the [CST](https://github.com/Liuhong99/CST), and it is our pleasure to acknowledge their contributions.


## Citation
If you use this code for your research, please consider citing:
```
@article{yue2023make,
  title={Make the U in UDA Matter: Invariant Consistency Learning for Unsupervised Domain Adaptation},
  author={Yue, Zhongqi and Sun, Qianru and Zhang, Hanwang},
  journal={Advances in neural information processing systems},
  year={2023}
}
```

## Contact
If you have any problem about our code, feel free to contact
- yuez0003@ntu.edu.sg