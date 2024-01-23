# KelpChallenge

说明：
## 数据处理
首先需要放置数据集，根据下面的格式：
```
KelpChallenge
└── datasets
    └── datasets
        ├── test_satellite
        ├── train_kelp
        └── train_satellite
```
之后，进入[dataset.ipynb](https://github.com/IICI-CV/KelpChallenge/blob/main/dataset.ipynb)执行即可生成一些训练需要的`txt`文件。

## 代码执行
- 若要运行代码
  - 请修改[scripts/train.sh](https://github.com/IICI-CV/KelpChallenge/blob/main/scripts/train.sh)
  - 之后执行 `bash scripts/train.sh GPU数量 123322` 即可运行。
