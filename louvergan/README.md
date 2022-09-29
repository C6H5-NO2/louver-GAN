# 调试方法
## `louvergan/config.py`
```py
class HyperParam:
    n_epoch: int = 2000  # 总训练轮数
    save_step: int = 50  # 测试步长

    latent_dim: int = 128  # 隐向量长度 100 - 300
    batch_size: int = 512  # 批大小 128, 256, 512

    gen_lr: float = 2e-4  # 生成器学习率 1e-4, 2e-4, 5e-3, 1e-3
    gen_weight_decay: float = 1e-6  # 生成器衰减（一般不用改）

    dis_lr: float = 2e-4 # 判别器学习率 1e-4, 2e-4, 5e-3, 1e-3


DATASET_EVAL = {
    'statistics': ['education'],  # 数据相似性测试的列名
    'classification': 'label',  # 分类测试的列名
    'regression': 'hours-per-week',  # 回归测试的列名
    'clustering': 'label',  # 聚类测试的列名（一般和分类的相同）
}
```



## `louvergan/corr/config.py`
这是给 correlation 解析用的，不是主模型。

```py
class CorrSolverConfig:
    n_epoch: int = 1000  # 总训练轮数
    save_step: int = 50  # 保存步长

    latent_dim: int = 128  # 隐向量长度
    batch_size: int = 256  # 批大小 128, 256, 512
    lr: float = 1e-4  # 统一的学习率
```


## Statistics
distance 越小越好


## Classification
所有越大越好


## Regression
r2 越大越好  
rmse 越小越好  
mae 越小越好  


## Clustering
nmi, ami 越大越好


## Correlation
nmi matrix 和真实的相近  
rmse, mae 越小越好  

pjsd 越小越好  
