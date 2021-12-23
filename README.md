# h3_credit_card_clustering
English version: [README_en.md](./docs/README_en.md)
# 文件夹结构
```
.
├── docs
│   └── README_en.md
├── data
│   └── cc_general.csv
├── fig
│   └── {method_name}.png
├── README.md
├── clustering
│   └── {method_name}.py
├── dataset.py
└── main.py
```

# 环境准备
```
pip install git+https://github.com/jqmviegas/jqm_cvi.git 
```

# Q&A
## 如何添加一个聚类算法
以 keamns 为例:
1. 在 `./clustering`中添加一个`.py` 文件
2. 在上述文件中实现一个函数`kmeans_clustering`用以进行聚类，其输入输出为：
    - 输入`[counts*dim]`的`ndarray`，内容为每条数据的参数
    - 输出长为`counts`的`list`，内容为每条数据的类别号
3. 在`main.py`中修改如下代码，其中`name`字段用于log和保存tSNE图：
    ```
    # 添加
    from clustering.kmeans import kmeans_clustering
    methods = [
            # 添加
            {"func":kmeans_clustering, "name":'kmeans'},
        ]
    ```
4. 运行`main.py`即可看到结果

# 任务目标
- [ ] 至少自己实现两种不同类型的聚类算法
    - [ ] 需要调研和选择
- [ ] 用多种评价指标对聚类算法、聚类趋势、聚类质量进行评估
    - [ ] 聚类算法（算法性能、稳定性、复杂度）
    - [x] 聚类趋势（如Hopkins Statistic）
    - [ ] 聚类质量（紧凑度和分离度）
        - [x] Silhouette Coefficient
        - [x] Calinski-Harabasz Index
        - [x] Davies-Bouldin Index(DB/DBI)
        - [ ] Dunn Validity Index(DVI)
- [ ] 分析上述不同聚类算法
    - [ ] 算法之间的差异、优缺点
    - [ ] 结合数据集特点，指出在题目中的任务场景下哪种聚类算法更适合
- [ ] 对聚类结果进行可视化和可解释性分析
    - [ ] 结合聚类结果充分分析不同簇对应用户群体的潜在消费行为模式
    - [ ] 并且尝试从不同角度对聚类结果进行分析和阐释