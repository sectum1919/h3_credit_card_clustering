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