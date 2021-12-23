# h3_credit_card_clustering

# folder structure
```
.
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
## How to add a clustering method
We take keamns as a example:
1. add a keamns.py file in ./clustering
2. implatement 
2. add `from clustering.kmeans import `