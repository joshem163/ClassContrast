# ClassContrast
![framework](https://github.com/joshem163/ClassContrast/assets/133717791/c50235ad-e30c-4b87-86d0-2c3cc1e37d3c)


# Overview

ClassContrast is an innovative machine learning model for node classification that utilizes a contrastive approach, effectively integrating local neighborhood information and domain features of the nodes. This repository contains the code for the ClassContrast paper.
# Requirements
ClassContrast depends on the followings:
1. natsort 8.4.0
2. networkx 3.1
3. ogb 1.3.6
4. sklearn 1.3.0
5. torch_geometric 2.4.0
6. xgboost 2.0.1
   
The code is implemented in python 3.11.4. 
# Datasets
In this study,  ten benchmark datasets have been utilized, comprising three homophilic, two OGBN, and five heterophilic datasets. The link to access these datasets is provided below:

[CORA](https://linqs-data.soe.ucsc.edu/public/datasets/cora/cora.zip) 

[CITESEER](https://linqs-data.soe.ucsc.edu/public/datasets/citeseer-doc-classification/citeseer-doc-classification.zip)

[OGBN](https://ogb.stanford.edu/docs/nodeprop/)

[PUBMED](https://linqs-data.soe.ucsc.edu/public/datasets/pubmed-diabetes/pubmed-diabetes.zip)

[WebKB](https://github.com/bingzhewei/geom-gcn/tree/master/new_data)

[Wiki](https://github.com/benedekrozemberczki/MUSAE/tree/master/input)

# Runing the  Experiments
1. Execute all cell of notebook to run the experiment:
2. The code has the ability to download and preprocess the dataset by itself. The dataset will be stored in the same directory.
3. Make sure to maintain the sequential order of execution as specified in the code.
4. Follow the prompts or instructions provided by the code to proceed with the experiment.
5. For OGBN datasets, the final feature vectors will be saved as CSV files in the same directory for future use in classification tasks. 

