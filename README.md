# Redefining Entity Integration: Theoretical Insights for Improved Recommender Systems
<!--#### -->
## Introduction

Recommender systems utilize Graph Neural Networks (GNNs) to learn vectorized representations of users and items from user-item interactions for predicting recommendations. Recent methods enhance recommendations by incorporating item-related entities through a technique known as the Collaborative Knowledge Graph (CKG). However, the integration of these entities remains underexplored, leading to unresolved challenges in maintaining structural consistency and effectively utilizing entity information. This paper addresses three key research questions: (1) What properties should GNN-based recommender models satisfy? (2) How well do CKG-based models align with these requirements? (3) Can an alternative graph structure better integrate entities into recommender systems?
To answer these questions, we define two critical properties for GNN-based recommender models: {\em Local Consistency} and {\em Having Indispensable Entities}. We analyze CKG-based models and identify a fundamental limitation: they fail to simultaneously satisfy both properties. To resolve this issue, we propose a novel graph structure, the Fusion Graph (FG), which introduces additional user-entity connections to enhance entity integration. Our theoretical analysis shows that FG-based models better meet the requirements of recommender systems.
<!-- ![image](Images/KWF.png "The structure of Knowledge-wedging Frame work") -->
## Datasets

* [Amazon-book](http://jmcauley.ucsd.edu/data/amazon)

* [LastFM](https://grouplens.org/datasets/)

* [Yelp2018](https://www.yelp.com/dataset/challenge)

* [MovieLens](https://grouplens.org/datasets/movielens/)

## Requirements

* python >= 3.9

* torch>=1.7.0

* dgl>=0.7.0

* scikit-learn>=0.24.0






### Command and configurations

#### on Amazon-book
```bash
python -u main.py --model_type baseline  --dataset amazon-book --gpu_id 0 --ue_lambda 0.1 --idf_sampling 1 --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 3000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --sprate 1
```
#### on LastFM
```bash
python -u main.py --model_type baseline --dataset last-fm --gpu_id 0 --ue_lambda 0.1 --idf_sampling 1 --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 3000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --sprate 1
```
#### on Yelp2018
```bash
python -u main.py --model_type baseline --dataset yelp2018 --gpu_id 0 --ue_lambda 0.1 --idf_sampling 1 --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 3000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --sprate 1
```
#### on MovieLens
```bash
python -u main.py --model_type baseline --dataset movie-lens --gpu_id 0 --ue_lambda 0.4 --idf_sampling 1 --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 3000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --sprate 1
```
#### General flags

```{txt}
optional arguments:
  --dataset                       dataset                               
  --idf_sampling                  negative entity number
  --layer_size                    size of each layer
  --embed_size                    dimension of embedding vector 
  --epoch                         max epochs before stop
  --pretrain                      use pretrain or not
  --batch_size                    batch size
```
