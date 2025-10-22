# Time Series Clustering and Classification with Network Latency Data

## Usage
For data preprocessing/cluster -> classification pipeline only use `ts_batch_preprocess.ipynb` and `train_patchtst.py`.
The notebook learns the important features from a random sample of the entire dataset (one random day from each region) and trains a global GMM. It then uses the trained global GMM to cluster the entire dataset. This notebook cleans up the dataset by removing sparse 15-second groups and interpolation for missing values (token: missing -> -1, nearest: missing -> nearest existing value). The resulting files are saved as `cluster_results.csv` which contain a mapping of 15-second groups to cluster and `clustering_data.csv` that contains a cleaned version of the dataset. Each file contains all data from a single day.
The training script is used on the preprocessed data to train a PatchTST Classification model. The current code was tested on ~400k 15-second groups classifying the cluster based on the first second of data (100 10ms datapoints). Additionally, the training script includes support for multi-gpu training. See example below:

```
torchrun --standalone --nproc_per_node=8 train_patchtst.py \
    --distributed \
    --root-dir data/LENS-2023-11-CSV/LENS-2023-11-CSV/inside-out/active \
    --channels token \
    --epochs 250 --batch-size 128 --num-workers 8 \
    --lr 0.001 --amp \
    --output-dir saved_models/patchtst2_reg_large \
    --eval-interval 5 --save-interval 25 \
    --monitor-metric val_loss \
    --num-hidden-layers 2 \
    --ffn-dim 512 \
    --patch-len 20 \
    --attention-dropout 0.1 \
    --ff-dropout 0.2 \
    --positional-dropout 0.05 \
    --path-dropout 0.05
```

## Files
- `plot_data.ipynb`: Initial experimental notebook to plot and get a feel of the data
- `train_patchtst.py`: Full patchtst training script that classifies subsequences to predefined clusters
- `ts_batch_preprocess.ipynb`: Run all cells to train a feature extractor/GMM clustering model that is used to cluster (and clean/preprocess) all raw data from a parent directory
- `ts_classification.ipynb`: Experimental notebook to train a patchtst model on preprocessed data
- `ts_cluster.py`: Utility python file for clustering and feature extraction
- `ts_preprocess.py`: Utility python file for cleaning and preprocessing data
- `ts_processing.ipynb`: Experimental notebook for preprocessing data
- `ts_training_irtt.ipynb`: Initial experimental notebook for slight modifications to clustering algorihtm