# ConMGIN
# Overview
ConMGIN (Convolutional Graph Isomorphism Network) is a graph representation learning framework designed to extract embeddings from gene expression profiles and spatial information through multi-layer Graph Isomorphism Networks (GIN), capturing global features of spatial data. The method first constructs two adjacent graphs and utilizes the GIN architecture to extract graph embeddings. Then, ConMGIN incorporates a hybrid Bayesian network to process the extracted embeddings, using probability distributions and prior distributions to effectively model data uncertainty and sample differences. Additionally, ConMGIN integrates graph contrastive learning to fully account for the spatial structure and local relationships of the images. Through interpretability analysis, ConMGIN demonstrates strong capabilities in cross-slice clustering and spatial trajectory analysis, capturing spatial structural similarities between different slices and identifying spatial hierarchies. Experimental results show that ConMGIN significantly outperforms existing baseline methods across multiple spatial transcriptomics datasets, with higher recognition accuracy and broad application potential.
# Package:ConMGIN
# Requirements
python 3.9

torch

anndata

argparse

sklearn

metrics

cluster

matplotlib

numpy

random

scipy
# Dataset
LIBD Human Dorsolateral Prefrontal Cortex (DLPFC) Dataset

Available at: http://spatial.libd.org/spatialLIBD
Processed Stereo-seq Dataset from Mouse Olfactory Bulb Tissue

Available at: https://github.com/JinmiaoChenLab/
10x Visium Spatial Transcriptomics Dataset of Human Breast Cancer

Available at: https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1
Slide-seqV2 Dataset

Available at: https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomicsat-near-cellular-resolution-with-slide-seqv2#study-summary
