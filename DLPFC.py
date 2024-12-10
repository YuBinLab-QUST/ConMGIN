from __future__ import division
from __future__ import print_function

import torch.optim as optim
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import os
import argparse
from config import Config
from utils import *
from models import ConMGIN
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
import torch

# 数据预处理与加载
def normalize(adata, highly_genes=3000):
    print("start select HVGs")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata

def load_ST_file(dataset, highly_genes, k, radius):
    path = "/DLPFC/" + dataset + "/"
    labels_path = path + "metadata.tsv"

    labels = pd.read_table(labels_path, sep='\t')
    labels = labels["layer_guess_reordered"].copy()
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()
    ground.replace('WM', '0', inplace=True)
    ground.replace('Layer1', '1', inplace=True)
    ground.replace('Layer2', '2', inplace=True)
    ground.replace('Layer3', '3', inplace=True)
    ground.replace('Layer4', '4', inplace=True)
    ground.replace('Layer5', '5', inplace=True)
    ground.replace('Layer6', '6', inplace=True)

    adata1 = sc.read_visium(path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata1.var_names_make_unique()
    obs_names = np.array(adata1.obs.index)
    positions = adata1.obsm['spatial']

    data = np.delete(adata1.X.toarray(), NA_labels, axis=0)
    obs_names = np.delete(obs_names, NA_labels, axis=0)
    positions = np.delete(positions, NA_labels, axis=0)

    adata = ad.AnnData(pd.DataFrame(data, index=obs_names, columns=np.array(adata1.var.index), dtype=np.float32))

    adata.var_names_make_unique()
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground
    adata.obsm['spatial'] = positions
    adata.obs['array_row'] = adata1.obs['array_row']
    adata.obs['array_col'] = adata1.obs['array_col']
    adata.uns['spatial'] = adata1.uns['spatial']
    adata.var['gene_ids'] = adata1.var['gene_ids']
    adata.var['feature_types'] = adata1.var['feature_types']
    adata.var['genome'] = adata1.var['genome']
    adata.var_names_make_unique()
    adata = normalize(adata, highly_genes=highly_genes)
    fadj = features_graph(adata.X, k=k)
    sadj, graph_nei, graph_neg = spatial_graph(adata, radius=radius)

    adata.obsm["fadj"] = fadj
    adata.obsm["sadj"] = sadj
    adata.obsm["graph_nei"] = graph_nei.numpy()
    adata.obsm["graph_neg"] = graph_neg.numpy()
    adata.var_names_make_unique()
    return adata

# 数据加载
def load_data(dataset):
    print("load data:")
    path = "../generate_data/DLPFC/" + dataset + "/Spatial_MGCN.h5ad"
    adata = sc.read_h5ad(path)
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground']
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    print("done")
    return adata, features, labels, nfadj, nsadj, graph_nei, graph_neg

# 训练模型
def train():
    model.train()
    optimizer.zero_grad()
    com1, com2, emb, pi, var, mean = model(features, sadj, fadj)
    bayes_loss = Bayesian(pi, theta=var, ridge_lambda=0).loss(features, mean, mean=True)
    graphcl_loss = graph_contrastive_loss(emb, graph_nei, graph_neg)
    con_loss = consistency_loss(com1, com2)
    total_loss = config.alpha * bayes_loss + config.beta * con_loss + config.gamma * graphcl_loss
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    return emb, mean, bayes_loss, graphcl_loss, con_loss, total_loss

# 主程序
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['151509']
    for i in range(len(datasets)):
        dataset = datasets[i]
        config_file = './config/DLPFC.ini'
        print(dataset)
        adata, features, labels, fadj, sadj, graph_nei, graph_neg = load_data(dataset)
        print(adata)

        plt.rcParams["figure.figsize"] = (3, 3)
        savepath = './result/DLPFC/' + dataset + '/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        title = "Manual annotation (slice #" + dataset + ")"
        sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title, show=False)
        plt.savefig(savepath + 'Manual Annotation.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        config = Config(config_file)
        cuda = not config.no_cuda and torch.cuda.is_available()
        use_seed = not config.no_seed

        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        config.n = len(ground)
        config.class_num = len(ground.unique())

        config.epochs = 200
        config.epochs = config.epochs + 1

        if cuda:
            features = features.cuda()
            sadj = sadj.cuda()
            fadj = fadj.cuda()
            graph_nei = graph_nei.cuda()
            graph_neg = graph_neg.cuda()

        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        if not config.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
        print(dataset, ' ', config.lr, ' ', config.alpha, ' ', config.beta, ' ', config.gamma)
        model = ConMGIN(nfeat=config.fdim,
                             nhid1=config.nhid1,
                             nhid2=config.nhid2,
                             dropout=config.dropout,
                            )
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        epoch_max = 0
        ari_max = 0
        idx_max = []
        mean_max = []
        emb_max = []

        for epoch in range(config.epochs):
            emb, mean, bayes_loss, graphcl_loss, con_loss, total_loss = train()
            print(dataset, ' epoch: ', epoch, ' bayes_loss = {:.2f}'.format(bayes_loss),
                  ' graphcl_loss = {:.2f}'.format(graphcl_loss), ' con_loss = {:.2f}'.format(con_loss),
                  ' total_loss = {:.2f}'.format(total_loss))
            kmeans = KMeans(n_clusters=config.class_num).fit(emb)
            idx = kmeans.labels_
            ari_res = metrics.adjusted_rand_score(labels, idx)
            if ari_res > ari_max:
                ari_max = ari_res
                epoch_max = epoch
                idx_max = idx
                mean_max = mean
                emb_max = emb
        print('final epoch: ', epoch_max)
        print('ARI = {:.2f}'.format(ari_max))

        title = 'Spatial-MGCN: ARI={:.2f}'.format(ari_max)
        adata.obs['idx'] = idx_max.astype(str)
        adata.obsm['emb'] = emb_max
        adata.obsm['mean'] = mean_max

        sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title, show=False)
        plt.savefig(savepath + 'Spatial_MGCN.jpg', bbox_inches='tight', dpi=600)
        plt.show()
