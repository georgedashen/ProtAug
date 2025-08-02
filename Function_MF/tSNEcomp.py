import argparse
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw tSNE map for protein features or extracted ppi features')
    parser.add_argument('--embedding', type=str, help='embedding folder')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--augfile', type=str, default='')
    parser.add_argument('--output', type=str, default='result')
    parser.add_argument('--only_train', action='store_true', default=False, help='only include training set, not augmented')
    parser.add_argument('--subset', type=str, default='aug', choices=['aug','test','full'])
    parser.add_argument('--test', action='store_true', default=False, help='only include a small subset of data')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train_embed = np.load(f'{args.embedding}/mf-X_train.npy')
    if not args.only_train and args.subset in ['aug','full']:
        if len(args.augfile) >0:
            aug_embed = np.load(args.augfile)
        else:
            args.augfile = list(filter(lambda x:'aug' in x, os.listdir(f'{args.embedding}')))[0]
            aug_embed = np.load(f'{args.embedding}/{args.augfile}')
    
    if args.subset in ['test','full']:
        test_embed = np.load(f'{args.embedding}/mf-X_test.npy')

    if args.test:
        train_embed = train_embed[:1000]
        if not args.only_train and args.subset in ['aug','full']:
            aug_embed = aug_embed[:1000]

    extend_labels = ['train' for i in range(train_embed.shape[0])]
    extend_features = train_embed
    if not args.only_train:
        if args.subset in ['aug','full']:
            extend_features = np.concatenate((extend_features, aug_embed), axis=0)
            extend_labels.extend(['augment' for i in range(aug_embed.shape[0])])
        if args.subset in ['test','full']:
            extend_features = np.concatenate((extend_features, test_embed), axis=0)
            extend_labels.extend(['test' for i in range(test_embed.shape[0])])
    feat_cols = [ 'latent'+str(i) for i in range(extend_features.shape[1]) ]
    df = pd.DataFrame(extend_features, columns=feat_cols)
    df['label'] = extend_labels

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df[feat_cols])

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    plt.figure(figsize=(12,8))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        data=df,
        legend="full",
        alpha=0.5,
        s=3
    )
    
    if args.test:
        plt.show()
    else:
        os.system('mkdir -p figures')
        plt.savefig(f'figures/{args.output}.pdf')
        os.system(f'pdftoppm figures/{args.output}.pdf figures/{args.output} -png -rx 300 -ry 300')
