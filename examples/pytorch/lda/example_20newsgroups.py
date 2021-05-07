# Copyright 2021 Yifei Ma
# Modified from scikit-learn example "plot_topics_extraction_with_nmf_lda.py"
# with the following original authors with BSD 3-Clause:
# * Olivier Grisel <olivier.grisel@ensta.org>
# * Lars Buitinck
# * Chyi-Kwei Yau <chyikwei.yau@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from time import time
import matplotlib.pyplot as plt
import warnings

import numpy as np
import scipy.sparse as ss
import torch
import dgl
from dgl import function as fn

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

from lda_model import LatentDirichletAllocation as LDAModel

n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20
device='cuda'

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

print("Loading dataset...")
t0 = time()
data, _ = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'),
                             return_X_y=True)
data_samples = data[:n_samples]
data_test = data[n_samples:2*n_samples]
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf_vectorizer.fit(data)
tf = tf_vectorizer.transform(data_samples)
tt = tf_vectorizer.transform(data_test)

tf_feature_names = tf_vectorizer.get_feature_names()
tf_uv = [(u,v)
         for u,v,e in zip(tf.tocoo().row, tf.tocoo().col, tf.tocoo().data)
         for _ in range(e)]
tt_uv = [(u,v)
         for u,v,e in zip(tt.tocoo().row, tt.tocoo().col, tt.tocoo().data)
         for _ in range(e)]
print("done in %0.3fs." % (time() - t0))
print()

print("Preparing dgl graphs...")
t0 = time()
G = dgl.heterograph({('doc','topic','word'): tf_uv}, device=device)
Gt = dgl.heterograph({('doc','topic','word'): tt_uv}, device=device)
print("done in %0.3fs." % (time() - t0))
print()

print("Training dgl-lda model...")
t0 = time()
model = LDAModel(G, n_components)
model.fit(G)
print("done in %0.3fs." % (time() - t0))
print()

print(f"dgl-lda training perplexity {model.perplexity(G):.3f}")
print(f"dgl-lda testing perplexity {model.perplexity(Gt):.3f}")

plot_top_words(
    type('dummy', (object,), {'components_': G.ndata['z']['word'].cpu().numpy().T}),
    tf_feature_names, n_top_words, 'Topics in LDA model')

print("Training scikit-learn model...")

print('\n' * 2, "Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0,
                                verbose=1,
                               )
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))
print()

print(f"scikit-learn training perplexity {lda.perplexity(tf):.3f}")
print(f"scikit-learn testing perplexity {lda.perplexity(tt):.3f}")
