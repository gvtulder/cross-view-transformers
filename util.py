# Cross-view transformers for multi-view analysis of unregistered medical images
# Copyright (C) 2021 Gijs van Tulder / Radboud University, the Netherlands
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics


def plot_confmat(cm):
    classes = list(range(cm.shape[0]))

    fig = plt.figure(figsize=(2, 2), dpi=160, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")

    fig.set_tight_layout(True)
    return fig


def plot_roc_curve(y, y_pred):
    # see sklearn.metrics.plot_roc_curve
    fpr, tpr, _ = sklearn.metrics.roc_curve(y, y_pred, pos_label=1)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    viz = sklearn.metrics.RocCurveDisplay(
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc
    )

    fig = plt.figure(figsize=(2, 2), dpi=160, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    viz.plot(ax=ax)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    ax.tick_params(axis='x', labelsize=4)
    ax.tick_params(axis='y', labelsize=4)
    ax.legend(prop=dict(size=7))
    ax.set_aspect('equal')
    fig.set_tight_layout(True)
    return fig
