#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 17:31:29 2018

@author: pipolose
"""

import numpy as np
import six
import matplotlib.pyplot as plt


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'],
                     edge_color='w', bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    '''
    Make a nice figure show an input dataframe as a table
    '''
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) *\
            np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    cell_text = [['{:.2f}'.format(j) if ~np.isnan(j) else '-' for j in i]
                 for i in data.values]
    mpl_table = ax.table(cellText=cell_text, bbox=bbox,
                         colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax
