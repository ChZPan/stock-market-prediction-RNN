import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_seqs(seqs, 
              datetime,
              labels=None, 
              title=None, 
              ylabel=None,
              linewidth=1.0):
    
    plt.figure(figsize=(20, 5))
    
    xticks_loc, years = set_xticks(datetime)
    
    if type(seqs) is list:
        for i, seq in enumerate(seqs):
            if labels is None:
                plt.plot(seq, linewidth=linewidth)
            else:
                plt.plot(seq, label=labels[i], linewidth=linewidth)
    else:
        if labels is None:
            plt.plot(seqs, linewidth=linewidth)
        else:
            plt.plot(seqs, label=labels, linewidth=linewidth)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.xticks(xticks_loc, years, rotation=45)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()
    
    
def set_xticks(datetime):
    year_month = datetime.dt.to_period("M")
    ym_unique = year_month.unique()
    xticks_loc = []
    for ym in ym_unique:
        loc = np.where(year_month == ym)[0][0] + datetime.index[0]
        xticks_loc.append(loc)
    return xticks_loc, ym_unique.astype(str)


def create_labels(labels, val_1, val_2 = None):
    if type(val_1) is not list:
        val_1 = [val_1]
        val_2 = [val_2]
    assert len(labels) - len(val_1) == 1, \
    "Some labels might be missing."

    for i in range(len(labels)):
        if i == 0:
            labels[i] += " (MSE/ACC)"
        else:
            labels[i] = labels[i] \
                        + " (" \
                        + str(round(val_1[i-1], 2)) \
                        + "/" \
                        + str(round(val_2[i-1], 2)) \
                        + ")"
    return labels