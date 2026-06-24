# incus_track_les/overview_plot.py

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import rcParams
import pandas as pd
import numpy as np
import datetime as dt
from palettable.cartocolors.qualitative import Prism_10

STYLE_PATH = Path(__file__).parent / "custom_style.mplstyle"
FONT_DIR = ["/home/gleung/scripts/futura"]

prism = Prism_10.mpl_colors
blue = prism[1]
cyan = prism[2]
green = prism[3]
purple = prism[0]
red = prism[7]
orange = prism[6]
yellow = prism[5]
gray = "#303039"

def apply_custom_style():
    for font_path in font_manager.findSystemFonts(FONT_DIR):
        entry = font_manager.FontEntry(
            fname=font_path,
            name="Futura",
            weight="normal",  
            style="normal"
        )
        font_manager.fontManager.ttflist.append(entry)

    plt.style.use(STYLE_PATH)
        
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Futura"]
    

def plot_metric_fig(run_paths, experiment_params, experiment_label, metric='valid_features', **kwargs):
    apply_custom_style()

    fig, axes = plt.subplots(2,3,figsize=(10,6),sharex=True)
    axes = axes.flatten()

    for ax, run in zip(axes, run_paths):
        ax.set_title(run.name)

        if metric=='valid_features':
            ax = plot_valid_feature_ax(ax, run, experiment_params, experiment_label)
        elif metric == 'violin':
            ax = plot_violin_ax(ax, run, experiment_params, experiment_label, **kwargs)
        elif metric=='cell_traj':
            ax = plot_cell_traj_ax(ax, run, experiment_params, experiment_label, **kwargs)

    plt.suptitle(f"Experiment: {experiment_label}",fontsize=20)

    return(fig)

def plot_valid_feature_ax(ax, run_path, experiment_params, experiment_label):

    for fts,trks,col,name in zip(sorted(run_path.glob('features*pq')),
                                 sorted(run_path.glob('tracks*pq')),
                                 prism, 
                                 experiment_params):
        tracks = pd.read_parquet(trks)
        features = pd.read_parquet(fts)
        ax.bar(name, 100*len(tracks)/len(features), color=col, width=0.25)

    ax.set_ylabel('% Features Tracked')
    ax.set_xlabel(experiment_label)

    return(ax)  

def plot_violin_ax(ax, run_path, experiment_params, experiment_label, **kwargs):
    p25 = []
    p75 = []

    for i, (trks,col) in enumerate(zip(
                                 sorted(run_path.glob('tracks*pq')),
                                 prism)):
        tracks = pd.read_parquet(trks)

        plot_values = tracks[kwargs.get('plotvar')].dropna().values
        p25.append(np.percentile(plot_values, 1))
        p75.append(np.percentile(plot_values, 99))

        ax.violinplot(plot_values,
                      [i],
                      facecolor=(col,0.5),
                      linecolor=col, 
                      showmedians=True,
                      showextrema=False)

    ax.axhline(0, ls=':',color='gray',lw=0.5)

    ax.set_xticks(range(len(experiment_params)),experiment_params)          

    ax.set_ylabel(kwargs.get('plotvar_label'))
    ax.set_xlabel(experiment_label)

    ax.set_ylim(np.min(p25),np.max(p75))

    return(ax)
        

def plot_cell_traj_ax(ax, run_path, experiment_params, experiment_label, **kwargs):

    for i, (trks,col,name) in enumerate(zip(
                                 sorted(run_path.glob('tracks*pq')),
                                 prism,
                                 experiment_params)):
        tracks = pd.read_parquet(trks)

        tracks['frac_lifetime'] = (tracks['time_cell']/(tracks['lifetime']*dt.timedelta(minutes=1)))

        xs = np.linspace(0,1,21)

        out = []
        for i, x in tracks.groupby('cell'):
            out.append(np.interp(xs,x.frac_lifetime, x[kwargs.get('plotvar')]))

        ax.plot(xs,np.array(out).mean(axis=0),color=(col,0.8),label=name)
        
    ax.set_ylabel(kwargs.get('plotvar_label'))
    ax.set_xlabel('Fraction of Cell Lifetime')


    return(ax)