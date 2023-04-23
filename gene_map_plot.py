import pandas as pd
from LabData import config_global as config
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

MWAS_DIR = Path(config.analyses_dir).joinpath('20220402_234043_mwas_bmi_10K')
BASE_DIR = Path(config.analyses_dir)
FIGS_DIR = BASE_DIR.joinpath('Figures', 'gene_map')
FIGS_DIR.mkdir(parents=True, exist_ok=True)



### visualization:

def genome_plot():

    # Initialize the figure and gridspec
    fig = plt.figure(figsize=(8.5, 3.5), constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=7, figure=fig)
    FS = 6
    bonn_5_line = .05 / 12686191

    ########## TOP: SNPs ##########
    snps_df = pd.read_csv(MWAS_DIR.joinpath('snp_annotations', 'Rep_3066_ALL.csv'))
    snps_df = snps_df.loc[snps_df['Contig'] == 'C_257']  # only plotting this contig

    ax0 = fig.add_subplot(spec[:5])

    ## for each SNP, plot according to its Position, and its Pval (uncorrected?)
    ax0.scatter(snps_df['Position'], snps_df['Pval'],
                marker='o', s=2, facecolor='black', edgecolor='black', alpha=.45
                )
    #                marker=markers[tp], s=msizes, facecolor=tp_facecolors[tp],
    #                edgecolor=edgecolors, alpha=.45, linewidths=1)


    ax0.set_yscale('log')
    min_pval = min(snps_df['Pval'].min(), bonn_5_line)
    min_pval_in_fig = min_pval * 10 ** (0.1 * np.log10(min_pval)) # Add 10% white space above
    ax0.set_ylim([1, min_pval_in_fig])
    ticks_major = max(np.log10(min_pval_in_fig) // 6, -30)
    if ticks_major < -5:
        ticks_major = 5 * np.floor(ticks_major / 5)
    ax0.set_yticks([10**i for i in np.arange(0, np.log10(min_pval_in_fig), ticks_major)])
    ax0.set_ylabel('P-value', fontsize=FS)#, labelpad=2)
    ax0.tick_params(labelsize=FS, axis='y')

    ax0.tick_params(axis='x', bottom=False, labelbottom=False)

    ## add the Bonferroni line
    ax0.axhline(bonn_5_line, ls='--', color='red', lw=1)


    ########## BOTTOM: gene map ##########


    genes_df = pd.read_csv(FIGS_DIR.joinpath('contig_genes.csv'))

    # Create a dictionary for gene function categories and their colors
    cat_colors = {
        'C': 'orangered',
        'I': 'gold',
        'K': 'lightseagreen',
        'S': 'lightgrey',
        '-': 'lightgrey'
    }

    cat_labels = {
        'C': 'Energy production and conversion',
        'I': 'Lipid metabolism',
        'K': 'Transcription',
        'S': 'Unknown',
        '-': 'Unknown'
    }

    # Initialize the bottom subplot
    ax1 = fig.add_subplot(spec[5])

    # Loop through each row in the DataFrame
    for i, row in genes_df.iterrows():
        start_pos = row['start_pos']
        end_pos = row['end_pos']
        direction = 1 if row['strand']=='+' else -1
        color = cat_colors[row['best_og_cat']]
        ax1.arrow(start_pos, 0, direction * (end_pos - start_pos), 0,
                  length_includes_head=True, head_width=0.6, head_length=150, fc=color, ec=color, width=0.5)
        # ax1.text((start_pos + end_pos) / 2, 0.5, row['product'], ha='center', va='bottom', fontsize=FS)


    # Set the x-axis limit to the maximum and minimum values in the DataFrame
    x_max = max(snps_df['Position'].max(), genes_df[['start_pos', 'end_pos']].max().max()) + 100
    ax1.set_xlim(0, x_max)

    ax1.set_ylim(-.4, .4)

    # Remove the y-axis labels
    ax1.set_yticks([])
    ax1.spines[['top', 'left', 'right']].set_visible(False)

    # Share the x-axis between both subplots
    ax0.sharex(ax1)

    # Add a legend for the gene function categories
    legend_handles = [mpatches.Patch(color=cat_colors[cat], label=cat_labels[cat]) for cat in cat_colors]
    # ax1.legend(handles=legend_handles[:-1], fontsize=FS)

    ax1.tick_params(labelsize=FS, axis='x')
    ax1.set_xlabel('Position', fontsize=FS)


    ##### legends ax
    ax2 = fig.add_subplot(spec[6])
    ax2.legend(handles=legend_handles[:-1], fontsize=FS)
    ax2.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    ax2.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    # Show the plot
    plt.savefig(FIGS_DIR.joinpath(f"genes_map.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('ji')
    return

def get_contig_genes():
    genes_df = pd.read_csv('/net/mraid08/export/genie/LabData/Data/Annotations/Segal_annots/Segal_annots_2021_07_31_prokka_eggnog.csv')
    genes_df = genes_df.loc[(genes_df['SGB'] == 3066) & (genes_df['contig'] == 257)]
    genes_df.to_csv(FIGS_DIR.joinpath('contig_genes.csv'))
    return


if __name__ == '__main__':
    genome_plot()

