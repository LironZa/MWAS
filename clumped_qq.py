from LabData import config_global as config
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

MWAS_DIR = Path(config.analyses_dir).joinpath('20220402_234043_mwas_bmi_10K')


def draw_clumped_qq():
    ## load the full annotated_df
    df = pd.read_csv(MWAS_DIR.joinpath('snp_annotations', 'snp_annotations_ALL.csv'))[['Pval', 'post_clumping']]
    # df = pd.read_csv(MWAS_DIR.joinpath('snp_annotations', 'snp_annotations.csv'))[['Pval', 'post_clumping']]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    df = df.dropna()
    df['logp'] = -np.log10(df['Pval'])
    df = df.sort_values('logp', ascending=True)
    n = len(df)
    df['xx'] = -np.log10(np.linspace(1, 1/n, n))
    ax.set_xlabel('Expected (-log10)', fontdict={'fontsize': 8})
    ax.set_ylabel('Observed (-log10)', fontdict={'fontsize': 8})
    plt.plot(df.loc[df['post_clumping']!=1, 'xx'], df.loc[df['post_clumping']!=1, 'logp'], '.', color='black', alpha=.7)
    plt.plot(df.loc[df['post_clumping']==1, 'xx'], df.loc[df['post_clumping']==1, 'logp'], '.', color='crimson', alpha=.7)
    # plt.plot(xx[-len(pvals_log):], pvals_log, 'o')
    plt.plot([0, df['xx'].iloc[-1]], [0, df['xx'].iloc[-1]], '--')
    # plt.plot(xx, xx, '--')
    ax.tick_params(axis='both', labelsize=8)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.savefig(MWAS_DIR.joinpath("qq_plots", f"clumped_qq.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def all_species_qq_plot_speciesN(): ## the expected p-val distribution of each species is based on its own N
    ## load the full annotated_df
    df = pd.read_csv(MWAS_DIR.joinpath('snp_annotations', 'snp_annotations_ALL.csv'))[['Species', 'Pval']]
    # df = pd.read_csv(MWAS_DIR.joinpath('snp_annotations', 'snp_annotations.csv'))[['Species', 'Pval']]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    df = df.dropna()
    df['logp'] = -np.log10(df['Pval'])
    df = df.sort_values('logp', ascending=True)

    colors = plt.cm.get_cmap('hsv', len(df['Species'].unique()))

    i=0
    max_n = 1
    for _, spec_df in df.groupby('Species'):
        n = len(spec_df)
        spec_df['xx'] = -np.log10(np.linspace(1, 1/n, n))
        plt.plot(spec_df['xx'], spec_df['logp'],
                 'o-', color=colors(i), alpha=.25, ms=1)
        i += 1
        max_n = min(max_n, n)

    ax.set_xlabel('Expected (-log10)', fontdict={'fontsize': 8})
    ax.set_ylabel('Observed (-log10)', fontdict={'fontsize': 8})
    plt.plot([0, 5.5], [0, 5.5], '--', color='black')
    ax.tick_params(axis='both', labelsize=8)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.savefig(MWAS_DIR.joinpath("qq_plots", f"all_species_diffN.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return


def big_species_qq_plot_speciesN(): ## the expected p-val distribution of each species is based on its own N
    ## load the full annotated_df
    df = pd.read_csv(MWAS_DIR.joinpath('snp_annotations', 'snp_annotations_ALL.csv'), index_col=[0, 1, 2, 3])
    biggest_species = ['Rep_449', 'Rep_3086', 'Rep_3079', 'Rep_3076', 'Rep_3077', 'Rep_3066']
    df = pd.concat([df.xs(rep, level='Species', drop_level=False) for rep in biggest_species]).reset_index()[['Species', 'Pval']]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    df = df.dropna()
    df['logp'] = -np.log10(df['Pval'])
    df = df.sort_values('logp', ascending=True)

    colors = plt.cm.get_cmap('hsv', 7)

    from LabData.DataAnalyses.MBSNPs.taxonomy import taxonomy_df
    from LabData.DataAnalyses.MBSNPs.Plots.MWASPaper.mwas_paper_fig_utils import get_segal_species_label
    tax_df = taxonomy_df(level_as_numbers=False)
    tax_df = tax_df.set_index('SGB')

    i=0
    max_n = 1
    for rep_id , spec_df in df.groupby('Species'):
        n = len(spec_df)
        spec_df['xx'] = -np.log10(np.linspace(1, 1/n, n))
        species_name = get_segal_species_label(rep_id, tax_df)
        plt.plot(spec_df['xx'], spec_df['logp'],
                 'o-', color=colors(i), alpha=1, ms=1, label=species_name)
        i += 1
        max_n = min(max_n, n)

    ax.set_xlabel('Expected (-log10)', fontdict={'fontsize': 10})
    ax.set_ylabel('Observed (-log10)', fontdict={'fontsize': 10})
    plt.plot([0, 5.5], [0, 5.5], '--', color='black')
    ax.tick_params(axis='both', labelsize=10)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend(fontsize=10)
    plt.savefig(MWAS_DIR.joinpath("qq_plots", f"big_species_diffN.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return
