from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from LabData import config_global as config
from LabData.DataAnalyses.MBSNPs.Plots.MWASPaper.mwas_paper_fig_utils import MWASPaperColors, phens_list
from LabData.DataLoaders.MBSNPLoader import MAF_1_VALUE
from UseCases.DataAnalyses.MBSNP_MWAS.mwas_liron import PARAMS_DICT

BASE_DIR = Path(config.analyses_dir)
FIGS_DIR = BASE_DIR.joinpath('Figures', 'all_snps_overview')
FIGS_DIR.mkdir(parents=True, exist_ok=True)

c = MWASPaperColors()

data_1_c = c.main_color_1
DPI = 300
FIGSIZE = (3.6, 2.4)
FONTSIZE = 8
TICK_FS = 7
MIN_SAMPLES_PER_SNP = 1000

COLS_annotated = ['Species', 'Contig', 'Position', 'N', 'GeneDistance', 'NonSymMutation'] # , 'Frac'
COLS_original = ['Species', 'Contig', 'Position', 'N']


#######################

def hist_snps_per_species(full_snps_df):
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    _, bins, _ = ax.hist(full_snps_df.groupby('Species').size(), bins=50, log=True, color=data_1_c, alpha=.8)
    ax.set_xlabel("Number of SNPs per species", fontsize=FONTSIZE)
    ax.set_ylabel('Number of species', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.set_xlim(-1, bins[-1])
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    n_species = len(full_snps_df.groupby('Species').first())
    plt.savefig(FIGS_DIR.joinpath(f"hist_snps_per_species____{n_species}_species.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Annotated SNPs: {len(full_snps_df)} SNPs, in {len(full_snps_df.groupby('Species').first())} species")
    print(f"median number of SNPs per species: {full_snps_df.groupby('Species').size().median()}")
    print(f"number of species with over 100,000 SNPs: {(full_snps_df.groupby('Species').size() >= 100000).sum()}")
    return


def hist_N_per_snp(full_snps_df):
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    _, bins, _ = ax.hist(full_snps_df['N'], bins=50, log=True, color=data_1_c, alpha=.8)
    ax.set_xlabel("Number of samples", fontsize=FONTSIZE)
    ax.set_ylabel('Number of SNPs', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.set_xlim(MIN_SAMPLES_PER_SNP, bins[-1])
    n_snps = len(full_snps_df)
    print(f"median number of samples (N) per SNP: {full_snps_df['N'].median()}")
    plt.savefig(FIGS_DIR.joinpath(f"hist_N_per_snp____{n_snps}_snps.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return


def bar_sig_per_species(full_snps_df):
    sig_snps_df = full_snps_df.loc[full_snps_df['Global_Bonferroni'] <= .05]

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 3))
    snps_per_species = sig_snps_df.groupby('Species').size().sort_values(ascending=False)
    print(snps_per_species)
    ax.bar(range(len(snps_per_species)), snps_per_species, tick_label=snps_per_species.index.values, log=True, color=data_1_c, alpha=.8)
    ax.set_ylabel('Number of BMI-associated SNPs', fontsize=TICK_FS)
    ax.set_xlim(-0.5, len(snps_per_species) - .05)
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    # from matplotlib import ticker
    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y) if y<100 else ))
    ax.tick_params(axis='y', labelsize=TICK_FS)
    ax.tick_params(axis='x', labelsize=TICK_FS, bottom=False)
    plt.xticks(rotation='vertical')

    #### add the number of post-clumping SNPs
    post_clump_counts = pd.read_csv(BASE_DIR.joinpath('Figures', 'first_clumping_counts.csv'), index_col=[0])  ## count de novo?
    ## order the same
    post_clump_counts = post_clump_counts.loc[snps_per_species.index]
    ## add numbers to the plot
    for i, rep in enumerate(snps_per_species.index):
        ax.text(x=i, y=1, s=post_clump_counts.loc[rep, 'first_clumping'], va='top', ha='center', color='white', size=TICK_FS)
        ax.text(x=i, y=snps_per_species.loc[rep], s=snps_per_species.loc[rep], va='bottom', ha='center', color='black', size=TICK_FS)

    plt.savefig(FIGS_DIR.joinpath(f"barplot_significant_snps_per_species.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    ### another plot for tested SNPs by species, same species (and same order)
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 3))
    sig_snps_per_species = snps_per_species
    snps_per_species = full_snps_df.groupby('Species').size()
    snps_per_species = snps_per_species.loc[sig_snps_per_species.index]
    print(snps_per_species)
    ax.bar(range(len(snps_per_species)), snps_per_species, tick_label=snps_per_species.index.values, log=True, color=data_1_c, alpha=.8)
    ax.set_ylabel('Number of tested SNPs', fontsize=TICK_FS)
    ax.set_xlim(-0.5, len(snps_per_species) - .05)
    ax.tick_params(axis='y', labelsize=TICK_FS)
    ax.tick_params(axis='x', labelsize=TICK_FS, bottom=False)
    plt.xticks(rotation='vertical')

    ## add numbers to the plot
    for i, rep in enumerate(snps_per_species.index):
        ax.text(x=i, y=snps_per_species.loc[rep], s=snps_per_species.loc[rep], va='bottom', ha='center', color='black', size=TICK_FS)

    plt.savefig(FIGS_DIR.joinpath(f"barplot_tested_snps_per_species.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    return
