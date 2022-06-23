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



