import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from LabData import config_global as config
from LabData.DataAnalyses.MBSNPs.Plots.MWASPaper.mwas_paper_fig_utils import phens_list, MWASPaperColors, phens_labels, round_pval, \
    phen_units_dict, choose_top_snps, compute_snps_effect_size, mid_sentence_labels, phen_labels_expanded
from LabData.DataLoaders.MBSNPLoader import MAF_1_VALUE
from LabQueue.qp import fakeqp
from LabUtils.addloglevels import sethandlers
from UseCases.DataAnalyses.MBSNP_MWAS.mwas_liron import PARAMS_DICT
from matplotlib import ticker

BASE_DIR = Path(config.analyses_dir)
phens_l = phens_list
OUTPUT_DIR = BASE_DIR.joinpath('Figures', 'volcano_plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

c = MWASPaperColors()

DPI = 300
PPT = False

if PPT:
    FONTSIZE = 12
    ANNOT_FS = 8
    FIG_SIZE = (8, 4)
else:
    FONTSIZE = 8
    ANNOT_FS = 6
    FIG_SIZE = (4, 3)


def add_snps_annots(ax, phen, all_xys=None, num_annots=8, xby='effect'):
    annot_df = pd.read_csv(BASE_DIR.joinpath('20220402_234043_mwas_bmi_10K', 'snp_annotations', 'snp_annotations.csv'),
                           index_col=[0, 1, 2, 3])
    annot_df = annot_df.loc[((annot_df['GeneDistance'] > -1) & annot_df['gene'].notnull())].reset_index()

    # choose genes by top pval, choose genes by top coef, and unite
    if xby == 'coef':
        top_ps = choose_top_snps(annot_df, opt_by='Pval', n=int(num_annots/2))
        top_cs = choose_top_snps(annot_df, opt_by='Coef', n=int(num_annots/2))
        top_snps = pd.concat([top_ps, top_cs])
        x_col = 'Coef'
        top_cs.Coef = top_cs.Coef.values * MAF_1_VALUE  # NOTE: if using, compare to history and adapt, this may be wrong.
    elif xby == 'effect':
        top_ps = choose_top_snps(annot_df, opt_by='Pval', n=int(num_annots/2))
        top_cs = choose_top_snps(annot_df, opt_by='diff', n=int(num_annots/2))
        top_snps = pd.concat([top_ps, top_cs])
        x_col = 'means_diff'
    elif xby == 'clump_effect':
        top_snps = choose_top_snps(annot_df, opt_by='clump', n=int(num_annots))
        x_col = 'means_diff'

    snps_annots = top_snps[['Species', 'Contig', 'Position', 'Pval', x_col, 'gene']]\
        .drop_duplicates(subset=['Species', 'gene'], keep='first')
    print(snps_annots)

    # move around labels to not overlap
    from adjustText import adjust_text
    tags = [ax.text(snp_row[x_col], snp_row['Pval'], snp_row['gene'], fontsize=ANNOT_FS, color=c.red, weight="bold")
            for _, snp_row in snps_annots.iterrows()]
    adjust_text(tags, x=all_xys[0], y=all_xys[1], arrowprops={'arrowstyle': '-', 'color': c.red})
    return


def draw_mwas_volcano_plot(df, output_file=None, phen=None, bonn_5_thresh=None, n=None, annot_snps=False):
    """Draws a scatter plot of the dataframe's effect size against its p-values.
    :param df: Data to draw.
    :param output_file: Path to save the image at.
    :param n: Consider the best n p-values. For performance.
    :param scale: Scale of the image. Larger scale -> larger image.
    :param title: Use a non-default title for the plot.
    """
    plt.rc('font', size=FONTSIZE)

    print(f"{phen} - there are {(df['Pval']==0).sum()} zero pvals to replace")
    df['Pval'] = df['Pval'].replace(to_replace=0, value=1e-300)
    if n is not None and len(df) > n:
        df = df.sort_values('Pval')[:n]
    df = df.sort_values('N')
    df = compute_snps_effect_size(df)
    pvals = df.Pval.values
    # coefs = df.Coef.values * MAF_1_VALUE
    effects = df['min_mn'].sub(df['maj_mn'], axis='index').values
    ns = df.N.values

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
    ax.grid(which='both', axis='both', b=False)

    scat = ax.scatter(effects, pvals, c=ns, alpha=.9, cmap=c.create_greys_cmap(), edgecolor=None, s=FONTSIZE-2)

    cbar = plt.colorbar(mappable=scat,
                        orientation='vertical', fraction=.05, pad=.01, ax=ax,
                        format=ticker.ScalarFormatter(useMathText=True))
    cbar.set_label('samples', fontsize=FONTSIZE)

    ax.set_yscale('log')
    min_pval = pvals.min()
    min_pval_in_fig = min_pval * 10 ** (0.1 * np.log10(min_pval))  # Add 10% white space above
    ax.set_ylim([1, min_pval_in_fig])

    ticks_major = max(np.log10(min_pval_in_fig) // 6, -30)
    ticks_major = 5 * np.floor(ticks_major / 5)
    ax.set_yticks([10**i for i in np.arange(0, np.log10(min_pval_in_fig), ticks_major)])
    # ax.set_yticks([10**i for i in np.arange(0, np.log10(pvals.min()) - 10, -30)])

    if phen is not None and phen in phen_units_dict.keys():
        ax.set_xlabel(f"Average {mid_sentence_labels[phen_labels_expanded[phen]]} difference between alleles\n"
                      f"(alternative allele mean - major allele mean; {phen_units_dict[phen]})", fontsize=FONTSIZE)
    else:
        ax.set_xlabel('difference in means', fontsize=FONTSIZE)
    ax.set_ylabel('P-value', fontsize=FONTSIZE)
    if bonn_5_thresh is not None:
        max_x = max(np.abs(effects[pvals < bonn_5_thresh].min()), np.abs(effects[pvals < bonn_5_thresh].max()))
        ax.set_xlim([-1.1 * max_x, 1.1 * max_x])
        ax.axhline(bonn_5_thresh, ls='--', color=c.red, lw=1)

    if annot_snps:
        add_snps_annots(ax, phen, all_xys=(effects, pvals), num_annots=50, xby='clump_effect')

    ax.tick_params(axis='both', labelsize=FONTSIZE)

    if output_file is None:
        if annot_snps:
            output_file = OUTPUT_DIR.joinpath(f'{phen}_volcano_annot_half.png')
        else:
            output_file = OUTPUT_DIR.joinpath(f'{phen}_volcano.png')

    # plt.pause(1e-13)
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    mwas_df = pd.read_hdf(BASE_DIR.joinpath('20220402_234043_mwas_bmi_10K', 'mb_gwas.h5'))
    draw_mwas_volcano_plot(mwas_df, output_file=None, phen='bmi', bonn_5_thresh=0.05 / len(mwas_df), n=20000000, annot_snps=True)