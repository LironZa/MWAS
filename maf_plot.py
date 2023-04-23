from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import mannwhitneyu
from LabData import config_global as config
from LabData.DataAnalyses.MBSNPs.MWASInterpreter import remove_contig_parts
from LabData.DataAnalyses.MBSNPs.taxonomy import taxonomy_df
from LabData.DataLoaders.MBSNPLoader import MAF_1_VALUE
from LabData.DataAnalyses.MBSNPs.Plots.MWASPaper.mwas_paper_fig_utils import MWASPaperColors, create_snp_label, \
    phen_labels_expanded, phen_units_dict, get_segal_species_label, round_pval


BASE_DIR = Path(config.analyses_dir)
FIGS_DIR = BASE_DIR.joinpath('Figures', 'snps_maf_plots')
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# the pre-made samples * features (phenotypes etc) table used for the MWAS
ROBUST_SAMPLES_DF_P = BASE_DIR.joinpath('MWAS_robust_samples_list/BasicMWAS_10K_singleSamplePerReg_20223003_234555.csv')
# species composition for the relevant samples
MB_ABUNDANCE_DF_P = BASE_DIR.joinpath('MWAS_robust_samples_list/BasicMWAS_10K_singleSamplePerReg_20223003_234555_mb_species_abundance.csv')

c = MWASPaperColors()
c.set_bplot_params()

red = c.red
bplotdots = c.grey
FIGSIZE = (4.7, 2.7)
ONE_SNP_BIN_FIGSIZE = (1.5, 2.7)
ONE_SNP_BIN_PLUS_ABSENT_FIGSIZE = (2, 2.7)
FONTSIZE = 8
TICK_FS = 6
DPI = 300
TOTAL_NUMBER_OF_HYPOTHSES = 120

N_BINS = 10



def plt_bin_maf_con_phen(snp_df_gr, output_dir, y, z=None, snp_label=None, phen=None, suff="", plot_dots=False,
                         samples_inds='RegistrationCode'):
    spcs, contig, pos = snp_df_gr[0]
    df = snp_df_gr[1].reset_index().set_index(samples_inds)

    fig, ax = plt.subplots(1, 1, figsize=ONE_SNP_BIN_FIGSIZE)
    ax.grid(False)

    data = [df.loc[df['MAF'] >= .99*MAF_1_VALUE, y], df.loc[df['MAF'] <= .01*MAF_1_VALUE, y]]
    labels = ['major allele\nN={}'.format(len(data[0])), 'alternative\nN={}'.format(len(data[1]))]
    medians = [d.median() for d in data]

    if plot_dots:
        for i in range(2):
            dot_y = data[i]
            dot_x = np.random.normal(1 + i, .05, size=len(dot_y))
            ax.plot(dot_x, dot_y, '.', c=bplotdots, alpha=.15, ms=7)

    plt.boxplot(data, labels=labels, notch=True, bootstrap=1000, showfliers=False, whis=(5, 95), widths=.35)
    try:
        ax.set_ylabel(phen_labels_expanded[y], fontsize=FONTSIZE)
        ax.annotate('', xy=(1.5, medians[0]), xytext=(1.5, medians[1]), color=c.grey,
                    arrowprops=dict(arrowstyle='|-|', shrinkA=0, shrinkB=0, mutation_scale=.6, color=c.grey))
        ax.text(x=1.55, y=(medians[0] + .5 * (medians[1] - medians[0])), s=f'{np.abs(medians[1] - medians[0]):.2g} {phen_units_dict[phen]}',
                size=TICK_FS, rotation='vertical', ha='left', va='center', color=c.grey)

    except KeyError:
        ax.set_ylabel(y, fontsize=FONTSIZE)

    ax.tick_params(axis='both', labelsize=TICK_FS)
    if snp_label is not None:
        ax.set_title(snp_label, fontsize=FONTSIZE)

    plt.savefig(output_dir.joinpath(f'2boxes_{spcs}_{contig}_{pos}_{y}_{suff}.png'),
                dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def plt_binary_maf_plus_absent_species(snp_df_gr, y, output_dir=None, ax=None, snp_label=None, phen=None, suff="",
                                       plot_dots=False, title_fs=FONTSIZE, ticks_fs=TICK_FS, label_y=True,
                                       samples_inds='RegistrationCode',
                                       robust_table_p=ROBUST_SAMPLES_DF_P, mb_table_p=MB_ABUNDANCE_DF_P, stats=False):
    spcs, contig, pos = snp_df_gr[0]
    df = snp_df_gr[1].reset_index().set_index(samples_inds)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=ONE_SNP_BIN_PLUS_ABSENT_FIGSIZE)
    ax.grid(False)

    ##  y data for the major & minor allele groups (the standard maf plot data)
    majors_data = df.loc[df['MAF'] >= .99*MAF_1_VALUE, y]
    minors_data = df.loc[df['MAF'] <= .01*MAF_1_VALUE, y]

    ##  y data for the cohort participants who don't have this species
    phenotypes_df = pd.read_csv(robust_table_p, index_col=0)
    mb_df = pd.read_csv(mb_table_p, index_col=0)
    no_species_samples = mb_df.loc[mb_df[spcs] == 1e-4].index
    no_species_data = phenotypes_df.loc[no_species_samples, y].dropna()


    data = [no_species_data, majors_data, minors_data]
    labels = [f'without\nspecies\nN={len(data[0])}', f'major\nallele\nN={len(data[1])}', f'alternative\nallele\nN={len(data[2])}']
    medians = [d.median() for d in data]

    print(phen, len(no_species_data)+len(df))

    if plot_dots:
        for i in range(3):
            dot_y = data[i]
            dot_x = np.random.normal(1 + i, .05, size=len(dot_y))
            ax.plot(dot_x, dot_y, '.', c=bplotdots, alpha=.15, ms=7)

    plt.rc('boxplot.medianprops', color=red, linewidth=1.2)
    plt.rc('boxplot.boxprops', color='black', linewidth=1.2)
    bps = ax.boxplot(data, labels=labels, notch=True, bootstrap=1000, showfliers=False, whis=(5, 95), widths=.35)

    if stats:
        ##### add pairwise comparisons statistics
        ## compute pvals (Mann-Whitney)
        no_vs_maj = mannwhitneyu(no_species_data, majors_data).pvalue
        no_vs_min = mannwhitneyu(no_species_data, minors_data).pvalue
        maj_vs_min = mannwhitneyu(majors_data, minors_data).pvalue

        ## add lines and pvals
        def add_stats(pval, left, right, height, bonferroni_factor=TOTAL_NUMBER_OF_HYPOTHSES):
            if bonferroni_factor is not None:
                pval = pval * bonferroni_factor
                pval = round_pval(pval, p_thresh=.05)
                pval = pval.replace('p', 'q')
            else:
                pval = round_pval(pval, p_thresh=.05)
            ax.annotate('', xy=(left, height), xytext=(right, height), color='black', xycoords=('data', 'axes fraction'),
                        arrowprops=dict(arrowstyle='-', color='black'))
            ax.text(x=(left + .5 * (right-left)), y= (height + .005), s=pval, transform=ax.get_xaxis_transform(),
                    size=ticks_fs-1, rotation='horizontal', ha='center', va='bottom', color='black')

        ylim0, ylim1 = ax.get_ylim()
        ax.set_ylim(ylim0, 40)
        add_stats(no_vs_maj, 1, 1.95, 0.8)
        add_stats(maj_vs_min, 2.05, 3, 0.8)
        add_stats(no_vs_min, 1, 3, 0.9)


    #### add the grey annotation for maj-min difference
    try:
        if label_y==True:
            ax.set_ylabel(phen_labels_expanded[y], fontsize=title_fs)
        ax.annotate('', xy=(2.5, medians[1]), xytext=(2.5, medians[2]), color=c.grey,
                    arrowprops=dict(arrowstyle='|-|', shrinkA=0, shrinkB=0, mutation_scale=.6, color=c.grey))
        ax.text(x=2.55, y=(medians[1] + .5 * (medians[2] - medians[1])), s=f'{np.abs(medians[2] - medians[1]):.2g} {phen_units_dict[phen]}',
                size=ticks_fs, rotation='vertical', ha='left', va='center', color=c.grey)

    except KeyError:
        if label_y==True:
            ax.set_ylabel(y, fontsize=title_fs, pad=3)

    ax.tick_params(axis='both', labelsize=ticks_fs)
    if snp_label is not None:
        ax.set_title(snp_label, fontsize=title_fs, pad=3)

    if ax is None:
        plt.savefig(output_dir.joinpath(f'3boxes_{spcs}_{contig}_{pos}_{y}_{suff}.png'),
                    dpi=DPI, bbox_inches='tight')
        plt.close(fig)
    return


######################################################################

def fix_data_vecs_df(data_vecs_df, samples_inds='SampleName'):
    ### NOTE: this is a temporary fix, since collect_data has gone crazy since the 'compact' mode
    ## (the normal mode duplicates the positions to both index and columns)
    inx = data_vecs_df.index.names
    data_vecs_df = data_vecs_df.reset_index(level='Position')
    data_vecs_df['MAF'] = data_vecs_df.apply(lambda x: x[int(x['Position'])], axis=1)
    data_vecs_df = data_vecs_df.reset_index().set_index(inx)
    data_vecs_df = data_vecs_df.drop(columns=[int(pos) for pos in data_vecs_df.index.get_level_values('Position').unique()])

    data_vecs_df.rename(columns={'y': data_vecs_df.index.unique('Y')[0]}, inplace=True)
    data_vecs_df = data_vecs_df.reset_index()
    data_vecs_df.Contig = remove_contig_parts(data_vecs_df.Contig.values)
    data_vecs_df.Position = data_vecs_df.Position.astype(int)
    data_vecs_df = data_vecs_df.set_index(['Species', 'Contig', 'Position', samples_inds])
    return data_vecs_df

######################################################################

def plot_top_four_snps(phen, workdir, data_file_suffix='', samples_inds='SampleName', by='mixed'): #TODO: by = 'pval' / 'mixed'
    output_dir = workdir.joinpath(f'maf_plots/')
    output_dir.mkdir(parents=True, exist_ok=True)

    y = phen

    # data to plot
    data_vecs_df = pd.read_hdf(workdir.joinpath(f'mb_gwas_{data_file_suffix}_data.h5'))
    data_vecs_df = fix_data_vecs_df(data_vecs_df, samples_inds=samples_inds)

    # SNPs annotations
    annots_df = pd.read_csv(workdir.joinpath('snp_annotations', 'annotated_clumped_0.3.csv')).set_index(['Species', 'Contig', 'Position'])
    annots_df = annots_df.loc[data_vecs_df.index.droplevel([samples_inds]).drop_duplicates()]

    if by == 'pval':
        ###  choose top 4 SNPs to plot based on p-value
        annots_df = annots_df.sort_values('Pval', ascending=True).head(4)
    if by == 'mixed':
        ###  choose top 4 SNPs to plot based on p-value + effect size
        annots_df_pval = annots_df.sort_values('Pval', ascending=True).head(2)
        annots_df_effect = annots_df.sort_values('abs_means_diff', ascending=False).head(2)
        annots_df = pd.concat([annots_df_pval, annots_df_effect])

    order = {annots_df.index[i]:i for i in range(len(annots_df))}

    data_vecs_df = data_vecs_df.reset_index().set_index(['Species', 'Contig', 'Position']).loc[annots_df.index]
    data_vecs_df = data_vecs_df.reset_index().set_index(['Species', 'Contig', 'Position', samples_inds])

    tax_df = taxonomy_df(level_as_numbers=False)
    tax_df = tax_df.set_index('SGB')

    fig, axs = plt.subplots(1, 4, figsize=(8.3, 2.7), gridspec_kw={'wspace': .5})

    stats_annots = True
    for rep, gr in data_vecs_df.groupby(['Species', 'Contig', 'Position']):  #rep = (species, contig, pos)
        species_name = get_segal_species_label(rep[0], tax_df)
        snp_label = create_snp_label(annots_df.loc[rep], species_cont_pos=rep, taxonomy=species_name,
                                     gene_annot_col='gene') #TODO: 'gene' / 'product'?
        plt_binary_maf_plus_absent_species((rep, gr), y=y, ax=axs[order[rep]], phen=phen, title_fs=6,
                                           snp_label=snp_label, samples_inds=samples_inds, stats=stats_annots)

    if stats_annots:
        plt.savefig(output_dir.joinpath(f'Top_four_SNPs_by_{by}_annot_fs.png'), dpi=DPI, bbox_inches='tight')
    else:
        plt.savefig(output_dir.joinpath(f'Top_four_SNPs_by_{by}.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return


def plot_all_clumped_snps(phen, workdir, data_file_suffix='', samples_inds='SampleName'):
    output_dir = workdir.joinpath(f'maf_plots/')
    output_dir.mkdir(parents=True, exist_ok=True)
    y = phen

    # data to plot
    data_vecs_df = pd.read_hdf(workdir.joinpath(f'mb_gwas_{data_file_suffix}_data.h5'))
    data_vecs_df = fix_data_vecs_df(data_vecs_df, samples_inds=samples_inds)

    # SNPs annotations
    annots_df = pd.read_csv(workdir.joinpath('snp_annotations', 'annotated_clumped_0.3.csv')).set_index(['Species', 'Contig', 'Position'])
    annots_df = annots_df.loc[data_vecs_df.index.droplevel([samples_inds]).drop_duplicates()]

    # take only the post-clumping SNPs. Sort by effect size?
    annots_df = annots_df.sort_values('abs_means_diff', ascending=False)
    assert len(annots_df) == 40
    data_vecs_df = data_vecs_df.reset_index().set_index(['Species', 'Contig', 'Position']).loc[annots_df.index]
    data_vecs_df = data_vecs_df.reset_index().set_index(['Species', 'Contig', 'Position', samples_inds])

    tax_df = taxonomy_df(level_as_numbers=False)
    tax_df = tax_df.set_index('SGB')

    fig, axs = plt.subplots(8, 5, figsize=(8.25, 11.75), gridspec_kw={'wspace': .15, 'hspace': .25}, sharey=True)
    axs = axs.reshape(1, -1)

    stats_annots = True
    for i, gr in enumerate(data_vecs_df.groupby(['Species', 'Contig', 'Position'])):
        species_name = get_segal_species_label(gr[0][0], tax_df)
        # snp_label = create_snp_label(annots_df.loc[gr[0]], species_cont_pos=gr[0], taxonomy=species_name,
        #                              gene_annot_col='gene')
        snp_label = f"[{gr[0][0]}, {gr[0][1]}, pos {gr[0][2]}]"
        plt_binary_maf_plus_absent_species(gr, y=y, ax=axs[0][i], phen=phen, title_fs=6, ticks_fs=5, label_y=False,
                                           snp_label=snp_label, samples_inds=samples_inds, stats=stats_annots)
        axs[0][i].tick_params(axis='x', bottom=False, labelbottom=False)

    if stats_annots:
        plt.savefig(output_dir.joinpath(f'All_40_clumped__SNPs_stats.png'), dpi=DPI, bbox_inches='tight')
    else:
        plt.savefig(output_dir.joinpath(f'All_40_clumped__SNPs.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return



