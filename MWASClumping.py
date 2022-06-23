from LabData import config_global as config
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from LabQueue.qp import fakeqp
from LabUtils.addloglevels import sethandlers
from LabData.DataAnalyses.MBSNPs.MWASInterpreter import remove_contig_parts

MWAS_DIR = Path(config.analyses_dir).joinpath('20220402_234043_mwas_bmi_10K')
BASE_DIR = Path(config.analyses_dir)
FIGS_DIR = BASE_DIR.joinpath('Figures', 'clumping')
FIGS_DIR.mkdir(parents=True, exist_ok=True)

CORRELATION_THRESH_P = .05


def correlate_snps(maf_df, snp1, snp2, species_id, corr_thresh_r, spearman=True):
    ## get the maf vectors of both snps
    ## remove samples in which one snp has maf == 255 (missing)
    snp1_maf = maf_df[('snps', species_id, snp1[0], snp1[1])]
    snp2_maf = maf_df[('snps', species_id, snp2[0], snp2[1])]
    good_inds = set(snp1_maf.loc[snp1_maf != 255].index).intersection(set(snp2_maf.loc[snp2_maf != 255].index))
    snp1_maf = snp1_maf.loc[good_inds]
    snp2_maf = snp2_maf.loc[good_inds]

    if spearman:
        ## calculate the correlation
        spr = spearmanr(snp1_maf, snp2_maf, nan_policy='raise')
        ## based on a correlation threshold, decide on whether the snps are correlated
        are_correlated = (abs(spr.correlation) >= corr_thresh_r) and (spr.pvalue <= CORRELATION_THRESH_P)

    else:
        prs = pearsonr(snp1_maf, snp2_maf)
        if (abs(prs[0]) >= corr_thresh_r) and (prs[1] <= CORRELATION_THRESH_P):
            are_correlated = True
        else:
            are_correlated = False

    return are_correlated


def clump_species(species_id, species_mwas_df, corr_thresh_r):
    ## load the MAF data
    maf_df = pd.read_hdf(MWAS_DIR.joinpath('mb_gwas_onlyMAF_data.h5'), key=species_id)

    species_mwas_df.reset_index(['Y', 'Species'], inplace=True)
    snps_l = list(species_mwas_df.index)
    correlations_counter = 0
    clumped_snps = {}
    while len(snps_l) > 0:
        this_snp = snps_l.pop(0)
        this_snp_correlated_snps = []

        ## correlate following snps and remove from the list snps that are correlated to this_snp
        other_snps_l = snps_l.copy()
        for other_snp in other_snps_l:
            ## correlate this_snp and other_snp
            correlations_counter += 1
            are_correlated = correlate_snps(maf_df, this_snp, other_snp, species_id, corr_thresh_r)

            ## if the snps are correlated: remove other_snp from the snps_l + add other snp to the list of this_snp_correlated_snps
            ## if the snps are not correlated, keep other_snp in the snps_l
            if are_correlated:
                snps_l.remove(other_snp)
                this_snp_correlated_snps.append(other_snp)

        ## add this_snp to the clumped df, together with a list of this_snp_linked_snps.
        ## first in a dict and later all clumped snps will become a df.
        clumped_snps[this_snp] = str(this_snp_correlated_snps)

    clumped_df = species_mwas_df.loc[clumped_snps.keys()].copy()
    clumped_df['correlated_snps'] = clumped_df.index.map(clumped_snps)
    clumped_df['n_correlated_snps'] = clumped_df['correlated_snps'].apply(lambda x: x.strip('[]').count('C_'))
    clumped_df = clumped_df.reset_index().set_index(['Y', 'Species', 'Contig', 'Position'])

    print(f"this species has {len(species_mwas_df)} snps. {correlations_counter} correlations calculated.")
    return clumped_df


def clump_mwas(corr_thresh_r=.9):
    ## load the results file mb_gwas.h5. reduce table to significant only, sort pvalues in ascending order
    mwas_df = pd.read_hdf(MWAS_DIR.joinpath('mb_gwas.h5'))
    mwas_df = mwas_df.loc[mwas_df['Global_Bonferroni'] <= .05].sort_values('Pval', ascending=True)

    clumped_results = pd.DataFrame()
    ## for each species: send the sub table for clumping
    for species_id, species_mwas_df in mwas_df.groupby('Species'):
        clumped_species_results = clump_species(species_id, species_mwas_df, corr_thresh_r)
        clumped_results = pd.concat([clumped_results, clumped_species_results])

    ## save the clumped results df
    clumped_results.to_csv(MWAS_DIR.joinpath(f'clumped_mb_gwas_{corr_thresh_r}.csv'))
    return


def annotate_clumped(corr_thresh_r, create_updownstream=False):
    ## load clumped
    clumped = pd.read_csv(MWAS_DIR.joinpath(f'clumped_mb_gwas_{corr_thresh_r}.csv')) #TODO: change if pearson
    clumped['Contig'] =  remove_contig_parts(clumped.Contig.values)
    clumped = clumped.set_index(['Y', 'Species', 'Contig', 'Position'])
    #TODO: for now removing this columns, may still need it later.
    clumped = clumped.drop(columns=['correlated_snps'])

    ## load annotated. Choose columns?
    annotated = pd.read_csv(MWAS_DIR.joinpath('snp_annotations', 'snp_annotations.csv'), index_col=[0, 1, 2, 3]).\
        drop(columns=['N', 'Coef', 'Pval', 'Global_Bonferroni', 'min_mn', 'maj_mn', 'min_n', 'maj_n', 'MajorAllele', 'MinorAllele',
                      'snp_desc', 'ID', 'seed_ortholog_evalue', 'seed_ortholog_score', 'IsSynonymous'])

    ## merge tables
    annotated_clumped = clumped.merge(annotated, left_index=True, right_index=True, how='inner', suffixes=('', '_ann'))

    ## save
    annotated_clumped.to_csv(MWAS_DIR.joinpath('snp_annotations', f'annotated_clumped_{corr_thresh_r}.csv'))


    ##### create an expanded annotation file -- including upstream/ downstream genes
    if create_updownstream:
        annotated_plus = pd.read_hdf(MWAS_DIR.joinpath('annotations', 'snps_gene_annotations.h5'))
        annotated_plus = annotated_plus.reset_index().set_index(['Y', 'Species', 'Contig', 'Position'])
        annotated_plus = annotated_plus.loc[clumped.index]
        annotated_plus.to_csv(MWAS_DIR.joinpath('snp_annotations', f'annotated_PLUS_clumped_{corr_thresh_r}.csv'))
    return


def add_clumping_info_to_results_tables(corr_thresh_r):
    ### to the 'mb_gwas.h5' df add the #correlated_snps column.
    ### to the annotated DFs add a more annoated column, to be used in visualizations:
    ### "post clumped" 0-excluded in clumping, 1-SNP passed the clumping, 2- non significant

    clumped = pd.read_csv(MWAS_DIR.joinpath(f'clumped_mb_gwas_{corr_thresh_r}.csv'))
    clumped = clumped.set_index(['Y', 'Species', 'Contig', 'Position'])

    clumped = clumped.reset_index()
    clumped['Contig'] =  remove_contig_parts(clumped.Contig.values)
    clumped = clumped.set_index(['Y', 'Species', 'Contig', 'Position'])

    ## update snp_annotations.csv
    annotated_df = pd.read_csv(MWAS_DIR.joinpath('snp_annotations', 'snp_annotations.csv'), index_col=[0,1,2,3])
    annotated_df = annotated_df.merge(clumped['n_correlated_snps'].to_frame('n_correlated_snps'),
                                      how='left', left_index=True, right_index=True)
    annotated_df.loc[annotated_df['n_correlated_snps'].notnull(), 'post_clumping'] = 1
    annotated_df.loc[annotated_df['n_correlated_snps'].isnull(), 'post_clumping'] = 0
    annotated_df.to_csv(MWAS_DIR.joinpath('snp_annotations', 'snp_annotations.csv'))

    ## update snp_annotations_ALL.csv
    annotated_df = pd.read_csv(MWAS_DIR.joinpath('snp_annotations', 'snp_annotations_ALL.csv'), index_col=[0,1,2,3])
    annotated_df = annotated_df.merge(clumped['n_correlated_snps'].to_frame('n_correlated_snps'),
                                      how='left', left_index=True, right_index=True)
    annotated_df.loc[annotated_df['n_correlated_snps'].notnull(), 'post_clumping'] = 1
    annotated_df.loc[annotated_df['Global_Bonferroni'] > .05, 'post_clumping'] = 2
    annotated_df.loc[annotated_df['post_clumping'].isnull(), 'post_clumping'] = 0
    annotated_df.to_csv(MWAS_DIR.joinpath('snp_annotations', 'snp_annotations_ALL.csv'))
    return

def create_annotated_linked_SNPs_df(corr_thresh_r):
    output_dir = MWAS_DIR.joinpath('snp_annotations', f'annotated_linked_SNPs_{corr_thresh_r}')
    output_dir.mkdir(exist_ok=True)

    ## load the clumped DF
    # NOTE: contigs have parts
    clumped = pd.read_csv(MWAS_DIR.joinpath(f'clumped_mb_gwas_{corr_thresh_r}.csv'))

    ## load the annotated SNPs DF (significant only, but w/o clumping)
    annotated = pd.read_csv(MWAS_DIR.joinpath('snp_annotations', 'snp_annotations.csv'), index_col=[0,1,2,3]). \
        drop(columns=['Coef', 'min_mn', 'maj_mn', 'min_n', 'maj_n',
                      'seed_eggNOG_ortholog', 'eggNOG OGs', 'MajorAllele', 'MinorAllele',
                      'ID', 'narr_og_name', 'narr_og_cat', 'best_og_name', 'best_og_cat', 'GOs', 'EC',
                      'KEGG_Module', 'KEGG_Reaction', 'KEGG_rclass', 'BRITE', 'KEGG_TC', 'CAZy', 'BiGG_Reaction',
                      'taxa', 'KEGG_ko', 'KEGG_Pathway',
                      'abs_means_diff', 'snp_desc', 'seed_ortholog_evalue', 'seed_ortholog_score', 'IsSynonymous'])

    ## for each SNP that survived the clumping and has [n_correlated_snps > 0], get the list of SNPs linked to it
    for _ , row in clumped.iterrows():
        if row['n_correlated_snps'] == 0:
            continue

        y = row['Y']
        species_id = row['Species']
        linked_list_txt = row['correlated_snps']
        linked_list_txt = linked_list_txt.strip('[]').split('(')
        linked_list_txt = [snp.strip('), ') for snp in linked_list_txt][1:]  # [1:] b/c the first item is ''
        linked_list = [(y, species_id, '_'.join(snp.split(', ')[0].strip("'").split('_')[:2]), int(snp.split(', ')[1]))
                       for snp in linked_list_txt]

        ## get the annotations of the SNPs from the annotated_df
        annotated_linked_snps = annotated.loc[linked_list]

        ## save an annotated SNPs df for all the SNPs linked to the current SNP (name file with the representative SNP's id)
        annotated_linked_snps.to_csv(output_dir.joinpath(
            f'{species_id}_{"_".join(row["Contig"].split("_")[:2])}_{row["Position"]}.csv'))

        ## add updownstream data?
    return


def hist_linkage_group_size(corr_thresh_r):
    clumped = pd.read_csv(MWAS_DIR.joinpath(f'clumped_mb_gwas_{corr_thresh_r}.csv'))

    group_sizes = clumped['n_correlated_snps']+1

    from LabData.DataAnalyses.MBSNPs.Plots.MWASPaper.mwas_paper_fig_utils import MWASPaperColors, phens_list
    import matplotlib.pyplot as plt
    c = MWASPaperColors()
    data_1_c = c.main_color_1
    DPI = 300
    FONTSIZE = 8
    TICK_FS = 7

    ### show the full range
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.4))
    _, bins, _ = ax.hist(group_sizes, bins=50, log=False, color=data_1_c, alpha=.8)
    ax.set_xlabel("Number of correlated SNPs in the group", fontsize=FONTSIZE)
    ax.set_ylabel('Number of groups', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.set_xlim(1, bins[-1])
    n_groups = len(clumped)
    print(f"median number of samples (N) per SNP: {group_sizes.median()}")
    plt.savefig(FIGS_DIR.joinpath(f"hist_snps_per_group__{n_groups}_groups_fullRange.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    ### now show only 1-100
    group_sizes = group_sizes.loc[group_sizes <= 100]
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.4))
    _, bins, _ = ax.hist(group_sizes, bins=np.arange(0,101,1), log=False, color=data_1_c, alpha=.8)
    ax.set_xlabel("Number of correlated SNPs in the group", fontsize=FONTSIZE)
    ax.set_ylabel('Number of groups', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.set_xlim(1, bins[-1])
    n_groups = len(clumped)
    print(f"median number of samples (N) per SNP: {group_sizes.median()}")
    plt.savefig(FIGS_DIR.joinpath(f"hist_snps_per_group__{n_groups}_groups_upTo100.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return
