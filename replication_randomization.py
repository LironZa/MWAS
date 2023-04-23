import os
from pathlib import Path
import pandas as pd
import numpy as np
import random
from LabData import config_global as config
from LabData.DataAnalyses.MBSNPs.MWAS import MWAS, CommonParams
from LabUtils.addloglevels import sethandlers
## take from the original MWAS the loaders for microbiome samples and for the covariates
from UseCases.DataAnalyses.MBSNP_MWAS.MWASPaper.mwas_robust_samples_list_10K import TABLES_DIR, robust_features_loader, \
    robust_samples_loader
from UseCases.DataAnalyses.MBSNP_MWAS.MWASPaper.contig_parts_update import match_contigs
from LabData.DataAnalyses.MBSNPs.MWASInterpreter import remove_contig_parts
from UseCases.DataAnalyses.MBSNP_MWAS.MWASPaper.mwas_replication_w_Lifelines import get_snps_set, lifelines_species_covariate_loader
import matplotlib.pyplot as plt
from LabData.DataAnalyses.MBSNPs.Plots.MWASPaper.mwas_paper_fig_utils import MWASPaperColors
from LabData.DataLoaders.MBSNPLoader import MAF_1_VALUE

NUM_SNPS = 40
NUM_REPS = 1000

BASE_DIR = Path(config.analyses_dir)
MWAS_DIR = BASE_DIR.joinpath('20220402_234043_mwas_bmi_10K')

TABLES_DIR = BASE_DIR.joinpath('MWAS_robust_samples_list')
DATA_DF_P = TABLES_DIR.joinpath('lifelines_data_20222010_140759.csv')
MB_COMPOSITION_DF_P = TABLES_DIR.joinpath('lifelines_data_20222010_140759_mb_species_abundance.csv')

# create a dir for the random SNPs lists, and a dir for the MWAS results
ANALYSIS_DIR = MWAS_DIR.joinpath('Replication_Randomization')
RAND_SNPS_LISTS_DIR = ANALYSIS_DIR.joinpath('random_snps_lists')
RAND_SNPS_LISTS_DIR.mkdir(parents=True, exist_ok=True)
RAND_MWAS_RESULTS_DIR = ANALYSIS_DIR.joinpath('random_mwas_results')
RAND_MWAS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIGS_DIR = BASE_DIR.joinpath('Figures', 'replication_randomization')
FIGS_DIR.mkdir(parents=True, exist_ok=True)

c = MWASPaperColors()



############################ Random SNPs list ################################################################

# sample 40 out of the 12M tested SNPs. Run 1000 times linearly to only load the 12M SNPs table once.
def create_random_SNPs_lists(num_snps=NUM_SNPS, num_reps=NUM_REPS):
    # load the full tested SNPs table
    mwas_df = pd.read_hdf(MWAS_DIR.joinpath('mb_gwas.h5'))

    # in each rep, choose a random pick of 40 (Species, Contig, Position) triads, and save as a table with [rep] in the name
    for i in range(num_reps):
        random_indices = random.sample(list(mwas_df.index), num_snps)
        random_snps_df = mwas_df.loc[random_indices, ['N', 'Coef', 'Pval']]
        random_snps_df.to_csv(RAND_SNPS_LISTS_DIR.joinpath(f"random_snps_set_{i}.csv"))
    return


# for the 40 SNPs, update the new contig parts. Run 1000 times in parallel using the queue.
def update_contig_parts(num_reps=NUM_REPS):
    sethandlers(file_dir=config.log_dir)
    os.chdir('/net/mraid08/export/genie/LabData/Analyses/lironza/jobs')
    with config.qp(jobname='CONTG', q=['himem7.q'], _trds_def=2, _mem_def='5G', _tryrerun=True, max_r=300) as q:
    # with fakeqp(jobname='CONTG', q=['himem7.q'], _trds_def=2, _mem_def='5G', _tryrerun=True, max_r=300) as q:
        q.startpermanentrun()
        waiton = []

        for i in range(num_reps):
            old_csv_p = new_csv_p = RAND_SNPS_LISTS_DIR.joinpath(f"random_snps_set_{i}.csv")
            # load the random_snps_set table, update contigs parts, and resave.
            # match_contigs(old_csv_p, new_csv_p)
            waiton.append(q.method(match_contigs, (old_csv_p, new_csv_p)))

        q.wait(waiton)
    return


########################## MWAS #######################################
def y_gen_f(subjects_df, species_set):
    return robust_features_loader(['bmi'], DATA_DF_P, subjects_df)

def covariate_gen_f(subjects_df):
    return robust_features_loader(['age', 'gender'], DATA_DF_P, subjects_df)

class Rand_Rep_BMI_Params(CommonParams):
    samples_set = robust_samples_loader(DATA_DF_P)
    jobname = 'RanRep'

    min_reads_per_snp = 1
    max_on_fraq_major_per_snp = 1
    max_on_most_freq_val_in_col = 1
    min_subjects_per_snp = 1 # ORIGINALLY WAS 1000
    min_on_minor_per_snp = 1 # ORIGINALLY WAS 50
    min_positions_per_sample = 0
    filter_by_species_existence = False
    max_pval_to_detailed = 1
    largest_sample_per_user = False
    subsample_dir = ''

    max_jobs = 400
    species_blocks = 40
    send_to_queue = True

    subjects_loaders = ['SubjectLoader']
    subjects_get_data_args = {'groupby_reg': 'first', 'study_ids': [32]}
    y_gen_f = staticmethod(y_gen_f)
    covariate_gen_f = staticmethod(covariate_gen_f)
    species_specific_cov_f = staticmethod(lifelines_species_covariate_loader)

    ret_cov_fields = True
    output_cols = ['N', 'Coef', 'Pval']


# run MWAS with the lifelines robust sample table and the random list of 40 SNPs
# save minimum output files, and create a unique file name for the results
def run_one_mwas(i):
    params = Rand_Rep_BMI_Params()
    params.snp_set = get_snps_set(RAND_SNPS_LISTS_DIR.joinpath(f"random_snps_set_{i}.csv"))
    params.results_file_suffix = f"_{i}"
    m = MWAS(params, work_dir=RAND_MWAS_RESULTS_DIR)
    m.gen_mwas()
    return

def run_mwases(num_reps=NUM_REPS):

    sethandlers(file_dir=config.log_dir)
    os.chdir('/net/mraid08/export/genie/LabData/Analyses/lironza/jobs')
    with config.qp(jobname='RanMWS', q=['himem7.q'], _trds_def=2, _mem_def='5G', _tryrerun=True, max_r=400) as q:
    # with fakeqp(jobname='CONTG', q=['himem7.q'], _trds_def=2, _mem_def='5G', _tryrerun=True, max_r=300) as q:
        q.startpermanentrun()
        waiton = []
        for i in range(num_reps):
            waiton.append(q.method(run_one_mwas, (i,)))
        q.wait(waiton)

    # for i in range(num_reps):
    #     run_one_mwas(i)
    return


######################## Analysis #######################################################

# create a results summary table.
# Go over each MWAS output df, and count the number of tested (40?) vs significantly associated SNPs (in the right direction!!)
def summarize_mwas_results(num_reps=NUM_REPS):
    # create a summary df
    summary = pd.DataFrame(columns=['tested', 'signif_any', 'signif_right_sign'], index=range(num_reps))

    original = pd.read_hdf(MWAS_DIR.joinpath('mb_gwas.h5'))
    original = original[['N', 'Coef', 'Pval']]
    original.reset_index(inplace=True)
    original.Contig = remove_contig_parts(original.Contig.values)
    original.set_index(['Y', 'Species', 'Contig', 'Position'], inplace=True)

    # go over all files
    for i in range(num_reps):
        mb_gwas = pd.read_hdf(RAND_MWAS_RESULTS_DIR.joinpath(f'mb_gwas_{i}.h5'))

        mb_gwas.reset_index(inplace=True)
        mb_gwas.Contig = remove_contig_parts(mb_gwas.Contig.values)
        mb_gwas.set_index(['Y', 'Species', 'Contig', 'Position'], inplace=True)
        mb_gwas = mb_gwas.merge(original, how='left', left_index=True, right_index=True, suffixes=('', '_orgnl'))

        summary.loc[i, 'tested'] = len(mb_gwas)
        summary.loc[i, 'min_N'] = mb_gwas['N'].min() #TODO:tmp
        summary.loc[i, 'signif_any'] = (mb_gwas['Pval'] <= 0.05 / NUM_SNPS).sum()
        summary.loc[i, 'signif_right_sign'] = ((mb_gwas['Pval'] <= 0.05 / NUM_SNPS) &
                                              (np.sign(mb_gwas['Coef']) == np.sign(mb_gwas['Coef_orgnl']))).sum()

    summary.to_csv(ANALYSIS_DIR.joinpath(f'significant_{NUM_SNPS}snps_{NUM_REPS}reps.csv'))
    print(summary, summary.sum())
    return


# visualization: hist the number of significant SNPs in each random sample, and compare to the number (17/40) we got originally.
def compare_random_real():
    # load the summary table.
    # Compute: min, max, mean, std
    summary = pd.read_csv(ANALYSIS_DIR.joinpath('significant_40snps_1000reps.csv'), index_col=0)
    print(summary['tested'].describe())
    print(summary['signif_right_sign'].describe())

    # plot histogram compared with empirical
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    fsize = 7
    ax.hist(summary['signif_right_sign'], color=c.main_color_1, label='Random SNP choice', alpha=.8, bins=np.arange(0, 41, 1))
    ax.axvline(17, ls='--', color=c.red, lw=1, label='MWAS replication')

    ax.set_xlabel('Number of significant SNPs', fontsize=fsize)
    ax.set_ylabel('Number of events', fontsize=fsize)
    ax.legend(fontsize=fsize)
    ax.tick_params(axis='both', labelsize=fsize)
    ax.set_xlim(0, 40)
    # ax.set_xlim(-.5, 40.5)
    # ax.set_yscale('log')

    # save
    plt.savefig(FIGS_DIR.joinpath("random_vs_real_int.png"), dpi=300, bbox_inches='tight')
    # plt.savefig(FIGS_DIR.joinpath("random_vs_real_log.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return

######################## Visualize real replication results #######################################################

def original_vs_replication():
    # load the original results, compute minor-major
    original = pd.read_hdf(MWAS_DIR.joinpath('mb_gwas.h5'))
    # original = original.loc[original['Global_Bonferroni'] <= .05]
    original.reset_index(inplace=True)
    original.Contig = remove_contig_parts(original.Contig.values)
    original.set_index(['Y', 'Species', 'Contig', 'Position'], inplace=True)
    # original['diff'] = original['min_mn'] - original['maj_mn']
    

    # load the (real) replication, compute minor-major
    replic = pd.read_hdf(BASE_DIR.joinpath('20221027_110736_mwas_bmi_lifelines_rep_inclusive', 'mb_gwas.h5'))
    replic.reset_index(inplace=True)
    replic.Contig = remove_contig_parts(replic.Contig.values)
    replic.set_index(['Y', 'Species', 'Contig', 'Position'], inplace=True)
    # replic['diff'] = replic['min_mn'] - replic['maj_mn']

    merge_cols = ['Coef', 'Coef_025', 'Coef_975', 'Global_Bonferroni']
    df = original[merge_cols].merge(replic[merge_cols], how='right', left_index=True, right_index=True, suffixes=('_Or', '_Rep'))
    
    # plot the coef-coef plot.
    df['CI_W_Or'] = df['Coef_975_Or'] - df['Coef_025_Or']
    df['CI_W_Rep'] = df['Coef_975_Rep'] - df['Coef_025_Rep']
    
    for s in ['_Or', '_Rep']:
        # tranform units to phenotype units:
        df['Coef' + s] = df['Coef'+s] * MAF_1_VALUE
        df['Coef_975' + s] = df['Coef_975'+s] * MAF_1_VALUE
        df['Coef_025' + s] = df['Coef_025'+s] * MAF_1_VALUE

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fsize = 8
    l_color = c.red
    data_1_c = c.main_color_1
    data_2_c = c.main_color_2

    # reference lines: zeros and '=='
    ax.axhline(0, ls='-', lw=.5, c=l_color, label='', zorder=1)
    ax.axvline(0, ls='-', lw=.5, c=l_color, label='', zorder=1)
    ax.plot(np.linspace(-4.5, 4.5, 20), np.linspace(-4.5, 4.5, 20), '--', ls='-', lw=.5, c=l_color, zorder=1)


    ### plot SNPs significant in both cohorts
    sig_sig_df = df.loc[df['Global_Bonferroni_Rep'] <= .05]
    print(len(sig_sig_df))
    plt.hlines(y=sig_sig_df['Coef_Rep'], xmin=sig_sig_df['Coef_025_Or'], xmax=sig_sig_df['Coef_975_Or'],
               colors=data_1_c, lw=1, alpha=1, label='Significant in DMP', zorder=3)
    plt.vlines(x=sig_sig_df['Coef_Or'], ymin=sig_sig_df['Coef_025_Rep'], ymax=sig_sig_df['Coef_975_Rep'],
               colors=data_1_c, lw=1, alpha=1, zorder=3)

    ### also plot non-significant (in the DMP) SNPs?
    sig_non_df = df.loc[df['Global_Bonferroni_Rep'] > .05]
    print(len(sig_non_df))
    plt.hlines(y=sig_non_df['Coef_Rep'], xmin=sig_non_df['Coef_025_Or'], xmax=sig_non_df['Coef_975_Or'],
               colors=data_2_c, lw=1, alpha=1, label='Non-significant', zorder=2)
    plt.vlines(x=sig_non_df['Coef_Or'], ymin=sig_non_df['Coef_025_Rep'], ymax=sig_non_df['Coef_975_Rep'],
               colors=data_2_c, lw=1, alpha=1, zorder=2)


    plt.legend(fontsize=fsize)
    ax.set_xlabel('Israel, SNP coefficient, BMI points', fontsize=fsize)
    ax.set_ylabel('Netherlands, SNP coefficient, BMI points', fontsize=fsize)
    ax.set_xticks(np.arange(-4,5,1))
    ax.set_yticks(np.arange(-4,5,1))
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.tick_params(axis='both', labelsize=fsize)

    plt.savefig(FIGS_DIR.joinpath("coef_coef_l.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return


