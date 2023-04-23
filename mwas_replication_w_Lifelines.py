import os
from pathlib import Path
import pandas as pd
import numpy as np
from LabData import config_global as config
from LabData.DataAnalyses.MBSNPs.MWAS import MWAS, CommonParams
from LabData.DataLoaders.Loader import LoaderData
from LabUtils.addloglevels import sethandlers
## take from the original MWAS the loaders for microbiome samples and for the covariates
from UseCases.DataAnalyses.MBSNP_MWAS.MWASPaper.mwas_robust_samples_list_10K import TABLES_DIR, robust_features_loader, \
    robust_samples_loader
from LabData.DataAnalyses.MBSNPs.MWASInterpreter import remove_contig_parts


BASE_DIR = Path(config.analyses_dir)
TABLES_DIR = BASE_DIR.joinpath('MWAS_robust_samples_list')

# path to SNPs list (with matched contigs), data df, RA df
SNPS_DF_P = TABLES_DIR.joinpath('sigSNPs_20223003_postClump_contig_match.csv')
DATA_DF_P = TABLES_DIR.joinpath('lifelines_data_20222010_140759.csv')
MB_COMPOSITION_DF_P = TABLES_DIR.joinpath('lifelines_data_20222010_140759_mb_species_abundance.csv')


## loader for species RA
def lifelines_species_covariate_loader(species_id, subjects_df, snp_loader) -> LoaderData:
    mb_data_p = MB_COMPOSITION_DF_P
    abundance = pd.read_csv(mb_data_p, index_col=0)[species_id]
    log_abundance = np.log10(abundance)
    df = log_abundance.to_frame('log10_RA')
    return LoaderData(df, None)


def robust_features_loader_clipped(features_cols, table_path, subjects_df=None):
    '''
    this will be the loader for the MWAS y/cov loaders. remember: index needs to be mb sample
    #TODO: add filter by subjects df?
    '''
    if type(features_cols) == str:
        features_cols = [features_cols]

    df = pd.read_csv(table_path).set_index('SampleName')
    features_df = df[features_cols]

    from LabData.DataUtils.DataProcessing import NormDistCapping
    features_df = NormDistCapping(sample_size_frac=.98, clip_sigmas=5, remove_sigmas=9).fit_transform(features_df, skip_binary=True)

    return LoaderData(pd.DataFrame(features_df), None)


## load the snps list and set the updated contigs
def get_snps_set(snps_set_p):
    snps_set = pd.read_csv(snps_set_p).rename(columns={'Contig_NEW': 'Contig'}).set_index(['Y', 'Species', 'Contig', 'Position'])
    return snps_set

## MWAS params
class Replication_bmi_Params(CommonParams):

    # robust_samples_data_p = DATA_DF_P
    samples_set = robust_samples_loader(DATA_DF_P)
    snp_set = get_snps_set(SNPS_DF_P) ## the significant SNPs in the discovery MWAS, with adjusted contig parts
    jobname = 'mwsBMI'

    min_reads_per_snp = 1
    work_dir_suffix = 'mwas_bmi_lifelines_rep_inclusive'
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
    species_blocks = 1

    # send_to_queue = False
    send_to_queue = True

    subjects_loaders = ['SubjectLoader']
    subjects_get_data_args = {'groupby_reg': 'first', 'study_ids': [32]}

    y_gen_f = lambda subjects_df, species_set: robust_features_loader(['bmi'], DATA_DF_P, subjects_df)
    covariate_gen_f = lambda subjects_df: robust_features_loader(['age', 'gender'], DATA_DF_P, subjects_df)
    species_specific_cov_f = lifelines_species_covariate_loader

    ret_cov_fields = True
    output_cols = ['N', 'Coef', 'Pval', 'Coef_025', 'Coef_975', 'Coef_SE', 'MAF_SE',
                   'maj_n', 'maj_mn', 'maj_sd', 'min_n', 'min_mn', 'min_sd', 'log10_RA_Pval', 'log10_RA_Coef']


## run MWAS with the (contig-adjusted) SNPs list, lifelines samples list
def run_mwas(params):
    m = MWAS(params)
    work_dir = m.gen_mwas()
    return work_dir


def merge_discovery_replication(lifelines_mwas_name='20221027_110736_mwas_bmi_lifelines_rep_inclusive'):
    # annotated mwas path
    discovery_p = BASE_DIR.joinpath('20220402_234043_mwas_bmi_10K/snp_annotations/annotated_clumped_0.3.csv')
    dis = pd.read_csv(discovery_p, index_col=[0, 1, 2, 3])

    # validation path
    replication_p = BASE_DIR.joinpath(lifelines_mwas_name, 'mb_gwas.h5')
    rep = pd.read_hdf(replication_p)

    # remove contig parts
    rep.reset_index(inplace=True)
    rep['ContigWithParts'] = rep['Contig']
    rep.Contig = remove_contig_parts(rep.Contig.values)
    rep.set_index(['Y', 'Species', 'Contig', 'Position'], inplace=True)

    # merge tables
    merged = dis.merge(rep, how='left', left_index=True, right_index=True, suffixes=('', '_lifelines'))
    merged['replicated'] = (merged['Global_Bonferroni_lifelines'] <= .05)
    print(merged)

    # save in the mwas dir
    OUTPT_DIR = BASE_DIR.joinpath('20220402_234043_mwas_bmi_10K', lifelines_mwas_name)
    OUTPT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPT_DIR.joinpath('merged_results.csv'))
    merged.to_excel(OUTPT_DIR.joinpath('merged_results.xlsx'))
    merged.reset_index().to_excel(OUTPT_DIR.joinpath('merged_results_unindexed.xlsx'))
    return



if __name__ == '__main__':
    sethandlers(file_dir=config.log_dir)
    os.chdir('/net/mraid08/export/genie/LabData/Analyses/lironza/jobs')
    run_mwas(Replication_bmi_Params)

    # merge_discovery_replication()