import os
from pathlib import Path
import pandas as pd
import numpy as np
from LabData import config_global as config
from LabData.DataLoaders.Loader import LoaderData
from LabData.DataAnalyses.MBSNPs.MWAS import MWAS, CommonParams
from LabData.DataAnalyses.MBSNPs.MWASInterpreter import MWASInterpreter
from LabData.DataAnalyses.MBSNPs.Plots.MWASPaper.mwas_paper_fig_utils import choose_top_snps
from LabQueue.qp import fakeqp
from LabUtils.addloglevels import sethandlers

BASE_DIR = Path(config.analyses_dir)
TABLES_DIR = BASE_DIR.joinpath('MWAS_robust_samples_list')

ROBUST_DATA_TABLE_NAME = 'BasicMWAS_10K_singleSamplePerReg_20223003_234555.csv'
ROBUST_DATA_TABLE_P = TABLES_DIR.joinpath(ROBUST_DATA_TABLE_NAME)

phen_label_to_name = {'age': 'age', 'sex': 'gender', 'bmi': 'bmi'}
PHENS_L = ['bmi']


##################### Loaders for the MWAS Run ############################

### To run MWAS with the pre-made samples list + features table, use these kinds of loaders:
# y_gen_f = lambda subjects_df: robust_features_loader(features_cols, table_path, subjects_df)
# covariate_gen_f = lambda subjects_df: robust_features_loader(features_cols, table_path, subjects_df)

def robust_features_loader(features_cols, table_path, subjects_df=None):
    '''
    this will be the loader for the MWAS y/cov loaders. remember: index needs to be mb sample
    #TODO: add filter by subjects df?
    '''
    if type(features_cols) == str:
        features_cols = [features_cols]

    df = pd.read_csv(table_path).set_index('SampleName')
    features_df = df[features_cols]
    return LoaderData(pd.DataFrame(features_df), None)


def robust_samples_loader(table_path):
    '''
    This will be the loader for mb samples list, which will be the basic input to run the MWAS
    '''
    df = pd.read_csv(table_path)
    samples_list = list(df['SampleName'].values)
    assert len(samples_list) == len(set(samples_list))
    return samples_list


def species_covariate_loader(species_id, subjects_df, snp_loader) -> LoaderData:
    mb_data_p = TABLES_DIR.joinpath(f'BasicMWAS_10K_singleSamplePerReg_20223003_234555_mb_species_abundance.csv')
    abundance = pd.read_csv(mb_data_p, index_col=0)[species_id]
    log_abundance = np.log10(abundance)
    df = log_abundance.to_frame('log10_RA')
    return LoaderData(df, None)


################################################################################################

class RobustSamplesParams(CommonParams):
    robust_samples_data_p = ROBUST_DATA_TABLE_P
    samples_set = robust_samples_loader(robust_samples_data_p)  # Note. #TODO: make sure this runs. commented out to make other runs quicker

    min_reads_per_snp = 1
    max_on_fraq_major_per_snp = .99
    max_on_most_freq_val_in_col = .95
    min_subjects_per_snp = 1000
    min_on_minor_per_snp = 50
    min_positions_per_sample = 0
    filter_by_species_existence = False  # Note. What does it actually do, now that we only map to present?
    max_pval_to_detailed = 1e-10
    largest_sample_per_user = False  # Note

    max_jobs = 400
    species_blocks = 2

    # species_set = ['Rep_449']  #  ['Rep_1011']
    # send_to_queue = False
    send_to_queue = True

    subjects_loaders = ['SubjectLoader']
    subjects_get_data_args = {'groupby_reg': 'first', 'study_ids': ['10K'], 'countries': ['IL']}

    species_specific_cov_f = species_covariate_loader
    ret_cov_fields = True

    output_cols = ['N', 'Coef', 'Pval', 'Coef_025', 'Coef_975', 'Coef_SE', 'MAF_SE',
                   'maj_n', 'maj_mn', 'maj_sd', 'min_n', 'min_mn', 'min_sd', 'log10_RA_Pval', 'log10_RA_Coef']


class Params_bmi_robust(RobustSamplesParams):
    work_dir_suffix = 'mwas_bmi_10K'
    jobname = 'mwsBMI'
    y_gen_f = lambda subjects_df, species_set: robust_features_loader(['bmi'], ROBUST_DATA_TABLE_P, subjects_df)
    covariate_gen_f = lambda subjects_df: robust_features_loader(['age', 'gender'], ROBUST_DATA_TABLE_P, subjects_df)


def run_mwas_new(params, work_dir=None, interpret_mwas=False, phen=None):
    if work_dir is None:
        m = MWAS(params)
        work_dir = m.gen_mwas()
    else:
        work_dir = Path(config.analyses_dir).joinpath(work_dir)
        m = MWAS(params, work_dir=work_dir)

    if interpret_mwas:
        params.y_to_show = phen_label_to_name[phen]
        m.interpret_mwas([(work_dir, params)])
    return work_dir


def collect_data_simple(work_dir_name, params, n, snp_set=None, collect_only_snps=False, results_file_suffix=None):
    work_dir = Path(config.analyses_dir).joinpath(work_dir_name)
    m = MWAS(params, work_dir=work_dir)
    m.collect_mwas_data(pval_cutoff=0.05, pval_col='Global_Bonferroni',
                        max_rows=n, max_rows_per_species=None,
                        snp_set=snp_set, results_file_suffix=results_file_suffix,
                        compact_collected_data=True, collect_only_snps=collect_only_snps)


def collect_top_snps(phen, n, opt_by):  # opt_by: 'diff', 'Coef', or 'Pval'
    from UseCases.DataAnalyses.MBSNP_MWAS.mwas_liron import PARAMS_DICT
    work_dir, params = PARAMS_DICT['SNPs'][phen]
    work_dir = Path(config.analyses_dir).joinpath(work_dir)

    annots_df = pd.read_csv(work_dir.joinpath('snp_annotations', 'snp_annotations.csv'), index_col=[0, 1, 2, 3]).reset_index()
    annots_df = annots_df.drop_duplicates(subset=['Y', 'Species', 'Contig', 'Position'])
    annots_df.Contig = annots_df['ContigWithParts']
    top_snps = choose_top_snps(annots_df, opt_by=opt_by, n=n, unique_by=['Species']).set_index(['Y', 'Species', 'Contig', 'Position'])

    m = MWAS(params, work_dir=work_dir)
    m.collect_mwas_data(pval_cutoff=0.05, pval_col='Global_Bonferroni',
                        max_rows=n, snp_set=top_snps, results_file_suffix=f'_{opt_by}',
                        compact_collected_data=False, collect_only_snps=False)
    return


def collect_snps_vecs(params, work_dir_name, n=None, snp_set=None, results_file_suffix='', compact=False):
    # with the defult params, MWAS will take the top n p-value SNPs and extract their data
    if (snp_set is not None and n is not None) and len(snp_set) > n:
        snp_set = snp_set.sort_values('Pval').iloc[:n]
    print(f"extracting data for {len(snp_set) if (snp_set is not None) else n} SNPs")
    work_dir = os.path.join(config.analyses_dir, work_dir_name)
    m = MWAS(params, work_dir)
    m.collect_mwas_data(pval_cutoff=0.05, pval_col='Global_Bonferroni',
                        max_rows=n, snp_set=snp_set, results_file_suffix=results_file_suffix,
                        compact_collected_data=compact, collect_only_snps=False
                        )
    return


def run_interpreter(params=None, work_dir_name=None):
    """ only for regular (SNPs) MWAS """
    work_dir = os.path.join(config.analyses_dir, work_dir_name)
    print(work_dir)
    mi = MWASInterpreter(params,  work_dir,
                         do_snp_annotations=True, annotate_all_snps=True,
                         do_manhattan_plot=False, do_qq_plot=True,  do_volcano_plot=False,
                         pval_col='Global_Bonferroni', pval_cutoff=.05,
                         do_annotated_manhattan=True, do_test_nonsynonymous_enrichment=True,
                         do_function_counts=True)
    mi.run()
    return


if __name__ == '__main__':
    run_mwas_new(Params_bmi_robust)


