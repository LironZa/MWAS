import logging
import os
from LabData import config_global as config
from LabData.DataAnalyses.MBSNPs.MBSNPAnalyses import MBMWAS, MBSpeciesMWAS, MBSNPCoverage, MBMWASFromMWAS
from LabData.DataAnalyses.MBSNPs.MBSNPAnalyses import MBSNPPairwiseDistances
from LabData.DataMergers.DataMerger import DataMerger
from LabUtils.Utils import date2_dir, write_members, to_list
from LabData.DataLoaders.GutMBLoader import GutMBLoader
from LabData.DataMergers.MultiDataLoader import MultiDataLoader
from LabData.DataLoaders.MBLoader import get_mb_loader_class
from collections import namedtuple
from LabUtils.timeutils import sample_date_19000101

_log = logging.getLogger(__name__)

mwas_run = namedtuple('mwas_run', ['mwas_dir', 'params'])

# SubjectLoader example args
subjects_args_pnp = {'groupby_reg': 'first', 'study_ids': ['PNP1'], 'countries': ['IL']}
subjects_args_age_gender = {'cols': ['age', 'gender'], 'groupby_reg': 'median'}


def y_gen_f_get_species_abundance(subjects_df, species_set):
    return GutMBLoader().get_data('{}_species'.format(config.mb_species_db_basename), col_names_as_ids=True,
                                  subjects_df=subjects_df, take_log=True, col_ids=species_set)

def species_cov_f_get_species_abundance(species, subjects_df, snp_loader):
    return GutMBLoader().get_data('{}_species'.format(config.mb_species_db_basename), col_names_as_ids=True,
                                  subjects_df=subjects_df, take_log=True, col_ids=[species])

def get_mb_samples(body_site, **kwargs):
    return get_mb_loader_class(body_site).\
        get_data(df='{}_species'.format(config.mb_species_db_basename), **kwargs).\
        df_metadata.index.get_level_values('SampleName').to_list()

# default function for testing binary ys
def is_binary_y_valid(y, max_on_most_freq_val_in_col=None, min_on_non_freq_val_for_y=None, y_binary=None):
    if y_binary is not None and not y_binary:
        return True
    if max_on_most_freq_val_in_col is not None or min_on_non_freq_val_for_y is not None:
        most_freq_val_count = y.value_counts().iloc[0]
        counts = y.count()
        if max_on_most_freq_val_in_col is not None and most_freq_val_count > max_on_most_freq_val_in_col * counts:
            return False
        if min_on_non_freq_val_for_y is not None and counts - most_freq_val_count < min_on_non_freq_val_for_y:
            return False
    return True

class CommonParams():
    species_set = ['SGB_8017'] if config.DEBUG else None
    min_positions_per_sample = 20 if config.DEBUG else 20000 # Min number of positions per sample
    min_subjects_per_snp_cached = 500 if config.DEBUG else 500 # For MAFs, min number of subjects per snp for cache file
    min_subjects_per_snp = 2 if config.DEBUG else 400 # Min number of analyzed samples per allele
    min_reads_per_snp = 1 if config.DEBUG else 1 # Min number of sequencing reads per analyzed allele
    send_to_queue = False if config.DEBUG else True
    max_jobs = 300
    is_y_valid_f = staticmethod(is_binary_y_valid) # Function executed just before the regression to checks whether the analyzed y is valid
    max_on_fraq_major_per_snp = 0.99 # Max fraction of major allele frequency in analyzed samples
    max_on_most_freq_val_in_col = 0.95 # Max fraction of the most frequent y in analyzed samples
    min_on_non_freq_val_for_y = None # Min number of subjects that do not have the most frequent y in analyzed samples
    min_on_minor_per_snp = 1 if config.DEBUG else 10 # Min number of analyzed samples with a minor allele
    body_site = 'Gut'
    species_blocks = 10
    jobname = 'mwas'


class MWAS(object):
    def __init__(self, params, work_dir=None, analyses_dir=None, base_mwas_results=None):
        self._params = params

        self._work_dir = os.path.join(params.analyses_dir, date2_dir() + '_' + params.work_dir_suffix)\
            if work_dir is None else work_dir

        params_fname = os.path.join(self._work_dir, 'PARAMS.txt')
        if not os.path.exists(params_fname):
            write_members(params_fname, self._params)

        self._subjects_gen_f = self._build_subjects_gen_f()
        self._covariate_gen_f = self._build_covariate_gen_f()
        self._species_cov_gen_f = None if not hasattr(self._params, 'species_specific_cov_f') else \
            self._params.species_specific_cov_f
        self._contig_cov_gen_f = None if not hasattr(self._params, 'contig_specific_cov_f') else \
            self._params.contig_specific_cov_f
        self._y_gen_f = self._build_y_gen_f()
        self._base_mwas_results = base_mwas_results
        self._mbwas = self._build_mbmwas_instance()

    @staticmethod
    def get_work_dir(mr):
        return os.path.join(mr.params.analyses_dir, mr.mwas_dir)

    @staticmethod
    def _data_gen(loaders, subjects_df=None, join_metadata=False, **kwargs):
        accepts_subjects_df = all([l != 'SubjectLoader' for l in to_list(loaders)])
        return MultiDataLoader(loaders, subjects_df=subjects_df, **kwargs).get_data(join_metadata=join_metadata) if \
            accepts_subjects_df else MultiDataLoader(loaders, **kwargs).get_data(join_metadata=join_metadata)


    # The signature of the y_gen_f function is y_gen_f(subjects_df, species_set) to allow other funcs to get species_set
    def _build_y_gen_f(self):
        return self._params.y_gen_f if hasattr(self._params, 'y_gen_f') else \
            lambda subjects_df, species_set=None: \
                self._data_gen(self._params.y_loaders, subjects_df=subjects_df, **self._params.y_get_data_args)

    # The signature of the subjects_gen_f function is subjects_gen_f()
    def _build_subjects_gen_f(self):
        if hasattr(self._params, 'subjects_gen_f'):
            return self._params.subjects_gen_f
        elif hasattr(self._params, 'subjects_loaders') and self._params.subjects_loaders is not None:
            return lambda: self._data_gen(self._params.subjects_loaders, join_metadata=True,
                                          **self._params.subjects_get_data_args)
        else:
            return None

    def _build_covariate_gen_f(self):
        return self._params.covariate_gen_f if hasattr(self._params, 'covariate_gen_f') else \
            lambda subjects_df: \
                self._data_gen(self._params.covariate_loaders, subjects_df=subjects_df, **self._params.covariate_get_data_args)


    def gen_mwas(self):
        self._mbwas.run(y_gen_f=self._y_gen_f, covariate_gen_f=self._covariate_gen_f,
                        subjects_gen_f=self._subjects_gen_f,
                        species_cov_gen_f=self._species_cov_gen_f, contig_cov_gen_f=self._contig_cov_gen_f,
                        species_blocks=self._params.species_blocks,
                        species_set=self._params.species_set, ignore_species=self._params.ignore_species,
                        max_jobs=self._params.max_jobs, jobname=self._params.jobname,
                        snp_set=self._params.snp_set, samples_set=self._params.samples_set,
                        other_samples_set=self._params.other_samples_set, collect_data=self._params.collect_data)

        if hasattr(self._params, 'test_subjects_get_data_args'):
            for test_name, data_args in self._params.test_subjects_get_data_args.items():
                test_subjects_gen_f = self._build_subjects_gen_f(self._params.subjects_loaders, data_args)
                self._mbwas.run(y_gen_f=self._y_gen_f, covariate_gen_f=self._covariate_gen_f,
                                subjects_gen_f=test_subjects_gen_f,
                                species_cov_gen_f=self._species_cov_gen_f, contig_cov_gen_f=self._contig_cov_gen_f,
                                species_set=self._params.species_set, results_file_suffix=test_name)
        return self._work_dir


    def collect_mwas_data(self, pval_cutoff=0.05, pval_col='Global_Bonferroni', max_rows=100000, snp_set=None,
                          convert_snp_to_coef=False, results_file_suffix=None,
                          compact_collected_data=False, collect_only_snps=False, max_rows_per_species=5000):
        if not self._mbwas.can_collect_mwas_data():
            return
        #TODO: when extracting many SNP's data, increase the (MBSNPAnalyses) _mem_def by a lot
        if snp_set is None:
            df = self._mbwas.load_results()
            if df.empty:
                return
            df = df[df[pval_col] <= pval_cutoff]
            if max_rows_per_species is not None:
                df = df.sort_values(pval_col).groupby('Species').head(max_rows_per_species)
            if max_rows is not None and len(df) > max_rows:
                df = df.sort_values(pval_col).iloc[:max_rows]
            snp_set=df
        collect_only_snps_f = lambda x: None if collect_only_snps else x
        self._mbwas.run(y_gen_f=collect_only_snps_f(self._y_gen_f),
                        covariate_gen_f=collect_only_snps_f(self._covariate_gen_f),
                        subjects_gen_f=self._subjects_gen_f,
                        species_cov_gen_f=collect_only_snps_f(self._species_cov_gen_f),
                        contig_cov_gen_f=collect_only_snps_f(self._contig_cov_gen_f),
                        species_blocks=self._params.species_blocks, snp_set=snp_set, collect_data=True,
                        collect_only_snps=collect_only_snps,
                        convert_snp_to_coef=convert_snp_to_coef,
                        results_file_suffix=results_file_suffix,
                        compact_collected_data=compact_collected_data)

