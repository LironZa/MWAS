import logging
import os
from abc import ABCMeta, abstractmethod
import random
import glob2
import numpy as np
import re
import pandas as pd
from LabData import config_global as config
from LabData.DataLoaders.Loader import Loader
from LabData.DataLoaders.MBSNPLoader import MAF_1_VALUE, MAF_MISSING_VALUE, get_mbsnp_loader_class
from LabData.DataLoaders.MBSNPLoader import MBSNPLoader
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from LabData.DataMergers.DataMerger import DataMerger
from LabQueue.qp import fakeqp
from LabUtils.Utils import date2_dir, mkdirifnotexists, grouper, to_list, isbinary, try_rm_by_pattern, \
    load_h5_files, intersection
from statutils import ols
from LabUtils.timeutils import sample_date_19000101, current_datetime_as_str, is_series_or_index_dt64_utc
from pandas import concat, DataFrame, Series, read_hdf, to_numeric
import itertools
from collections import defaultdict

_log = logging.getLogger(__name__)

class RunResults(object):
    def __init__(self, df=None, df_counts=None, df_detailed=None, df_snp_clusters=None, df_snp_data_fname=None):
        self.df = df
        self.df_counts = df_counts
        self.df_detailed = df_detailed
        self.df_snp_clusters = df_snp_clusters
        self.df_snp_data_fname = df_snp_data_fname

class MBSNPAnalyses(object):
    __metaclass__ = ABCMeta

    def __init__(self, params, work_dir=None):
        self._cache_dir = os.path.join(config.mb_pipeline_dir, 'Analyses', 'MBSNP')

        self._work_dir = os.path.join(params.analyses_dir, date2_dir()) if work_dir is None else work_dir
        mkdirifnotexists(self._work_dir)

        self._jobs_dir = os.path.join(self._work_dir, 'jobs')
        mkdirifnotexists(self._jobs_dir)
        os.chdir(self._jobs_dir)

        self._min_positions_per_sample = params.min_positions_per_sample
        self._min_reads_per_snp = params.min_reads_per_snp
        self._min_subjects_per_snp = params.min_subjects_per_snp
        self._min_subjects_per_snp_cached = params.min_subjects_per_snp_cached
        self._max_on_fraq_major_per_snp = params.max_on_fraq_major_per_snp
        self._max_on_most_freq_val_in_col = params.max_on_most_freq_val_in_col
        self._min_on_non_freq_val_for_y = params.min_on_non_freq_val_for_y
        self._min_on_minor_per_snp = params.min_on_minor_per_snp
        self._largest_sample_per_user = params.largest_sample_per_user
        self._groupby_reg = params.groupby_reg
        self._qp = config.qp if params.send_to_queue else fakeqp
        self._is_y_valid_f = params.is_y_valid_f
        self._indices = None
        self._body_site = params.body_site
        self._filter_by_species_existence = params.filter_by_species_existence
        self._subsample_dir = params.subsample_dir
        self._select_n_rand_samples = params.select_n_rand_samples
        self._mwas_data_clusterer = params.mwas_data_clusterer
        self._verbose = params.verbose
        self._results_file_suffix = None

    def _get_species_list(self):
        samples_per_species = self._get_mbsnp_loader().\
            get_num_samples_per_species(self._min_positions_per_sample, self._min_subjects_per_snp)
        return list(samples_per_species.index.get_level_values(0))

    def _get_subjects_df(self, subjects_gen_f):
        subjects_dl = SubjectLoader().get_data(groupby_reg='first') if subjects_gen_f is None else subjects_gen_f()
        subjects_df = subjects_dl.df.reset_index('Date', drop=True)
        assert subjects_df.index.duplicated().sum() == 0
        return subjects_df

    def run(self, y_gen_f=None, covariate_gen_f=None, species_blocks=10, species_set=None, ignore_species=None,
            run_per_contig=True, subjects_gen_f=None, results_file_suffix='', snp_set=None, collect_data=False,
            species_cov_gen_f=None, contig_cov_gen_f=None, samples_set=None, other_samples_set=None,
            collect_only_snps=False, compact_collected_data=False, convert_snp_to_coef=False, compute_mrs=False,
            max_jobs=100, jobname='mwas'):
        """
        Runs a SNP analysis by iterating over contigs
        :param y_gen_f: Function to generate y, ensuring that the full y is not passed across the queue
        :param covariate_gen_f: Optional. Function to generate covariates, ensuring that covariates are not sent to qp
        :param species_blocks: Number of species to run as a block within a single queue job
        :param species_set: Full set of species to run on. If None, then run on all species available
        :param ignore_species: Set of species to ignore from. If None, will run all species in species_set
        :param run_per_contig: bool, whether to combine contigs or run the analysis separately on contigs
        :param subjects_gen_f: Optional. Function to generate subjects DataFrame
        :param results_file_suffix: str, optional. Str to use as suffix to the results file
        :param snp_set: DataFrame, optional. Only work on this set of SNPs
        :param collect_data: bool. Only collect the snp data + covariates + ys to analyze
        :param collect_only_snps: bool. Only collect snp data without ys
        :param compact_collected_data: bool. Condense the resulting collected data into a DataFrame
                with samples on rows and one column per SNP position, include covariates and various ys.
                Only works when collect_data is True
        :param convert_snp_to_coef: bool. Return the coef of the snp from the mwas (requires mwas as snp_set)
        :param max_jobs: int, optional. Max number of parallel jobs to send to the queue (default: 100)
        :return: No return value. Results are saved in files
        """
        self._results_file_suffix = results_file_suffix
        self._collect_data = collect_data
        self._collect_only_snps = collect_only_snps
        assert collect_data or not compact_collected_data
        self._compact_collected_data = compact_collected_data
        self._convert_snp_to_coef = convert_snp_to_coef
        self._compute_mrs = compute_mrs
        assert not convert_snp_to_coef or snp_set is not None
        out_fname = os.path.join(self._work_dir, '{}.h5'.format(self._result_files_prefix()))
        if os.path.exists(out_fname):
            _log.info(f'Skipping run: Primary output file {out_fname}')
            return

        # snp_set = self._debug_snp_set()
        if snp_set is not None:
            species_set = snp_set.index.unique('Species').tolist()
        if species_set is None:
            species_set = self._get_species_list()
        if ignore_species is not None:
            species_set = [spc for spc in species_set if spc not in ignore_species]
        with self._qp(jobname=jobname, _delete_csh_withnoerr=True, q=['himem7.q'], _trds_def=2, _max_secs=-1,
                      max_r=max_jobs, _mem_def='10G', delay_batch=2, _tryrerun=True) as q:
            q.startpermanentrun()

            tkttores = {}
            for i, species_block in enumerate(grouper(random.sample(species_set, len(species_set)), species_blocks)):
                species_block = [s for s in species_block if s]  # remove empty ones from last block
                _log.info('Sending analysis for {} species ({} to {})'.
                          format(len(species_block), species_block[0], species_block[-1]))
                tkttores[i] = q.method(self._run_species_set, (species_block, run_per_contig, y_gen_f,
                                                               covariate_gen_f, subjects_gen_f,
                                                               snp_set, species_cov_gen_f, contig_cov_gen_f,
                                                               samples_set, other_samples_set))

            rets = {k: q.waitforresult(v) for k, v in tkttores.items()}

        self._post_full_run(rets)

    def _filter_by_positions_in_snp_set(self, df, snp_set, species, contig):
        if snp_set is None or 'Position' not in snp_set.index.names:
            return df, None
        snp_set = snp_set[np.logical_and(
            snp_set.index.get_level_values('Species') == species,
            snp_set.index.get_level_values('Contig') == contig)]
        positions = snp_set.index.unique('Position')
        try:
            ys = snp_set.index.unique('Y')
        except KeyError:
            ys = None
        df_poitions = df.columns.get_level_values('Position') if 'Position' in df.columns.names else df.columns
        return df.loc[:, df_poitions.isin(positions)], ys

    def _run_species_contig(self, species, contig_dl, contig, y_dl=None, co_dl=None, num_covariates=0,
                            run_per_contig=True, snp_set=None, rets_counts=None, rets_detailed=None):
        def foreach_y(y_col):
            return self._run(species, contig, x, yy[y_col], y_col, num_covariates,
                             rets_counts=rets_counts, rets_detailed=rets_detailed)

        num_tests = pd.Series(dtype=int)

        if run_per_contig:
            _log.info('Running {}:{} Samples={} Positions={}'.
                      format(species, contig, contig_dl.df.shape[0], contig_dl.df.shape[1]))
        else:
            _log.info('Running {}: {} contigs together'.format(species, len(contig_dl)))

        contig_dl.df, snp_set_ys = self._filter_by_positions_in_snp_set(contig_dl.df, snp_set, species, contig)

        if run_per_contig:
            x_dl = [contig_dl, co_dl] if co_dl is not None else [contig_dl]
        else:
            x_dl = contig_dl.append(co_dl) if co_dl is not None else [contig_dl]

        # y may be null when we are just collecting snp data
        if self._collect_only_snps:
            x = DataMerger(x_dl).get_x(inexact_index='Date', res_index_names=['RegistrationCode', 'Date'],
                       fillna_values={'RegistrationCode': 'EmptyRegistrationCode', 'Date': sample_date_19000101()}).df
            x = self._filter_snps_by_stats_of_major(x, num_covariates)
            x = x.replace(MAF_MISSING_VALUE, np.NaN).dropna(how='all')
            if self._compute_mrs:
                return self._compute_mrs_species_contig(species, contig, snp_set, x)
            if self._convert_snp_to_coef:
                # Instead of SNPs, return the mwas coef (average across Ys if several exist)
                assert len(snp_set.index.unique('Y')) == 1
                x = x.mul(snp_set.loc[(slice(None), species, contig), :]['Coef'].droplevel(['Y', 'Species', 'Contig']))
            x.columns = pd.MultiIndex.from_arrays(list(itertools.chain(
                [['snps'] * len(x.columns)], [[species] * len(x.columns)], [[contig] * len(x.columns)],
                [list(x.columns.get_level_values('Position'))])), names=['Type', 'Species', 'Contig', 'Position'])
            return [x]

        if {dl.df.index.name for dl in x_dl + [y_dl]} == {'SampleName'}:
            # this speeds up the index merging in case the indices are SampleName instead of RegistrationCode and Date
            x = x_dl[0].df
            for i in np.arange(1, len(x_dl)):
                x = x.join(x_dl[i].df, how='inner')
            combined_indices = set(x.index.values) & set(y_dl.df.index.values)
            x = x.loc[combined_indices]
            yy = y_dl.df.loc[combined_indices]
            index_name = 'SampleName'

        else:
            x, yy = DataMerger(x_dl, y_dl). \
                get_xy(inexact_index='Date', res_index_names=['RegistrationCode', 'Date'],
                       fillna_values={'RegistrationCode': 'EmptyRegistrationCode', 'Date': sample_date_19000101()})
            index_name = 'RegistrationCode'
        yy = yy.dropna(how='all', axis=1)

        if x.empty:
            return []

        assert x.index.get_level_values(index_name).duplicated().sum() == 0

        x = self._filter_snps_by_stats_of_major(x, num_covariates)

        if x.shape[1] - num_covariates <= 0:
            return []

        yy = self._filter_ys(yy)
        if yy.empty:
            return []

        ys = yy.columns if snp_set_ys is None else intersection(yy.columns, snp_set_ys)

        return [val for sublist in map(foreach_y, ys) if sublist for val in sublist]

    def _filter_snps_by_stats_of_major(self, x, num_covariates):
        if self._max_on_fraq_major_per_snp is not None or \
                self._min_on_minor_per_snp is not None or \
                self._min_subjects_per_snp is not None:
            major_count = MBSNPLoader.compute_maf_major_counts(x).values
            maf_counts = MBSNPLoader.compute_maf_counts(x).values
            res = Series(True, index=x.columns)
            if self._max_on_fraq_major_per_snp is not None:
                maf_freq = major_count / np.clip(maf_counts, 1, None)
                res.iloc[:] = np.logical_and(maf_freq <= self._max_on_fraq_major_per_snp,
                                             maf_freq >= 1 - self._max_on_fraq_major_per_snp)
            if self._min_on_minor_per_snp is not None:
                res.iloc[:] = np.logical_and(res.iloc[:],
                                             np.logical_and(maf_counts - major_count >= self._min_on_minor_per_snp,
                                                            major_count >= self._min_on_minor_per_snp))
            if self._min_subjects_per_snp is not None:
                res.iloc[:] = np.logical_and(res.iloc[:], maf_counts >= self._min_subjects_per_snp)
            if num_covariates > 0:
                res.loc[x.columns[-num_covariates:]] = True
            x = x[res[res].index]
        return x

    def _filter_ys(self, yy):
        if self._min_subjects_per_snp is not None:
            org_cols = yy.shape[1]
            yy = yy.loc[:, yy.count() >= self._min_subjects_per_snp]
            _log.info('Filtered ys from {} to {} cols'.format(org_cols, yy.shape[1]))
        yy = Loader._filter_cols_by_max_on_most_freq_val_in_col(yy, self._max_on_most_freq_val_in_col)
        yy = Loader._filter_cols_by_min_on_non_freq_val_in_col(yy, self._min_on_non_freq_val_for_y)
        return yy

    def _get_mbsnp_loader(self):
        return get_mbsnp_loader_class(self._body_site)

    def _run_species_set(self, species_set, run_per_contig, y_gen_f, covariate_gen_f=None,
                         subjects_gen_f=None, snp_set=None, species_cov_gen_f=None, contig_cov_gen_f=None,
                         samples_set=None, other_samples_set=None):
        snp_loader = self._get_mbsnp_loader()
        subjects_df = self._get_subjects_df(subjects_gen_f)  # this way loading subjects is done once
        y_dl = None if y_gen_f is None else y_gen_f(subjects_df=subjects_df, species_set=species_set)
        g_co_dl = None if covariate_gen_f is None else covariate_gen_f(subjects_df=subjects_df)
        rets = []
        rets_counts = []
        rets_detailed = []
        rets_snp_clusters = []
        all_contigs = []
        for species in to_list(species_set):
            if species_cov_gen_f is not None:  # treat species specific covariates
                # load it, append to co_dl
                try:
                    species_co_dl = species_cov_gen_f(species, subjects_df, snp_loader)
                except FileNotFoundError:
                    continue
                s_co_dl = DataMerger([g_co_dl, species_co_dl]). \
                    get_x(how='left', inexact_index='Date', res_index_names=['RegistrationCode', 'Date'],
                          fillna_values={'RegistrationCode': 'EmptyRegistrationCode', 'Date': sample_date_19000101()})
            else:
                s_co_dl = g_co_dl
            for contig_dl in \
                    snp_loader.get_data(df=species, min_reads_per_snp=self._min_reads_per_snp,
                                        min_positions_per_sample=self._min_positions_per_sample,
                                        min_samples_per_snp=self._min_subjects_per_snp,
                                        min_samples_per_snp_cached=self._min_subjects_per_snp_cached,
                                        data_gen_f='_species_maf_contig_iter',
                                        max_on_fraq_major_per_snp=self._max_on_fraq_major_per_snp,
                                        largest_sample_per_user=self._largest_sample_per_user, subjects_df=subjects_df,
                                        groupby_reg=self._groupby_reg, snp_set=snp_set, samples_set=samples_set,
                                        filter_by_species_existence=self._filter_by_species_existence,
                                        subsample_dir=self._subsample_dir,
                                        column_clusterer=self._mwas_data_clusterer):
                if contig_dl.df.empty:
                    continue
                contig = contig_dl.added_data

                if contig_dl.df_column_clusters is not None:
                    rets_snp_clusters.append(contig_dl.df_column_clusters)

                if contig_cov_gen_f is not None:  # treat species specific covariates
                    contig_cov_dl = contig_cov_gen_f(species, contig.split('_')[1], subjects_df, snp_loader)
                    co_dl = DataMerger([s_co_dl, contig_cov_dl]). \
                        get_x(how='left', inexact_index='Date', res_index_names=['RegistrationCode', 'Date'],
                              fillna_values={'RegistrationCode': 'EmptyRegistrationCode',
                                             'Date': sample_date_19000101()})
                else:
                    co_dl = s_co_dl

                num_covariates = 0 if covariate_gen_f is None else len(co_dl.df.columns)
                if run_per_contig:
                    rets = rets + self._run_species_contig(species, contig_dl, contig, y_dl, co_dl, num_covariates,
                                                           run_per_contig, snp_set, rets_counts, rets_detailed)
                else:
                    all_contigs.append(contig_dl)
                # if len(rets) > 30000:
                #     break

            if not run_per_contig:
                if len(all_contigs) == 0:
                    return DataFrame()
                rets = rets + self._run_species_contig(species, all_contigs, 'all_contigs', y_dl, co_dl, num_covariates,
                                                       run_per_contig, snp_set, rets_counts, rets_detailed)

        return self._post_run(species_set, rets, rets_counts, rets_detailed, rets_snp_clusters)

    @abstractmethod
    def _run(self, species, contig, x, y, y_col, num_covariates, run_per_contig=True, rets_counts=None, rets_detalied=None):
        pass

    def _concat_collected_data(self, rets):
        assert False

    def _post_run(self, species_set, rets, rets_counts=None, rets_detailed=None, rets_snp_clusters=None):
        # When collecting data and compacting it call a special condense function
        if self._compact_collected_data:
            return RunResults(df_snp_data_fname=self._concat_collected_data(rets))

        if len(rets) == 0:
            df = DataFrame()
        elif isinstance(rets, DataFrame) or isinstance(rets, Series):
            df = rets
        elif isinstance(rets[0], DataFrame):
            df = concat(rets).sort_index()
        else:
            df = DataFrame(rets, columns=self._run_result_df_columns()). \
                set_index(self._run_result_df_indices()).sort_index()

        if not df.empty:
            if 'Date' in df.index.names and is_series_or_index_dt64_utc(df.index.get_level_values('Date')):
                df['Date'] = df.index.get_level_values('Date').tz_convert(None)
                df = df.reset_index('Date', drop=True).set_index('Date', append=True)
            df = df.apply(to_numeric, errors='ignore')
            df.to_hdf(os.path.join(self._work_dir, '{}_tmp{}_N{}.h5'.
                                   format(self._result_files_prefix(), current_datetime_as_str(),
                                          len(species_set))), species_set[0])

        df_counts = None
        if rets_counts is not None and not self._collect_data:
            df_counts = pd.DataFrame(rets_counts, columns=['Y', 'Species', 'Contig', 'N']).\
                set_index(['Species', 'Contig', 'Y'])
            if not df_counts.empty:
                df_counts.to_hdf(os.path.join(self._work_dir, '{}_tmp{}_N{}_counts.h5'.
                                              format(self._result_files_prefix(), current_datetime_as_str(),
                                                     len(species_set))), species_set[0])

        df_detailed = None
        if rets_detailed is not None and not self._collect_data:
            boxplot_cols = ['N', 'Mean', 'Median', 'Std', 'Q1', 'Q3', 'Min', 'Max']
            boxplot_cols_minor = ['Minor{}'.format(c) for c in boxplot_cols]
            boxplot_cols_major = ['Major{}'.format(c) for c in boxplot_cols]
            df_detailed = pd.DataFrame(rets_detailed, columns=['Y', 'Species', 'Contig', 'Position', 'R2', 'Pval'] +
                                                              boxplot_cols_minor + boxplot_cols_major).\
                set_index(['Y', 'Species', 'Contig', 'Position'])
            if not df_detailed.empty:
                df_detailed.to_hdf(os.path.join(self._work_dir, '{}_tmp{}_N{}_detailed.h5'.
                                                format(self._result_files_prefix(), current_datetime_as_str(),
                                                       len(species_set))), species_set[0])

        df_snp_clusters = None
        if rets_snp_clusters is not None and len(rets_snp_clusters) > 0:
            df_snp_clusters = pd.concat(rets_snp_clusters)
            df_snp_clusters.to_hdf(
                os.path.join(self._work_dir, '{}_tmp{}_N{}_snp_clusters.h5'.
                             format(self._result_files_prefix(), current_datetime_as_str(),
                                    len(species_set))), species_set[0])

        return RunResults(df=df, df_counts=df_counts, df_detailed=df_detailed, df_snp_clusters=df_snp_clusters)

    def _add_multiple_hypotheses_corrections(self, df, df_counts):
        if not df.empty and 'Pval' in df.columns:
            df['Global_Bonferroni'] = np.clip(df.Pval * df_counts.sum().sum(), 0, 1)
            y_bonferroni = df[['Pval']].join(df_counts.groupby('Y').sum())
            df = df.join(np.clip(y_bonferroni['Pval'] * y_bonferroni['N'], 0, 1).rename('Y_Bonferroni'))
        return df

    def _post_full_run_compact_data(self, rets):
        out_fname = os.path.join(self._work_dir, '{}.h5'.format(self._result_files_prefix()))
        with pd.HDFStore(out_fname, 'w') as hdf_out:
            for run_result in rets.values():
                with pd.HDFStore(run_result.df_snp_data_fname, 'r') as hdf_in:
                    for key in hdf_in.keys():
                        hdf_out[key] = hdf_in[key]

    def _write_out_files(self, df, df_counts, df_detailed, df_snp_clusters):
        _log.info('post_full_run ended with {} entries'.format(len(df)))
        df.to_hdf(os.path.join(self._work_dir, '{}.h5'.format(self._result_files_prefix())),
                  self._result_files_prefix(), mode='w')
        df_counts.to_hdf(os.path.join(self._work_dir, '{}_counts.h5'.format(self._result_files_prefix())),
                         self._result_files_prefix(), mode='w')
        if not df_detailed.empty:
            df_detailed.to_hdf(
                os.path.join(self._work_dir, '{}_detailed.h5'.format(self._result_files_prefix())),
                self._result_files_prefix(), mode='w')
        if not df_snp_clusters.empty:
            df_snp_clusters.to_hdf(
                os.path.join(self._work_dir, '{}_snp_clusters.h5'.format(self._result_files_prefix())),
                self._result_files_prefix(), mode='w')

    def _post_full_run(self, rets):
        if self._collect_data and self._compact_collected_data:
            self._post_full_run_compact_data(rets)
        else:
            df = [r.df for r in rets.values() if r.df is not None]
            df_counts = [r.df_counts for r in rets.values() if r.df_counts is not None]
            df_detailed = [r.df_detailed for r in rets.values() if r.df_detailed is not None]
            df_snp_clusters = [r.df_snp_clusters for r in rets.values() if r.df_snp_clusters is not None]
            if len(df_counts) > 0:
                df_counts = concat(df_counts, sort=False)
            else:
                df_counts = DataFrame()
            if len(df) > 0:
                df = concat(df, sort=False)
                df = self._add_multiple_hypotheses_corrections(df, df_counts)
            else:
                df = DataFrame()
            if len(df_detailed) > 0:
                df_detailed = concat(df_detailed, sort=False)
                df_detailed = self._add_multiple_hypotheses_corrections(df_detailed, df_counts)
            else:
                df_detailed = DataFrame()
            if len(df_snp_clusters) > 0:
                df_snp_clusters = concat(df_snp_clusters, sort=False)
            else:
                df_snp_clusters = DataFrame()

            self._write_out_files(df, df_counts, df_detailed, df_snp_clusters)

        try_rm_by_pattern(os.path.join(self._work_dir, '{}_tmp*_N*.h5').format(self._result_files_prefix()))

    def load_results(self, results_file_suffix=''):
        self._results_file_suffix = results_file_suffix
        return read_hdf(os.path.join(self._work_dir, '{}.h5'.format(self._result_files_prefix())),
                        self._result_files_prefix())

    def post_full_run_recovery_from_files(self):
        def f_pattern(work_dir, results_files_prefix, suffix):
            return os.path.join(work_dir, '{}_*_N[0-9]*{}.h5').format(result_files_prefix, suffix)

        result_files_prefix = self._result_files_prefix()
        df_files = [f for f in glob2.glob(f_pattern(self._work_dir, result_files_prefix, '')) if
                    re.search('N[0-9]+.h5$', f) is not None]
        df = load_h5_files(df_files)
        df_counts = load_h5_files(glob2.glob(f_pattern(self._work_dir, result_files_prefix, '_counts')))
        df_detailed = load_h5_files(glob2.glob(f_pattern(self._work_dir, result_files_prefix, '_detailed')))
        df_snp_clusters = load_h5_files(glob2.glob(f_pattern(self._work_dir, result_files_prefix, '_snp_clusters')))

        if len(df) > 0:
            df = self._add_multiple_hypotheses_corrections(df, df_counts)
        if len(df_detailed) > 0:
            df_detailed = self._add_multiple_hypotheses_corrections(df_detailed, df_counts)

        self._write_out_files(df, df_counts, df_detailed, df_snp_clusters)

    def _run_result_df_indices(self):
        return self._indices

    @abstractmethod
    def _run_result_df_columns(self):
        pass

    @abstractmethod
    def _result_files_prefix(self):
        pass


class MBMWAS(MBSNPAnalyses):
    def __init__(self, params, *args, **kwargs):
        """
        Constructor for MB MWAS analysis
        :param constant_covariate: bool, default True. Whether to add a constant covariate to each analysis
        """
        super(MBMWAS, self).__init__(params, *args, **kwargs)
        self._ols_fields = ['N', 'R2', 'Coef', 'Pval', 'Coef_025', 'Coef_975']\
            if params.output_cols is None else params.output_cols
        self._indices = ['Y', 'Species', 'Contig', 'Position']
        self._constant_covariate = params.constant_covariate
        self._max_pval_to_report = params.max_pval_to_report
        self._max_pval_to_detailed = params.max_pval_to_detailed
        self._ret_cov_fields = params.ret_cov_fields

    def _concat_collected_data(self, rets):
        out_fname = os.path.join(self._work_dir, '{}_tmp{}_N{}.h5'.
                                 format(self._result_files_prefix(), current_datetime_as_str(), len(rets)))

        with pd.HDFStore(out_fname, 'w') as hdf:
            if self._collect_only_snps:
                dfs_snps = defaultdict(lambda: list())
                for df in rets:
                    for species in df.columns.get_level_values('Species').unique():
                        dfs_snps[species].append(df)
                for species, dfs in dfs_snps.items():
                    dfs = pd.concat(dfs, axis='columns')
                    if self._compute_mrs:
                        dfs = dfs.T.groupby(['Type', 'Y', 'PvalCutoff', 'Species']).sum().T
                        for type, y, pval, species in [col for col in dfs[['mrs']].columns]:
                            dfs.loc[:, (type, y, pval, species)] /= dfs.loc[:, (type + '_num_snps', y, pval, species)]
                    elif not self._convert_snp_to_coef:
                        dfs = MBSNPLoader.pack_snp_data(dfs)
                    hdf[species] = dfs
            else:
                dfs_snps = {}
                df_snps = pd.concat([ret.iloc[:, -2] for ret in rets])
                samples_iden = ['RegistrationCode', 'Date'] if 'RegistrationCode' in df_snps.index.names else ['SampleName']
                for species, group in df_snps.groupby('Species'):
                    df = group.groupby(level=['Species', 'Contig', 'Position'] + samples_iden).first()
                    df.index = df.index.set_levels(df.index.levels[2].astype(int), level=2)
                    df = df.sort_index().unstack(level=['Species', 'Contig', 'Position']).apply(pd.to_numeric)
                    df.columns = pd.MultiIndex.from_arrays(list(itertools.chain(
                        [['snps'] * len(df.columns)],
                        [list(df.columns.get_level_values(i)) for i in range(len(df.columns.names))])),
                        names=['Type', 'Species', 'Contig', 'Position'])
                    dfs_snps[species] = df

                dfs_covariates = {}
                df_covariates = pd.concat([ret[[c for c in ret.columns[:-2] if c != 'CONSTANT']] for ret in rets])
                for species, group in df_covariates.groupby('Species'):
                    df = group.groupby(samples_iden).first()
                    num_covs = len(df.columns)
                    df.columns = pd.MultiIndex.from_arrays(
                        [['covariates'] * num_covs, df.columns, [None]*num_covs, [None]*num_covs],
                        names=['Type', 'Species', 'Contig', 'Position'])
                    dfs_covariates[species] = df

                dfs_ys = {}
                df_ys = pd.concat([ret.iloc[:, -1] for ret in rets]).unstack('Y')
                for species, group in df_ys.groupby('Species'):
                    df = df_ys.groupby(samples_iden).first()
                    num_ys = len(df.columns)
                    df.columns = pd.MultiIndex.from_arrays(
                        [['y']*num_ys, df.columns, [None]*num_ys, [None]*num_ys],
                        names=['Type', 'Species', 'Contig', 'Position'])
                    dfs_ys[species] = df

                for species in dfs_snps.keys():
                    hdf[species] = pd.concat([dfs_snps[species], dfs_covariates[species], dfs_ys[species]], axis='columns')

        return out_fname

    def _get_result_details(self, xy, test_x_col):
        # ['N', 'Mean', 'Median', 'Std', 'Q1', 'Q3', 'Min', 'Max']
        local_xy = xy.iloc[:, [test_x_col, -1]][xy.iloc[:, test_x_col] != MAF_MISSING_VALUE]
        return list(np.ravel(local_xy.iloc[:, -1].groupby((local_xy.iloc[:, 0] > MAF_1_VALUE / 2).astype(int)).agg(
            [np.size,
             np.mean,
             np.median,
             np.std,
             lambda x: np.percentile(x, 25),
             lambda x: np.percentile(x, 75),
             np.min,
             np.max,
             ])))

    def _fix_contig_pos(self, contig, pos):
        return contig, pos

    def _run(self, species, contig, x, y, y_col, num_covariates, run_per_contig=True, rets_counts=None, rets_detailed=None):
        def foreach_x(test_x_col, ret_total_tests, rets_detailed, is_y_valid_f):

            pos = x.columns[test_x_col]
            xy_cols[test_x_col_idx] = test_x_col
            mgwas = ols(x, xy_cols, test_x_col_idx, add_constant=False,
                        logistic_regression=logistic_regression,
                        is_y_valid_f=is_y_valid_f,
                        collect_data=self._collect_data,
                        ret_cov_fields=self._ret_cov_fields,
                        illegal_test_x_val=MAF_MISSING_VALUE)
            if mgwas is None:
                if self._verbose:
                    _log.info('this ols result is empty: sp {}, c {}, pos {}, log reg? {}'.
                              format(species, contig, pos, logistic_regression))
            else:
                if self._collect_data:
                    indices = self._indices + list(mgwas.index.names)
                    columns = list(mgwas.columns.get_level_values(0)[:-1]) + ['y']
                    fixed_contig, fixed_pos = self._fix_contig_pos(contig, pos)
                    return DataFrame(np.concatenate([np.tile([y_col, species, fixed_contig, fixed_pos], len(mgwas)).reshape(-1, 4),
                                                     mgwas.reset_index(mgwas.index.names)[mgwas.index.names].values,
                                                     mgwas.values], axis=1),
                                     columns=indices + columns).set_index(indices)
                else:
                    ret_total_tests[0] += 1
                    fixed_contig, fixed_pos = self._fix_contig_pos(contig, pos)
                    if self._max_pval_to_detailed is not None and mgwas.Pval <= self._max_pval_to_detailed:
                        rets_detailed.append([y_col, species, fixed_contig, fixed_pos, mgwas.R2, mgwas.Pval] +
                                             self._get_result_details(x, test_x_col))
                    if mgwas.Pval < 1e-10:
                        _log.info('{}:{}:{}: P={:.2e} y={} Coef={:.4f} N={}'.
                                  format(species, fixed_contig, fixed_pos, mgwas.Pval, y_col, mgwas.Coef, mgwas.N))
                    if mgwas.Pval<=self._max_pval_to_report:
                        return [y_col, species, fixed_contig, fixed_pos] + [getattr(mgwas, f) for f in self._ols_fields]

        if self._constant_covariate:
            x['CONSTANT'] = 1
            num_covariates += 1
        x['Y'] = y

        assert rets_counts is not None

        # optimize: remove empty y and covariates
        x = x.dropna(subset=x.columns[-num_covariates-1:])

        # optimize: filtering again the number of snps after joining with y
        x = self._filter_snps_by_stats_of_major(x, num_covariates + 1)

        # Important to check if y is binary AFTER the above rows that filter on empty ys
        y_binary = isbinary(x['Y'])
        logistic_regression = y_binary
        is_y_valid_f = None if self._is_y_valid_f is None else\
            lambda y: \
                self._is_y_valid_f(y, max_on_most_freq_val_in_col=self._max_on_most_freq_val_in_col,
                                   min_on_non_freq_val_for_y=self._min_on_non_freq_val_for_y, y_binary=y_binary)

        _log.info('Testing {} SNPs in {} subjects {} covariates for {}:{} in {}'.
                  format(x.shape[1] - num_covariates - 1, x.shape[0], num_covariates, species, contig, y_col))

        num_x_cols = len(x.columns)
        last_x_col = num_x_cols - num_covariates - 1
        covariate_cols_inds = list(range(last_x_col, num_x_cols - 1))
        y_col_idx = [num_x_cols - 1]
        xy_cols = covariate_cols_inds + [-1] + y_col_idx
        test_x_col_idx = len(xy_cols) - 2
        total_tests = [0]

        foreach_x_f = lambda x: foreach_x(x, total_tests, rets_detailed, is_y_valid_f)
        res = [r for r in map(foreach_x_f, range(0, last_x_col)) if r is not None]
        rets_counts.append((y_col, species, contig, total_tests[0]))
        return res

    def _run_result_df_columns(self):
        return self._indices + self._ols_fields

    def _result_files_prefix(self):
        data_suffix = '_data' if hasattr(self, '_collect_data') and self._collect_data else ''
        results_suffix = '' if self._results_file_suffix is None else self._results_file_suffix
        return 'mb_gwas{}{}'.format(results_suffix, data_suffix)

