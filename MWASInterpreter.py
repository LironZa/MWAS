import logging
import os
import re
from typing import List, Iterable
import functools

import pandas as pd
from pandas import HDFStore
from LabData.DataAnalyses.MBSNPs.Plots import manhattan_plot, qq_plot
from LabData.DataAnalyses.MBSNPs.Plots.calculate_snps_stats import count_and_test_enrichment, count_repeated_functions
from LabData.DataAnalyses.MBSNPs.Plots.volcano_plot import draw_volcano_plot_pyplot
from LabData.DataAnalyses.MBSNPs.Plots.pairwise_comparison_plot import draw_pairwise_comparison_plot
from LabData.DataAnalyses.MBSNPs.Plots.boxplot import BoxplotDrawer
from LabData.DataAnalyses.MBSNPs.Plots.summary_plot import SummaryDrawer
from LabData.DataAnalyses.MBSNPs.Plots.species_snps import SpeciesSNPsDrawer
from LabData.DataMergers.MultiDataLoader import MultiDataLoader
# from LabData.DataAnalyses.MBSNPs.Plots.pairwise_dists_plot import PairwiseDistsDrawer
from LabData.DataAnalyses.MBSNPs.annotationist import Annotationist
from LabData.DataAnalyses.MBSNPs.taxonomy import taxonomy_df
from LabData.DataLoaders.GeneAnnotLoader import GeneAnnotLoader, ANNOTS_03_2020_EXPANDED_NEW
from LabData.DataLoaders.MBSNPLoader import GutMBSNPLoader
from LabUtils.SeqUtils import translate_codon, codons_inv
from LabUtils.Utils import remove_illegal_file_basename_chars, mkdirifnotexists
from LabData import config_global as config
from LabData.DataAnalyses.MBSNPs.mwas_annots import add_surrounding_genes, add_codons, flatten_surrounding_genes, \
    add_amino_acids, add_taxonomy, add_gene_annotations, add_synonymous, add_sequences, add_snp_descriptions, add_stop
from LabUtils.snp_h5 import SNPH5
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
import numpy as np

_ANNOTS_FILE = ANNOTS_03_2020_EXPANDED_NEW

_log = logging.getLogger(__name__)


def gen_if_needed(gen_dir, force=True):
    def gen_if_needed_inner(gen_f):
        @functools.wraps(gen_f)
        def wrapper(self, *args, **kwargs):
            out_dir = os.path.join(self._out_dir, gen_dir)
            if force or not os.path.exists(out_dir):
                out_dir = self._make_interpret_dir(gen_dir)
                return gen_f(self, out_dir=out_dir, *args, **kwargs)
        return wrapper
    return gen_if_needed_inner

class MWASInterpreter(object):
    def \
            __init__(self, params, work_dir: str, mwas_fname: str = 'mb_gwas.h5',
                     out_dir: str = None, mwas_detailed_fname: str = 'mb_gwas_detailed.h5',
                     mwas_data_fname: str = 'mb_gwas_data.h5',
                     mwas_counts_fname: str = 'mb_gwas_counts.h5',
                     mwas_dists_fname: str = 'mb_dists.h5', save_by_y=False,
                     do_manhattan_plot=False, do_qq_plot=False,
                     do_snp_annotations=False,
                     annotate_all_snps=False, do_annotated_manhattan=False,
                     pval_col='Global_Bonferroni', pval_cutoff=0.05,
                     top_n_snps=None,
                     do_function_counts=False,
                     SNPs_to_plot_dct={},
                     body_site='Gut',
                     is_species_mwas=False,
                     ):
        """
        :param params:
        :param work_dir:
        :param mwas_fname:
        :param out_dir:
        :param do_manhattan_plot:
        :param do_mafs_plot:
        :param do_qq_plot:
        :param do_snp_annotations:
        :param annotate_all_snps: Bool, default: False. Instead of creating an annotated df only for top SNPs, create
                                                        for all? will take longer and will create a much larger file.
                                                        If the file already exists, will simply load it instead.
        :param do_annotated_manhattan: Bool, default: False. If True, will create a manhattan annotated by SNP function.
                                                                If True, annotate_all_snps must be True as well.
        :param pval_col:
        :param pval_cutoff:
        :param top_n_snps: int, optional. Do the analyses on the top_n_snps with the lowest p-values
        :param body_site: str, default: Gut
        """
        self._params = params
        self._work_dir = work_dir
        self._out_dir = work_dir if out_dir is None else out_dir
        self._save_by_y = save_by_y
        self._mwas_fname = mwas_fname
        self._mwas_detailed_fname = mwas_detailed_fname
        self._mwas_data_fname = os.path.join(self._work_dir, mwas_data_fname)
        self._mwas_counts_fname = mwas_counts_fname
        self._mwas_dists_fname = mwas_dists_fname
        self._pval_col = pval_col
        self._pval_cutoff = pval_cutoff
        self._top_n_snps = top_n_snps
        self._do_manhattan_plot = do_manhattan_plot
        self._do_qq_plot = do_qq_plot
        self._do_snp_annotations = do_snp_annotations
        self._annotate_all_snps = annotate_all_snps
        self._do_annotated_manhattan = do_annotated_manhattan
        self._do_function_counts = do_function_counts
        self._SNPs_to_plot = SNPs_to_plot_dct
        self._body_site = body_site
        self._is_species_mwas = is_species_mwas

        # self._mbsnp_loader = GutMBSNPLoader() if mbsnp_loader is None else mbsnp_loader
        self._cache_dir = os.path.join(config.cache_dir, 'MWASInterpreter')

    @property
    def data_cache_name(self):
        return 'mwas_interpreter'

    def _get_significant_mwas(self, df) -> pd.DataFrame:
        df = df[df[self._pval_col] <= self._pval_cutoff]
        _log.info('Found {} mwas with {} <= {}'.format(len(df), self._pval_col, self._pval_cutoff))
        return df

    def _load_mwas_file(self, mwas_fname) -> pd.DataFrame:
        fname = os.path.join(self._work_dir, mwas_fname)
        if not os.path.exists(fname):
            return pd.DataFrame()

        df: pd.DataFrame = pd.read_hdf(fname)

        if df.empty:
            return df

        if self._pval_cutoff < 1. and not self._annotate_all_snps:
            df = self._get_significant_mwas(df)

        if self._top_n_snps is not None:
            df = df.sort_values('Pval').iloc[:self._top_n_snps]

        '''
        NOTE:
        Replacing 0 p-values with 1e-300, as many of the downstream analyses and visualizations don't work well with 0.
        Original p-values, in the mb_gwas.h5 file, remain unchanged.
        We recommend checking how many of the MWAS p-values are == zero, as having more than a few may imply that
        something went wrong in the regression/ input variables.        
        '''
        df['Pval'] = df['Pval'].replace(to_replace=0, value=1e-300)

        # Remove the part from the contig key
        idx = df.index.names
        df.reset_index(inplace=True)
        df['ContigWithParts'] = df['Contig']
        df.Contig = remove_contig_parts(df.Contig.values)
        df.set_index(idx, inplace=True)
        return df

    def _load_mwas(self) -> pd.DataFrame:
        if hasattr(self, '_mwas_df'):
            return self._mwas_df
        self._mwas_df = self._load_mwas_file(self._mwas_fname)
        if self._mwas_df.empty:
            return self._mwas_df
        self._mwas_num_tests = int((self._mwas_df['Global_Bonferroni'] / self._mwas_df['Pval']).max())
        return self._mwas_df

    def _load_mwas_detailed(self) -> pd.DataFrame:
        if hasattr(self, '_mwas_detailed_df'):
            return self._mwas_detailed_df
        self._mwas_detailed_df = self._load_mwas_file(self._mwas_detailed_fname)
        if self._mwas_detailed_df.empty:
            return self._mwas_detailed_df
        self._mwas_detailed_df['N'] = self._mwas_detailed_df['MajorN'] + self._mwas_detailed_df['MinorN']
        self._mwas_detailed_df['MedianDiff'] = self._mwas_detailed_df['MajorMedian'] - self._mwas_detailed_df['MinorMedian']
        self._mwas_detailed_df = self._mwas_detailed_df.sort_index()
        return self._mwas_detailed_df

    def _load_mwas_counts(self):
        if hasattr(self, '_mwas_counts_df'):
            return self._mwas_counts_df
        self._mwas_counts_df = pd.read_hdf(os.path.join(self._work_dir, self._mwas_counts_fname))
        if self._mwas_counts_df.empty:
            return 0, 0, 0
        self._mwas_num_tests = self._mwas_counts_df.sum()[0]
        self._mwas_num_snps_tested = self._mwas_counts_df.groupby('Y').sum().max()[0]
        self._mwas_num_species_tested = len(self._mwas_counts_df.index.unique('Species'))
        return self._mwas_num_tests, self._mwas_num_snps_tested, self._mwas_num_species_tested

    def run(self):
        self._load_mwas()
        _log.info('Loaded {} mwas for {} Ys'.
                  format(len(self._mwas_df), len(self._mwas_df.index.unique('Y'))))
        print(self._mwas_df.shape, str(len(self._mwas_df.reset_index()['Species'].unique())) + ' different species')
        if self._do_qq_plot:
            self._qq_plot()
        if self._do_snp_annotations:
            if self._do_annotated_manhattan:
                self._annotate_all_snps = True
            self._snp_annotations()
        if self._do_annotated_manhattan:
            assert self._do_snp_annotations, 'plotting the annotated Manhattan requires do_snp_annotations to be TRUE'
            self._annotated_manhattan_plot()
        if self._do_function_counts:
            self._function_counts()
        if self._do_manhattan_plot:
            self._manhattan_plot()

    @staticmethod
    def pick_top_annotation(snp_annots):
        ''' priorities:
        1. 'Current' annotation that is non-synonymous
        2. 'Current' annotation that is synonymous
        3. if the SNP is not within any gene, pick the one that it is closest to.
        #TODO: consider prioritizing downstream genes over upstream
        '''
        if len(snp_annots) == 1:
            return snp_annots.iloc[0]
        if 'Current' in snp_annots['GeneRelation'].values:
            if ((snp_annots['GeneRelation'] == 'Current').sum() > 1) and (snp_annots['IsSynonymous'] == False).sum() > 0:
                return snp_annots.loc[(snp_annots['GeneRelation'] == 'Current') & (snp_annots['IsSynonymous'] == False)].iloc[0]
            return snp_annots.sort_values('GeneRelation').iloc[0]
        return snp_annots.sort_values('GeneDistance').iloc[0]

    def _add_snp_annotations(self, mwas_df, only_top_annot_per_snp=False):
        ''' NOTE:
        Some SNPs that exist in the mwas_df will be missing from the output DF,
        probably since their contigs were too short to include any gene
        '''
        self._load_gene_annotations()
        self._load_annotations_list()
        if 'Y' in mwas_df.index.names:
            mwas_df = mwas_df.reset_index('Y')
        mwas_df = add_surrounding_genes(mwas_df, annotations=self._annotations_list)
        # mwas_df = self._set_contig_index(mwas_df, 'Contig', 'ContigWithParts')
        mwas_df = add_codons(mwas_df, annotations=self._annotations_list)
        # mwas_df = self._set_contig_index(mwas_df, 'Contig', 'ContigWithParts')
        mwas_df = flatten_surrounding_genes(mwas_df)
        mwas_df = add_amino_acids(mwas_df)
        mwas_df = add_taxonomy(mwas_df, mb_species_db_basename=config.mb_species_db_basename, body_site=self._body_site)
        mwas_df = add_gene_annotations(mwas_df, annotations=self._gene_annotations)
        mwas_df = add_synonymous(mwas_df)
        if only_top_annot_per_snp:
            mwas_df = mwas_df.reset_index(level=['GeneRelation', 'strand'])\
                .groupby(['Y', 'Species', 'Contig', 'Position']).apply(self.pick_top_annotation)
            mwas_df = mwas_df.set_index(['GeneRelation', 'strand'], append=True)
        mwas_df = add_stop(mwas_df)
        mwas_df = add_snp_descriptions(mwas_df)
        return mwas_df

    def _snp_annotations(self):
        annotations_dir = os.path.join(self._out_dir, 'snp_annotations')
        if self._annotate_all_snps and os.path.exists(os.path.join(annotations_dir, 'snp_annotations_ALL.csv')):
            print('not creating a new annotations df, loading existing one')
            df = pd.read_csv(os.path.join(annotations_dir, 'snp_annotations_ALL.csv'), index_col=[0, 1, 2, 3])
            assert df.index.names == ['Y', 'Species', 'Contig', 'Position']
            self._annotated_mwas_df = df
            return

        else:
            os.makedirs(annotations_dir, mode=0o744, exist_ok=True)
            if self._annotate_all_snps:
                original_df = self._mwas_df
            else:
                original_df = self._get_significant_mwas(self._mwas_df)


            original_df = original_df.loc[:, ['N', 'Coef', 'ContigWithParts', 'Pval', 'Global_Bonferroni',
                                              'min_mn', 'maj_mn', 'min_n', 'maj_n']]
            original_df['means_diff'] = original_df['min_mn'].sub(original_df['maj_mn'], axis='index')
            original_df['abs_means_diff'] = np.abs(original_df['means_diff'])

            annotated_df = self._add_snp_annotations(original_df, only_top_annot_per_snp=True)
            annotated_df = annotated_df.reset_index(level=['GeneRelation', 'strand'], drop=False)

            cds_snps = annotated_df.loc[annotated_df['IsSynonymous'].notnull()]
            annotated_df = annotated_df.join((cds_snps['IsSynonymous'] == False).rename('NonSymMutation'))
            annotated_df['GeneDistance'] = annotated_df.apply(
                lambda x: -1 * np.abs(x['GeneDistance']) if x['GeneRelation'] != 'Current' else x['GeneDistance'], axis=1)

        # add to the annotated_df the SNPs that are missing (no genes in these contigs), from the original_df
        cols_order = annotated_df.columns
        df = annotated_df.combine_first(original_df)
        df = df.loc[:, cols_order]
        assert len(df) == len(original_df)

        self._annotated_mwas_df = df

        if not self._annotate_all_snps:
            df.to_csv(os.path.join(annotations_dir, 'snp_annotations.csv'))
            df.reset_index().to_excel(os.path.join(annotations_dir, 'snp_annotations.xlsx'))

        else:     # save the expanded, then filter for significant and save the smaller ('regular') one as well
            df2 = self._get_significant_mwas(df)

            important_cols = ['GeneID', 'GeneRelation', 'GeneDistance', 'ContigWithParts', 'feature', 'N', 'Coef', 'Pval',
                              'Global_Bonferroni', 'taxa', 'gene', 'product', 'NonSymMutation', 'strand',
                              'min_mn', 'maj_mn', 'means_diff'] # , 'eggNOG OGs', 'KEGG_ko', 'Coef_SE',
            df = df.loc[:, important_cols]
            df.to_csv(os.path.join(annotations_dir, 'snp_annotations_ALL.csv'))
            if not df2.empty:
                df2.to_csv(os.path.join(annotations_dir, 'snp_annotations.csv'))
                df2.reset_index().to_excel(os.path.join(annotations_dir, 'snp_annotations.xlsx'))
        return

    def _manhattan_plot(self):
        manhattan_plots_dir = os.path.join(self._out_dir, 'manhattan_plots')
        manhattan_df = self._annotated_mwas_df if hasattr(self, '_annotated_mwas_df') else self._mwas_df
        for y, group in manhattan_df.groupby('Y'):
            if self._save_by_y:
                manhattan_plots_dir = os.path.join(self._out_dir, y)
            os.makedirs(manhattan_plots_dir, mode=0o744, exist_ok=True)
            file_name = os.path.join(manhattan_plots_dir, f'manhattan_{remove_illegal_file_basename_chars(y)}')
            manhattan_plot.draw_manhattan_plot(group, out_file=file_name, title=y,
                                               pval_col=self._pval_col, pval_cutoff=self._pval_cutoff)

    def _build_multiple_ys_legend(self, max_separate_ys):
        ys_order = self._mwas_df.groupby('Y')['Pval'].min().sort_values().index.get_level_values('Y')

        colors = get_cmap('tab20').colors
        self._ys_color_dict = {y: colors[i % len(colors)] for i, y in enumerate(ys_order[:max_separate_ys])}
        self._ys_color_dict.update({y: 'lightgray' for y in ys_order[max_separate_ys:]})

        ys_legend = list(ys_order) + ['All other ys']
        self._ys_color_dict['All other ys'] = 'lightgray'
        self._ys_legend = [Line2D([0], [0], linewidth=0, label=y, alpha=0.5, markersize=10, marker='o',
                                  markeredgewidth=0, markerfacecolor=self._ys_color_dict[y], markeredgecolor='white')
                           for y in ys_legend]

    def _draw_multiple_ys(self, df, **kwargs):
        d = dict()
        d['marker'] = 'o'
        # d['s'] = -np.log10(df[kwargs['pval_col']]).values / 2
        d['facecolor'] = df.reset_index('Y')['Y'].map(self._ys_color_dict).values
        # d['edgecolor'] = 'black'
        # d['linewidths'] = 1
        d['alpha'] = 0.5

        return df, d, self._ys_legend

    def _make_interpret_dir(self, dir_name):
        dir = os.path.join(self._out_dir, dir_name)
        os.makedirs(dir, mode=0o744, exist_ok=True)
        return dir

    def _load_gene_annotations(self):
        if not hasattr(self, '_gene_annotations'):
            if config.mb_species_db_basename == 'segata':
                annotations_fname = os.path.join(config.annotations_dir, 'annot_all_03_2020_expanded_0_based_new_eggnog.csv')
            else:
                annotations_fname = \
                    os.path.join(config.annotations_dir, 'Segal_annots', 'Segal_annots_2021_07_31_prokka_eggnog.csv')
            self._gene_annotations = GeneAnnotLoader(annotations_fname, body_site=self._body_site).get_annot()
        return self._gene_annotations

    def _load_annotations_list(self):
        if not hasattr(self, '_annotations_list'):
            self._load_gene_annotations()
            self._annotations_list = Annotationist(self._gene_annotations, body_site=self._body_site, load_sequences=False)
        return self._annotations_list

    def _load_snp_desc(self):
        annotations_fname = os.path.join(self._out_dir, 'annotations', 'snps_gene_annotations.h5')
        if not os.path.exists(annotations_fname):
            return None
        return pd.read_hdf(annotations_fname).groupby(['Y', 'Species', 'Contig', 'Position']).first()

    def _load_snp_annotations(self):
        annotations_fname = os.path.join(self._out_dir, 'annotations', 'snps_gene_annotations.h5')
        if not os.path.exists(annotations_fname):
            return None
        return pd.read_hdf(annotations_fname)

    def _set_contig_index(self, df, contig_index_name, contig_col_name):
        org_index_names = df.index.names
        assert contig_index_name in org_index_names
        return df.reset_index().\
            rename({contig_index_name: contig_col_name, contig_col_name: contig_index_name}, axis='columns').\
            set_index(org_index_names)

    @staticmethod
    def remove_contig_part_from_df(df):
        idx = df.index.names
        df.reset_index(inplace=True)
        df.Contig = remove_contig_parts(df.Contig.values)
        df.set_index(idx, inplace=True)
        return df

    def _annotated_manhattan_plot(self):
        manhattan_plots_dir = self._make_interpret_dir('manhattan_plots')
        for y, group in self._annotated_mwas_df.groupby('Y'):
            if self._save_by_y:
                manhattan_plots_dir = os.path.join(self._out_dir, y)
            os.makedirs(manhattan_plots_dir, mode=0o744, exist_ok=True)
            file_name = os.path.join(manhattan_plots_dir, remove_illegal_file_basename_chars(y)+'_annot') #TODO: change ppt back
            manhattan_plot.draw_annotated_manhattan_plot(group, file_name, pval_col='Pval', pval_cutoff=.1,  #TODO: change cutoff back to 1
                                                         color_by_func=False, ppt=False, summ_txt=False, #TODO: change ppt back
                                                         annot_df=self._annotated_mwas_df, label_genes=True)

    def _qq_plot(self):
        qq_plots_dir = self._make_interpret_dir('qq_plots')
        for y, group in self._mwas_df.groupby('Y'):
            if self._save_by_y:
                qq_plots_dir = os.path.join(self._out_dir, y)
            os.makedirs(qq_plots_dir, mode=0o744, exist_ok=True)
            out_file = os.path.join(qq_plots_dir, f'qq_{remove_illegal_file_basename_chars(y)}')
            qq_plot.qqplot(group.Pval, 'P-Values for ' + y, out_file)

    def _function_counts(self):
        ### examples for other values you may want to use for function_col: 'gene', 'best_og_desc_NEW', 'narr_og_desc_NEW'
        assert self._annotate_all_snps, 'for testing enrichment, annotation data for *all* SNPs is required.'

        #TODO: improve the way this works with different function_cols automatically. Doing this since the 'eggNOG OGs' col wasn't saved before
        if 'eggNOG OGs' not in self._annotated_mwas_df.columns:
            self._load_gene_annotations()
            self._annotated_mwas_df = add_gene_annotations(self._annotated_mwas_df, annotations=self._gene_annotations, rsuffix='_n')

        outdir = os.path.join(self._out_dir, 'snp_annotations')
        for y, group in self._annotated_mwas_df.groupby('Y'):
            count_repeated_functions(snps_df=group, outdir=outdir, phen=y,
                                     function_col='product', pval_col=self._pval_col, pval_cutoff=self._pval_cutoff,
                                     max_dist=0, only_nonsyn=True)
            count_repeated_functions(snps_df=group, outdir=outdir, phen=y,
                                     function_col='product', pval_col=self._pval_col, pval_cutoff=self._pval_cutoff,
                                     max_dist=0, only_nonsyn=False)
            count_repeated_functions(snps_df=group, outdir=outdir, phen=y,
                                     function_col='NOG', pval_col=self._pval_col, pval_cutoff=self._pval_cutoff,
                                     max_dist=0, only_nonsyn=True)

    def _get_gene_annots(self):
        if not hasattr(self, '_gene_annots'):
            _log.info('Loading gene annotations')
            self._gene_annots = GeneAnnotLoader(_ANNOTS_FILE).get_annot()
        return self._gene_annots


def remove_contig_parts(contigs: Iterable[str]) -> List[str]:
    """Returns the given contig names without the trailing '_P###'."""
    # contig_part_re = re.compile('_P\\d+$')
    contig_part_re = re.compile('^C_\\d+')

    # Maps old contig name to new contig name, to avoid repetition.
    cache = {}

    def cached_name(contig):
        if contig not in cache:
            # cache[contig] = contig_part_re.sub('', contig)
            cache[contig] = contig_part_re.findall(contig)[0]
        return cache[contig]

    return [cached_name(x) for x in contigs]


def assign_gene_id_and_distance(df: pd.DataFrame):
    """Converts annotationist gene assignments to old style, with gene ID and
    distance."""
    # Names are p/m for plus/minus, c/u/d for current/upstream/downstream,
    # gi/gp/gd for gene ID/position/distance.
    cols = list(df.columns)
    pcgi, pcgp, pugi, pugd, pdgi, pdgd, mcgi, mcgp, mugi, mugd, mdgi, mdgd = [
        cols.index(x) for x in Annotationist.lookup_result_column_names()]

    genes = []
    current_both = 0
    several_nearest = 0

    for row in df.values:
        row = [nan_to_none(x) for x in row]
        if row[pcgi] and not row[mcgi]:  # Has current only in plus.
            genes.append((row[pcgi], -row[pcgp]))
        elif row[mcgi] and not row[pcgi]:  # Has current only in minus.
            genes.append((row[mcgi], -row[mcgp]))
        elif row[pcgi] and row[mcgi]:  # Has current in both strands.
            # Select plus arbitrarily and report.
            genes.append((row[pcgi], -row[pcgp]))
            current_both += 1
        else:  # No current, look for nearest
            enclosing_genes = [
                (row[pugi], row[pugd]),
                (row[pdgi], row[pdgd]),
                (row[mugi], row[mugd]),
                (row[mdgi], row[mdgd]),
            ]
            # Remove NAs.
            enclosing_genes = [x for x in enclosing_genes if x[0]]
            if len(enclosing_genes) == 0:
                genes.append((None, None))
                continue
            enclosing_genes = sorted(enclosing_genes, key=lambda x: x[1])
            genes.append(enclosing_genes[0])

            distance = enclosing_genes[0][1]
            if len(enclosing_genes) > 1 and \
                    enclosing_genes[1][1] <= distance+20:
                several_nearest += 1

    df['GeneID'] = [x[0] for x in genes]
    df['GeneDistance'] = [x[1] for x in genes]
    return current_both, several_nearest


def nan_to_none(x):
    """A convenience function for dealing with nans."""
    if isinstance(x, float) and pd.isna(x):
        return None
    return x
