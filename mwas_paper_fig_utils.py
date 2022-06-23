'''
Utils for figures in our 1st MWAS paper
'''
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from LabData import config_global as config
from matplotlib import cm
from matplotlib.colors import ListedColormap

BASE_DIR = Path(config.analyses_dir)
FIGS_DIR = BASE_DIR.joinpath('Figures')
FIGS_DIR.mkdir(parents=True, exist_ok=True)

phens_list = ['age', 'sex', 'bmi']

phens_labels = {'background': 'All tested SNPs',
                'age': 'Age',
                'sex': 'Sex',
                'bmi': 'BMI',
                }

phen_labels_expanded = {**phens_labels,
                        **{'gender': phens_labels['sex']}}

mid_sentence_labels = {'Age': 'age',
                       'Sex': 'sex',
                       'BMI': 'BMI'}

phen_units_dict = {'age': 'years',
                   'sex': 'male frequency',
                   'gender': 'male frequency',
                   'bmi': 'BMI points',
                   }

phen_official_to_usage_dict = {'age': 'age', 'gender': 'sex', 'bmi': 'bmi'}

############## Colors ##############
class MWASPaperColors(object):
    def __init__(self):
        self.red = 'crimson'  # '#F54F29'
        self.grey = 'dimgrey'
        self.light_grey = 'lightgrey'
        self.main_color_1 = '#483B59'
        self.main_color_2 = '#FF974F'
        self.main_color_3 = '#9C9B7A'
        self.phens_colors = {'background': 'grey',
                             'age': '#405952',
                             'sex': '#9C9B7A',
                             'gender': '#9C9B7A',
                             'bmi': '#FFD393',
                             }  # '#405952' , '#9C9B7A' , '#FFD393' , '#FF974F' , '#F54F29'
        self.type_colors = {'Intergenic': '#FF974F',
                            'Non-protein coding region (rRNA, tRNA, etc.)': '#FFD393',
                            'Protein coding - synonymous': '#9C9B7A',
                            'Protein coding - nonsynonymous': '#405952'
                            }

    def set_bplot_params(self):
        plt.rc('boxplot.medianprops', color=self.red, linewidth=2)  # '#F54F29'
        plt.rc('boxplot.boxprops', color='black', linewidth=2)  # '#13929E'
        plt.rc('boxplot.whiskerprops', color='black', linestyle='-', linewidth=1.2)
        plt.rc('boxplot.capprops', color='black')


    def create_greys_cmap(self):
        greys_modified = cm.get_cmap('Greys', 256)
        return ListedColormap(greys_modified(np.linspace(.2, .95, 256)))


##############################################
def round_pval(pvalue, p_thresh=None):
    if p_thresh is not None and pvalue > p_thresh:
        return "n.s."
    if pvalue == 0:
        return r"$p < 10^{{{}}}$".format(-300)
    if pvalue >= 0.01:
        return r"$p = {:.1g}$".format(pvalue)
    power = int(np.ceil(np.log10(pvalue)))
    return r"$p < 10^{{{}}}$".format(power)


def create_snp_label(snp_df_row, species_cont_pos=None, taxonomy=None, gene_annot_col='product'):
    """ SGB, contig(?), (predicted species: )
        ['intergenic' / product + 'non/synonymous' / product (if RNA gene)], number of SNPs grouped?
    """
    if taxonomy is None:
        if config.mb_species_db_basename == 'segata':
            taxonomy = snp_df_row['a_spec'].lstrip('s_')
            # taxonomy = r"$\bf{{{}}}$".format(snp_df_row['a_spec'].lstrip('s_').replace('_', '\ '))
        else:
            tax = snp_df_row['taxa']
            taxonomy = tax
            # taxonomy = r"$\bf{{{}}}$".format(taxonomy)
    taxonomy = taxonomy.replace(' ', '\ ').replace('_', '\ ')
    taxonomy = r"$\bf{{{}}}$".format(taxonomy)

    if species_cont_pos is None:
        taxonomy += f"\n[{snp_df_row['Species']}, {snp_df_row['Contig']}]"
    else:
        taxonomy += f"\n[{species_cont_pos[0]}, {species_cont_pos[1]}, pos {species_cont_pos[2]}]"


    # P value from the regression
    pval_s = 'regression ' + round_pval(snp_df_row['Global_Bonferroni'], p_thresh=.05)


    # unannotated/ unknown
    if np.isnan(snp_df_row['GeneDistance']):
        return taxonomy + '\n' + 'unknown annotation' + '\n' + pval_s
    # intergenic
    if snp_df_row['GeneDistance'] < 0:
        return taxonomy + '\n' + 'intergenic' + '\n' + pval_s
    # RNA gene
    if snp_df_row['feature'] != 'CDS':
        return taxonomy + '\n' + snp_df_row['product'] + '\n' + pval_s
    # protein coding gene:
    syn = 'non-synonymous' if snp_df_row['NonSymMutation'] else 'synonymous'

    gene_annot = snp_df_row[gene_annot_col]
    if type(gene_annot) is not str:
        gene_annot = 'unknown gene'
    return taxonomy + '\n' + f"{gene_annot}, {syn}" + '\n' + pval_s


def get_segal_species_label(segal_id, tax_df):
    species = tax_df.loc[segal_id, 'Species']
    if species == 'unknown':
        species = segal_id
    return species


def create_snp_stats(snp_df_row, p_thresh=None):
    pval_s = round_pval(snp_df_row['Pval'], p_thresh=p_thresh)
    print(snp_df_row['Pval'], pval_s)
    n_s = f"n={int(snp_df_row['N'])}"
    return pval_s + ', ' + n_s


def create_snp_stats2(snp_df_row, p_thresh=None, p_col=None):
    if pd.isna(snp_df_row[p_col]):
        return ''
    pval_s = round_pval(snp_df_row[p_col], p_thresh=p_thresh)
    print(snp_df_row[p_col], pval_s)
    n_s = f"n={int(snp_df_row['N'])}"
    return pval_s + ', ' + n_s


def choose_top_snps(snps_df, opt_by='Pval', n=6, unique_by=['Species', 'GeneID']):
    """
    Note: if you only want in-gene SNPs, CDS SNPs, SNPs with gene tag, etc., filter the df before calling this function.
    :param snps_df:
    :param opt_by: the smallest pvalues? the largest coefficient? largest phenotype difference btw the min/maj allele groups?
    :param n: how many best SNPs do you want?
    :param unique_by: only one (best) SNP per species? choose ['Species']. Per species-gene? choose ['Species', 'GeneID']. etc.
    :return:
    """
    if opt_by == 'Pval':
        return snps_df.sort_values('Pval', ascending=True).drop_duplicates(subset=unique_by, keep='first').head(n)
    elif opt_by == 'Coef':
        snps_df.loc[:, 'abs_coef'] = snps_df.apply(lambda x: np.abs(x['Coef']), axis=1)
        return snps_df.sort_values('abs_coef', ascending=False).drop_duplicates(subset=unique_by, keep='first').head(n)
    elif opt_by == 'diff':
        return snps_df.sort_values('abs_means_diff', ascending=False).drop_duplicates(subset=unique_by, keep='first').head(n)
    elif opt_by == 'clump':
        snps_df = snps_df.loc[snps_df['post_clumping'] == 1]
        return snps_df.sort_values('Pval', ascending=True).head(n)
    else:
        print('this optimization parameter is not implemented, choose another')
        raise


def compute_snps_effect_size(mwas_df):
    mwas_df['abs_means_diff'] = np.abs(mwas_df['min_mn'].sub(mwas_df['maj_mn'], axis='index'))
    return mwas_df
