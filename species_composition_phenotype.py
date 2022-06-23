from pathlib import Path
import pandas as pd
import numpy as np
from LabData import config_global as config
from scipy import stats
from LabData.DataAnalyses.MBSNPs.taxonomy import taxonomy_df


BASE_DIR = Path(config.analyses_dir)
FIGS_DIR = BASE_DIR.joinpath('Figures', 'species_composition')
FIGS_DIR.mkdir(parents=True, exist_ok=True)

MB_ABUNDANCE_DF_P = BASE_DIR.joinpath('MWAS_robust_samples_list/BasicMWAS_10K_singleSamplePerReg_20223003_234555_mb_species_abundance.csv')
ROBUST_SAMPLES_DF_P = BASE_DIR.joinpath('MWAS_robust_samples_list/BasicMWAS_10K_singleSamplePerReg_20223003_234555.csv')


def test_one_phen_all_species(phen, mb_gwas_df_p, name=''):
    # for the phenotype, choose the species to test -- load the mb_gwas.h5, filter for Bon<.05, and take list of species
    phen_df = pd.read_hdf(mb_gwas_df_p)
    phen_df = phen_df.loc[phen_df['Global_Bonferroni'] <= .05]
    sig_species = phen_df.groupby('Species').size()

    # load the species composition table of our robust samples list
    species_composition = pd.read_csv(MB_ABUNDANCE_DF_P, index_col=0)
    phenotypes = pd.read_csv(ROBUST_SAMPLES_DF_P, index_col=0).rename(columns={'bt__hba1c': 'hba1c', 'gender': 'sex'})

    # for each phenotype, for each species in its list, run an association test
    # put the results in one table, add a column of Bonferroni correction.
    results = pd.DataFrame(index=sig_species.index)

    ### for each phenotype, for each species in its list, compare the phenotype of people w/ and w/o the bacteria
    ### statistical test: scipy.stats.mannwhitneyu(x, y, use_continuity=True, alternative='two-sided', axis=0, method='auto')
    if name == 'presence':
        for species_id in sig_species.index:
            abundance = species_composition[species_id]
            wo_bac = abundance.loc[abundance == 0.0001].index
            w_bac = abundance.loc[abundance > 0.0001].index
            phen_wo_bac = phenotypes.loc[wo_bac, phen].dropna()
            phen_w_bac = phenotypes.loc[w_bac, phen].dropna()
            results.loc[species_id, f'pval_{phen}'] = stats.mannwhitneyu(phen_wo_bac, phen_w_bac).pvalue

    ### correlation of species relative abundance & phenotype, where the species exists
    if name == 'abundance_if_exists':
        for species_id in sig_species.index:
            abundance = species_composition[species_id]
            abundance_w_bac = abundance.loc[abundance > 0.0001]
            phen_w_bac = phenotypes.loc[abundance_w_bac.index, phen].dropna()
            abundance_w_bac = abundance_w_bac.loc[phen_w_bac.index]
            results.loc[species_id, f'pval_{phen}'] = stats.spearmanr(phen_w_bac, abundance_w_bac).pvalue

    ### correlation of species relative abundance & phenotype, including samples with zero abundance
    if name == 'log10abundance':
        for species_id in sig_species.index:
            abundance = species_composition[species_id]
            phen_w_bac = phenotypes.loc[abundance.index, phen].dropna()
            abundance = abundance.loc[phen_w_bac.index]
            log10abundance = np.log10(abundance)
            results.loc[species_id, f'pval_{phen}'] = stats.spearmanr(phen_w_bac, log10abundance).pvalue


    num_hypotheses = len(sig_species)
    results[f'bonferroni_{phen}'] = results.apply(lambda x: x[f'pval_{phen}']*num_hypotheses, axis=1)

    tax_df = taxonomy_df(level_as_numbers=False).set_index('SGB')[['Species']].rename(columns={'Species': 'taxa'})
    # ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    # print(tax_df.loc[results.index])
    results = results.join(tax_df, on='Species')

    print(results.sort_values(f'bonferroni_{phen}'))
    results.to_csv(FIGS_DIR.joinpath(f'species_composition_{name}_{phen}.csv'))
    return



if __name__ == '__main__':
    name = 'log10abundance'

    mb_gwas_df_p = BASE_DIR.joinpath('20220402_234043_mwas_bmi_10K', 'mb_gwas.h5')
    test_one_phen_all_species('bmi', mb_gwas_df_p, name=name)


