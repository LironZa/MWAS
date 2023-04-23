from LabData import config_global as config
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

MWAS_DIR = Path(config.analyses_dir).joinpath('20220402_234043_mwas_bmi_10K')
BASE_DIR = Path(config.analyses_dir)
FIGS_DIR = BASE_DIR.joinpath('Figures', 'enrichment')
FIGS_DIR.mkdir(parents=True, exist_ok=True)


### enrichment: are the significant SNPs in Rep_3066 (C_257) enriched for COG=C?
def rep_3066_cog_enrichment():
    ## load the full annotated SNPs df of rep_3066
    full_df = pd.read_csv(MWAS_DIR.joinpath('snp_annotations', 'Rep_3066_ALL.csv'))
    contig_df = full_df.loc[full_df['Contig'] == 'C_257']
    significant_contig_df = contig_df.loc[contig_df['Global_Bonferroni'] <= .05]

    ## count COG=C vs ALL in the full rep_3066 df
    all_all = len(full_df)
    all_cogC = (full_df['best_og_cat'] == 'C').sum()

    ## count COG=c vs ALL in the SIGNIFICANT SNPs
    contig_all = len(significant_contig_df)
    contig_cogC = (significant_contig_df['best_og_cat'] == 'C').sum()

    ## calculate statistics of enrichment
    enrichment = is_enriched_or_depleted(drawn_special=contig_cogC,
                                         drawn_all=contig_all,
                                         total_special=all_cogC,
                                         total_all=all_all)
    print(f"all SNPs, [all; COG=C]: [{all_all}; {all_cogC}]. Rep_3066 C_257 SNPs, [all; COG=C]: [{contig_all}; {contig_cogC}]")
    print(enrichment)
    return


############################# Statistical Tests ###################################################################


def is_enriched(drawn_special, drawn_all, total_special, total_all):
    return stats.hypergeom.sf(drawn_special-1, total_all, total_special, drawn_all)


def is_depleted(drawn_special, drawn_all, total_special, total_all):
    return stats.hypergeom.cdf(drawn_special, total_all, total_special, drawn_all)


def is_enriched_or_depleted(drawn_special, drawn_all, total_special, total_all, none_drwn_spec_as_zero=True):
    """
    enrichment analysis using hyper-geometric distribution.
    typical question: Is it exciting that when looking at a set of 100 genes which were found associated with host
    cholesterol, 50 relate to amino acid metabolism, given that I tested a total of 1000 genes, out of which 200 are
    involved in amino acid metabolism?
    :param drawn_special: int. The number of objects that were highlighted in your analysis
                            ('all the genes associated with cholesterol')
    :param drawn_all: int. The number of objects highlighted in your analysis, which also belong to the 'interesting'
                            type ('gene associated with cholesterol and also part of amino acid metabolism')
    :param total_special: int. how many of the objects you tested also belong to the 'interesting' type?
                            (how many of the tested genes are associated with amino acid metabolism?)
    :param total_all: int. How many objects did you test, overall?
    :param none_drwn_spec_as_zero: If 'drawn_special' is 'none', can we treat it as zero?
    :return: ( depleted or enriched (str), (fold_change, p-value))
    """
    if np.isnan(drawn_special):
        if none_drwn_spec_as_zero:
            drawn_special = 0
        else:
            return np.NaN
    fold_change = (float(drawn_special) / drawn_all) / (float(total_special) / total_all)
    if fold_change >= 1:
        enriched_p = is_enriched(drawn_special, drawn_all, total_special, total_all)
        return 'enriched', fold_change, enriched_p
    else:
        depleted_p = is_depleted(drawn_special, drawn_all, total_special, total_all)
        return 'depleted', fold_change, depleted_p


if __name__ == '__main__':
    rep_3066_cog_enrichment()