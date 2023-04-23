from LabData import config_global as config
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.power import tt_solve_power

MWAS_DIR = Path(config.analyses_dir).joinpath('20220402_234043_mwas_bmi_10K')
BASE_DIR = Path(config.analyses_dir)
FIGS_DIR = BASE_DIR.joinpath('Figures', 'Power')
FIGS_DIR.mkdir(parents=True, exist_ok=True)



def calc_n():
    alpha = .05 / 12686191
    power = .9

    # df = pd.read_csv(MWAS_DIR.joinpath('snp_annotations', 'annotated_clumped_0.3.csv'))
    df = pd.read_csv(MWAS_DIR.joinpath('clumped_mb_gwas_0.3.csv'))

    df['standardized_effect'] = df.apply(lambda x:
                                         np.abs(x['Coef']) / ((np.abs(x['Coef_025'] - x['Coef_975'])/3.924) * np.sqrt(x['N'])),
                                         axis=1)

    df[f'min_N_{power}_bon_all'] = df.apply(lambda x: tt_solve_power(effect_size=x['standardized_effect'],
                                                                     nobs=None, alpha=alpha, power=power, alternative='two-sided'),
                                            axis=1)

    df[f'min_N_{power}_bon_40'] = df.apply(lambda x: tt_solve_power(effect_size=x['standardized_effect'],
                                                                    nobs=None, alpha=.05/40, power=power, alternative='two-sided'),
                                           axis=1)

    print(df)
    df.to_csv(FIGS_DIR.joinpath('minimal_n.csv'))
    return


def calc_power():
    alpha = .05 / 12686191

    # df = pd.read_csv(MWAS_DIR.joinpath('snp_annotations', 'annotated_clumped_0.3.csv'))
    df = pd.read_csv(MWAS_DIR.joinpath('clumped_mb_gwas_0.3.csv'))

    df['standardized_effect'] = df.apply(lambda x:
                                         np.abs(x['Coef']) / ((np.abs(x['Coef_025'] - x['Coef_975'])/3.924) * np.sqrt(x['N'])),
                                         axis=1)

    for n in [100, 500, 1000, 5000, 10000]:
        df[f'power_N_{n}_bon_all'] = df.apply(lambda x: tt_solve_power(effect_size=x['standardized_effect'],
                                                                       nobs=n, alpha=alpha, power=None, alternative='two-sided'),
                                              axis=1)

        df[f'power_N_{n}_bon_40'] = df.apply(lambda x: tt_solve_power(effect_size=x['standardized_effect'],
                                                                      nobs=n, alpha=.05/40, power=None, alternative='two-sided'),
                                             axis=1)

    print(df)
    df.to_csv(FIGS_DIR.joinpath('power_by_n.csv'))
    return


def plot_power_by_n():
    ## load the table
    df = pd.read_csv(FIGS_DIR.joinpath('power_by_n.csv'))

    ## dict: column names to labels. translate column names.
    cols = {'power_N_100_bon_all': 'N=100',
            'power_N_500_bon_all': 'N=500',
            'power_N_1000_bon_all': 'N=1000',
            'power_N_5000_bon_all': 'N=5000',
            'power_N_10000_bon_all': 'N=10000',
            }
    df = df[list(cols.keys())]
    df = df.rename(columns=cols)

    ## plot:
    ## x axis: N. y axis: power. box plot values.
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.4))
    df.boxplot(ax=ax)
    ax.set_ylabel('Power')
    ax.set_yscale('log')

    ## save
    plt.savefig(FIGS_DIR.joinpath(f"power_by_n.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return


def compare_n_and_replication():
    ## load the merged replication results
    replication = pd.read_csv(BASE_DIR.joinpath('20220402_234043_mwas_bmi_10K',
                                                '20221027_110736_mwas_bmi_lifelines_rep_inclusive', 'merged_results.csv'))

    ## load the calculated minimal N
    min_n = pd.read_csv(FIGS_DIR.joinpath('minimal_n.csv'))
    min_n['Contig'] = min_n['Contig'].apply(lambda x: '_'.join([x.split('_')[0], x.split('_')[1]]))

    ## add minimal_n column
    merge_inds = ['Species', 'Contig', 'Position']
    min_n = min_n.reset_index().set_index(merge_inds)
    replication = replication.reset_index().set_index(merge_inds)
    replication = replication.merge(min_n['min_N_0.9_bon_40'].to_frame('minimal_n'), how='left', left_index=True, right_index=True)

    ## add is_sufficient column
    replication['is_sufficient'] = replication.apply(lambda x: x['N_lifelines'] >= x['minimal_n'], axis=1)


    replication.to_csv(FIGS_DIR.joinpath('replication_results.csv'))
    replication.to_csv(BASE_DIR.joinpath('20220402_234043_mwas_bmi_10K', '20221027_110736_mwas_bmi_lifelines_rep_inclusive',
                                         'replication_results.csv'))
    return

if __name__ == '__main__':
    # calc_n()
    # calc_power()
    plot_power_by_n()
    # compare_n_and_replication()



