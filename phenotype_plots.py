import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from UseCases.DataAnalyses.MBSNP_MWAS.MWASPaper.mwas_robust_samples_list_PNP1_10K import TABLES_DIR
from pathlib import Path
from LabData import config_global as config
from LabData.DataAnalyses.MBSNPs.Plots.MWASPaper.mwas_paper_fig_utils import MWASPaperColors

# ROBUST_DATA_TABLE_NAME = 'lifelines_data_20222010_140759.csv'
ROBUST_DATA_TABLE_NAME = 'BasicMWAS_10K_singleSamplePerReg_20223003_234555.csv'
ROBUST_DATA_TABLE_P = TABLES_DIR.joinpath(ROBUST_DATA_TABLE_NAME)
BASE_DIR = Path(config.analyses_dir)
FIGS_DIR = BASE_DIR.joinpath('Figures', 'cohort')
FIGS_DIR.mkdir(parents=True, exist_ok=True)

c = MWASPaperColors()

def plot_cohort_phenotypes():
    # load the saved robust df, which has all the relevant phenotypes, for each participant
    df = pd.read_csv(ROBUST_DATA_TABLE_P, index_col=0)
    df = df[['gender', 'age', 'bmi']]
    df = df.dropna()
    # df.columns = df.columns.map({'gender': 'Sex', 'age': 'Age, years', 'bmi': 'BMI, points'})
    df.columns = df.columns.map({'gender': 'Sex', 'age':'age', 'bmi':'bmi'})
    female_label = f"Female (n={(df['Sex']==0).sum()})"
    male_label = f"Male (n={(df['Sex']==1).sum()})"
    df['Sex'] = df['Sex'].map({0: female_label, 1: male_label})
    print(df)
    fe = df.loc[df['Sex'] == female_label]
    ma = df.loc[df['Sex'] == male_label]
    assert len(fe)+len(ma)==len(df)
    fe_c = c.main_color_1
    ma_c = c.main_color_2
    fsize = 7

    # plot
    fig, ax = plt.subplots(1, 3, figsize=(8.25, 2.5), gridspec_kw={'wspace': .25})

    # ax0 -- age dist
    bins = np.arange(df['age'].min(), df['age'].max() + 5, 5)
    density = False
    ax[0].hist(fe['age'], color=fe_c, label=female_label, density=density, alpha=.5, bins=bins)
    ax[0].hist(ma['age'], color=ma_c, label=male_label, density=density, alpha=.5, bins=bins)
    ax[0].set_xlabel('Age, years', fontsize=fsize)
    # ax[0].tick_params(left=False, labelleft=False)
    ax[0].tick_params(axis='both', labelsize=fsize)
    # ax[0].legend(loc='upper left', fontsize=fsize)

    # ax1 -- BMI dist
    bins = np.arange(df['bmi'].min(), df['bmi'].max() + 2, 2)
    ax[1].hist(fe['bmi'], color=fe_c, label=female_label, density=density, alpha=.5, bins=bins)
    ax[1].hist(ma['bmi'], color=ma_c, label=male_label, density=density, alpha=.5, bins=bins)
    ax[1].set_xlabel('BMI, points', fontsize=fsize)
    # ax[1].tick_params(left=False, labelleft=False)
    ax[1].tick_params(axis='both', labelsize=fsize)
    ax[1].legend(loc='upper right', fontsize=6)

    # ax2 -- age-BMI correlation
    def plot_corr(ax, sub_df, color):
        # data points
        ax.plot(sub_df['age'], sub_df['bmi'], '.', alpha=0.3, color=color, zorder=1, ms=1.5)
        # trend
        b, a = np.polyfit(sub_df['age'].values, sub_df['bmi'].values, deg=1)
        xseq = np.linspace(sub_df['age'].min(), sub_df['age'].max(), num=100)
        ax.plot(xseq, a + b * xseq, color=color, lw=1, zorder=2)
        return

    plot_corr(ax[2], fe, fe_c)
    plot_corr(ax[2], ma, ma_c)
    ax[2].set_xlabel('Age, years', fontsize=fsize)
    ax[2].set_ylabel('BMI, points', fontsize=fsize)
    ax[2].tick_params(axis='both', labelsize=fsize)

    # save
    plt.savefig(FIGS_DIR.joinpath(ROBUST_DATA_TABLE_NAME.split('.')[0] + f"_phenotypes.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return

