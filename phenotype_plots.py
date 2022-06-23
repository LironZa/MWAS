import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from UseCases.DataAnalyses.MBSNP_MWAS.MWASPaper.mwas_robust_samples_list_PNP1_10K import TABLES_DIR
from pathlib import Path
from LabData import config_global as config
from LabData.DataAnalyses.MBSNPs.Plots.MWASPaper.mwas_paper_fig_utils import MWASPaperColors

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
    df.columns = df.columns.map({'gender': 'Sex', 'age': 'Age, years', 'bmi': 'BMI, points'})
    female_label = f"Female (n={(df['Sex']==0).sum()})"
    male_label = f"Male (n={(df['Sex']==1).sum()})"
    df['Sex'] = df['Sex'].map({0: female_label, 1: male_label})
    print(df)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.4))

    g = sns.pairplot(df, hue='Sex', plot_kws={'s': 5}, palette={female_label: c.main_color_2, male_label: '#405952'}, corner=True)
    g.map_lower(sns.regplot, scatter_kws={'s': 1, 'alpha': .6})

    # save
    plt.savefig(FIGS_DIR.joinpath(f"phenotypes.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return


if __name__ == '__main__':
    plot_cohort_phenotypes()
