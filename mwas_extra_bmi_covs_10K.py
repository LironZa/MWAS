from LabData import config_global as config
import os
from pathlib import Path
import pandas as pd
import numpy as np
from pandas import Timedelta
import re
import matplotlib.pyplot as plt
import seaborn as sns

from LabData.DataLoaders.Loader import LoaderData
from LabData.DataAnalyses.MBSNPs.MWAS import CommonParams
from LabData.DataMergers.DataMerger import DataMerger
from LabUtils.addloglevels import sethandlers
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from LabData.DataLoaders.DietLoggingLoader import DietLoggingLoader
from LabData.DataAnalyses.TenK_Trajectories.utils import get_diet_logging_around_stage, get_baseline_medications
from LabData.DataLoaders.Medications10KLoader import Medications10KLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataLoaders.LifeStyleLoader import LifeStyleLoader
from UseCases.DataAnalyses.MBSNP_MWAS.MWASPaper.mwas_robust_samples_list_10K import loader_gen_f, species_covariate_loader
from UseCases.DataAnalyses.MBSNP_MWAS.MWASPaper.mwas_replication_w_Lifelines import SNPS_DF_P, get_snps_set, run_mwas
from UseCases.DataAnalyses.MBSNP_MWAS.MWASPaper.mwas_robust_samples_list_10K import robust_samples_loader, robust_features_loader


BASE_DIR = Path(config.analyses_dir)
TABLES_DIR = BASE_DIR.joinpath('MWAS_robust_samples_list')
FIGS_DIR = BASE_DIR.joinpath('Figures', 'BMI_covariates')
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DF_P = TABLES_DIR.joinpath('BasicMWAS_10K_singleSamplePerReg_20223003_234555.csv')
COVS_DF_P = TABLES_DIR.joinpath('BMI_extra_covariates_Jan23.csv')

y_get_data_args = {
    'groupby_reg': None,    # return all reported phenotypes, so we can match with the mb sample by Date
    'norm_dist_capping': {'sample_size_frac': 0.98, 'clip_sigmas': 5, 'remove_sigmas': 9},
}

exercise_cols = ['walking_minutes_day', 'moderate_activity_minutes', 'vigorous_activity_minutes',
                 'physical_activity_maderate_days_a_week', 'walking_10min_days_a_week']

drugs_cols = ['Agents acting on the renin-angiotensin system',
              'Antianemic preparations',
              'Antihistamines for systemic use',
              'Antithrombotic agents',
              'Calcium channel blockers',
              'Drugs for acid related disorders',
              'Drugs for obstructive airway diseases',
              'Drugs used in diabetes',
              'Lipid modifying agents',
              'Mineral supplements',
              'Psychoanaleptics',
              'Sex hormones and modulators of the genital system',
              'Thyroid therapy drugs',
              'Urologicals',
              'Vitamins',
              'beta-adrenergic blocking agents']

diet_cols = ['AlcoholicDrinks',
             'Beefveallambandothermeatproducts',
             'Bread',
             'Bread_wholewheat',
             'Cannedvegandfruits',
             'Cereals',
             'Deepfriedfoods',
             'Drinks',
             'Eggsandtheirproducts',
             'FastFoods',
             'Fishandseafood',
             'Fruits',
             'Hardcheese',
             'Industrializedvegetarianfoodreadytoeat',
             'Lowcaloriesanddietdrinks',
             'MedOilandfats',
             'Nutsseedsandproducts',
             'Oilsandfats',
             'Others',
             'PastaGrainsandSidedishes',
             'PastaGrainsandSidedishes_wholewheat',
             'Poultryanditsproducts',
             'Proccessedmeatproducts',
             'Pulsesandproducts',
             'Snacks',
             'Soupsandsauces',
             'Spicesandherbs',
             'Vegetables',
             'bakedgoods',
             'fruitjuicesandsoftdrinks',
             'milkcreamcheeseandyogurts',
             'sweetmilkproducts',
             'sweets']

cal_cols = ['energy_kcal']


def fix_date_dtype(loader): # Noam
    index_names = loader.df.index.names
    assert 'Date' in index_names
    loader.df.reset_index(inplace=True)
    loader.df_metadata.reset_index(inplace=True)
    loader.df['Date'] = loader.df['Date'].astype('datetime64[ns]')
    loader.df_metadata['Date'] = loader.df_metadata['Date'].astype('datetime64[ns]')
    loader.df.set_index(index_names, inplace=True)
    loader.df_metadata.set_index(index_names, inplace=True)
    return loader


def add_date_from_body_measures(loader, body_measures): # Noam
    loader.df = loader.df.join(body_measures.df.reset_index('Date')[['Date']], how='left').reset_index()
    loader.df.set_index(['RegistrationCode', 'Date'], inplace=True)
    loader.df_metadata.index = loader.df.index
    return loader


def create_covariates_df():
    '''
    Use the original robust samples DF to filter the samples, add additional covariates, and save.
    '''
    #TODO: make sure to take the FIRST/ BASELINE measurement

    ### get the list of participants/ sample used for the MWAS
    mwas_robust_df = pd.read_csv(DATA_DF_P, index_col=0)
    # samples_l = mwas_robust_df.index
    regs_l = list(mwas_robust_df['RegistrationCode'].values)


    ### arrange it all back into DataLoaders form
    subjects_dl = SubjectLoader()
    subjects_data = subjects_dl.get_data(reg_ids=regs_l)
    subjects_df = subjects_data.df

    mwas_robust_df['Date'] = mwas_robust_df['Date'].astype('datetime64[ns]')
    mb_samples_loader = loader_gen_f(mwas_robust_df, metadata_df=mwas_robust_df)
    mb_samples_loader.df = mb_samples_loader.df.dropna(subset=['age', 'gender', 'bmi']) # remove samples w/ missing data
    mb_samples_loader.df_metadata = mb_samples_loader.df_metadata.loc[mb_samples_loader.df.index] # remove samples w/ missing data

    ## merging the mb loader with other loaders will lose the 'SampleName' from index.
    #TODO: still needed?
    mb_samples_loader.df = mb_samples_loader.df.reset_index()
    mb_samples_loader.df_metadata = mb_samples_loader.df_metadata.reset_index()



    ##############    Drugs    ################
    #TODO: filter drug categories?
    # pivot by 2 or 4?

    tenk_data = get_baseline_medications()
    tenk_data = Medications10KLoader().get_data(df=tenk_data.df, subjects_df=subjects_df, pivot_by=2)
    # in case there are multiple reportings, keep a positive one if exists
    tenk_data.df = tenk_data.df.groupby('RegistrationCode').max()
    medication_names = tenk_data.df.columns
    # medication matrix should be complete, adding participants from body measures loader
    body_measures = BodyMeasuresLoader().get_data(subjects_df=subjects_df, groupby_reg='first', research_stage=['baseline'])
    # merging loaders to get metadata
    tenk_data = DataMerger([body_measures, tenk_data]).get_x(res_index_names=['RegistrationCode'])
    # keeping only the medication columns
    tenk_data.df = tenk_data.df.loc[:, medication_names]
    tenk_data.df.fillna(0, inplace=True)
    tenk_data.df = tenk_data.df.loc[:, tenk_data.df.sum() >= 50].astype(float) #TODO: set a different threshold?
    tenk_data.df = tenk_data.df.rename(columns=lambda x: x[0]+x[1:].lower())
    # tenk_data.df = tenk_data.df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    united_dl = add_date_from_body_measures(tenk_data, body_measures)


    ##############    Diet    ################


    ### Percent of daily calories (average), by food groups
    diet_dl = DietLoggingLoader()
    diet_data = diet_dl.get_data(study_ids=['10K'], reg_ids=regs_l)
    diet_data.df = get_diet_logging_around_stage(diet_data.df, stage='baseline', delta_before=2, delta_after=14)
    diet_data = diet_dl.daily_mean_food_consumption_energy(df=diet_data.df, kcal_limit=500, min_col_present_frac=0.05, level='MainCategoryEng')
    diet_data.df = diet_data.df.divide(diet_data.df.sum(axis=1), axis=0)
    diet_data = add_date_from_body_measures(diet_data, body_measures)
    diet_data = fix_date_dtype(diet_data)
    united_dl = DataMerger([united_dl, diet_data]).get_x(inexact_index='Date',
                                                         res_index_names=['RegistrationCode', 'Date'],
                                                         inexact_index_tolerance=Timedelta(days=180))

    ### Total caloric intake
    dl = DietLoggingLoader()
    diet_data = dl.get_data(study_ids=['10K'], reg_ids=regs_l)
    diet_data.df = get_diet_logging_around_stage(diet_data.df, stage='baseline', delta_before=2, delta_after=14)
    log = dl.add_nutrients(diet_data.df, nutrient_list=['energy_kcal'])  ### Remove the list to get all nutrients
    log['just_date'] = log.index.get_level_values('Date').date
    # summing total nutrient consumption per day
    daily_total_nutrient_consumption = log.groupby(['RegistrationCode', 'just_date']).sum()
    daily_median_nutrient_consumption = daily_total_nutrient_consumption.groupby('RegistrationCode').median()

    tenk_cals = LoaderData(daily_median_nutrient_consumption,
                           diet_data.df_metadata.loc[daily_median_nutrient_consumption.index], None)
    tenk_cals = add_date_from_body_measures(tenk_cals, body_measures)
    tenk_cals = fix_date_dtype(tenk_cals)
    united_dl = DataMerger([united_dl, tenk_cals]).get_x(inexact_index='Date',
                                                             res_index_names=['RegistrationCode', 'Date'],
                                                             inexact_index_tolerance=Timedelta(days=180))
    united_dl.df = united_dl.df.drop(columns=['weight'])


    ##############    Lifestyle    ################

    ### Physical activity
    # exercise_cols = ['walking_minutes_day', 'moderate_activity_minutes', 'vigorous_activity_minutes',
    #                  'physical_activity_maderate_days_a_week', 'walking_10min_days_a_week']
    tenk_sport = LifeStyleLoader().get_data(study_ids=['10K'], min_col_present=5000, df='english', groupby_reg='first',
                                            cols=exercise_cols, stage='baseline')
    tenk_sport = fix_date_dtype(tenk_sport)
    united_dl = DataMerger([united_dl, tenk_sport]).get_x(inexact_index='Date',
                                                          res_index_names=['RegistrationCode', 'Date'],
                                                          inexact_index_tolerance=Timedelta(days=180))

    ##############    Save table    ################
    print(united_dl.df)
    united_dl.df.to_csv(COVS_DF_P)
    return


def extra_covs_features_loader(table_path, filter_list, return_list, subjects_df=None):
    '''
    this will be the loader for the MWAS y/cov loaders. remember: index needs to be mb sample
    '''
    covs_df = pd.read_csv(table_path)
    covs_df = covs_df.set_index('SampleName')
    covs_df = covs_df.drop(columns=['RegistrationCode', 'Date', 'yob', 'StudyTypeID', 'SNPSNumPos', 'country', 'bmi'])
    ## filter nans according to the filter_list
    covs_df = covs_df.dropna(subset=filter_list)
    ## return values according to the return_list
    covs_df = covs_df[return_list]
    return LoaderData(pd.DataFrame(covs_df), None)


class extra_covariates_BMI_MWAS(CommonParams):
    ##### params class
    ## snps set -- 40 post clumping
    ## samples set -- same participants we used in the original MWAS
    ## covariates -- everything in the table
    ## return columns -- pvalue for each covariate
    ## covariates loader -- capping or other pre-processing for the covariates?

    samples_set = robust_samples_loader(DATA_DF_P)
    snp_set = get_snps_set(SNPS_DF_P) ## the significant SNPs in the discovery MWAS, with adjusted contig parts
    jobname = 'mwsBMI'

    min_reads_per_snp = 1
    max_on_fraq_major_per_snp = 1
    max_on_most_freq_val_in_col = 1
    min_subjects_per_snp = 1  # ORIGINALLY WAS 1000
    min_on_minor_per_snp = 1  # ORIGINALLY WAS 50
    min_positions_per_sample = 0
    filter_by_species_existence = False
    max_pval_to_detailed = 1
    largest_sample_per_user = False
    subsample_dir = ''
    max_jobs = 400
    species_blocks = 1
    # send_to_queue = False
    send_to_queue = True

    subjects_loaders = ['SubjectLoader']
    subjects_get_data_args = {'groupby_reg': 'first', 'study_ids': ['10K'], 'countries': ['IL']}

    species_specific_cov_f = species_covariate_loader

    ret_cov_fields = True



class drug_covs_BMI_MWAS(extra_covariates_BMI_MWAS):
    work_dir_suffix = 'mwas_bmi_w_drugs_cov'

    y_gen_f = lambda subjects_df, species_set: robust_features_loader(['bmi'], COVS_DF_P, subjects_df)
    covariate_gen_f = lambda subjects_df: extra_covs_features_loader(COVS_DF_P,
                                                                     drugs_cols+['age', 'gender'], drugs_cols+['age', 'gender'],
                                                                     subjects_df)

    output_cols = ['N', 'Coef', 'Pval', 'Coef_025', 'Coef_975', 'log10_RA_Pval', 'log10_RA_Coef']
    output_cols += [f"{cov.replace(' ', '_').replace('-', '_')}_Pval" for cov in drugs_cols]

class wo_drug_covs_BMI_MWAS(extra_covariates_BMI_MWAS):
    work_dir_suffix = 'mwas_bmi_wo_drugs_cov'

    y_gen_f = lambda subjects_df, species_set: robust_features_loader(['bmi'], COVS_DF_P, subjects_df)
    covariate_gen_f = lambda subjects_df: extra_covs_features_loader(COVS_DF_P,
                                                                     drugs_cols+['age', 'gender'], ['age', 'gender'],
                                                                     subjects_df)

    output_cols = ['N', 'Coef', 'Pval', 'Coef_025', 'Coef_975', 'log10_RA_Pval', 'log10_RA_Coef']


class diet_covs_BMI_MWAS(extra_covariates_BMI_MWAS):
    work_dir_suffix = 'mwas_bmi_w_diet_cov'

    y_gen_f = lambda subjects_df, species_set: robust_features_loader(['bmi'], COVS_DF_P, subjects_df)
    covariate_gen_f = lambda subjects_df: extra_covs_features_loader(COVS_DF_P,
                                                                     diet_cols+['age', 'gender'], diet_cols+['age', 'gender'],
                                                                     subjects_df)

    output_cols = ['N', 'Coef', 'Pval', 'Coef_025', 'Coef_975', 'log10_RA_Pval', 'log10_RA_Coef']
    output_cols += [f"{cov.replace(' ', '_').replace('-', '_')}_Pval" for cov in diet_cols]
    output_cols += [f"{cov.replace(' ', '_').replace('-', '_')}_Coef" for cov in diet_cols]

class wo_diet_covs_BMI_MWAS(extra_covariates_BMI_MWAS):
    work_dir_suffix = 'mwas_bmi_wo_diet_cov'

    y_gen_f = lambda subjects_df, species_set: robust_features_loader(['bmi'], COVS_DF_P, subjects_df)
    covariate_gen_f = lambda subjects_df: extra_covs_features_loader(COVS_DF_P,
                                                                     diet_cols+['age', 'gender'], ['age', 'gender'],
                                                                     subjects_df)

    output_cols = ['N', 'Coef', 'Pval', 'Coef_025', 'Coef_975', 'log10_RA_Pval', 'log10_RA_Coef']

class exercise_covs_BMI_MWAS(extra_covariates_BMI_MWAS):
    work_dir_suffix = 'mwas_bmi_w_exercise_cov'

    y_gen_f = lambda subjects_df, species_set: robust_features_loader(['bmi'], COVS_DF_P, subjects_df)
    covariate_gen_f = lambda subjects_df: extra_covs_features_loader(COVS_DF_P,
                                                                     exercise_cols+['age', 'gender'], exercise_cols+['age', 'gender'],
                                                                     subjects_df)

    output_cols = ['N', 'Coef', 'Pval', 'Coef_025', 'Coef_975', 'log10_RA_Pval', 'log10_RA_Coef']
    output_cols += [f"{cov.replace(' ', '_').replace('-', '_')}_Pval" for cov in exercise_cols]

class wo_exercise_covs_BMI_MWAS(extra_covariates_BMI_MWAS):
    work_dir_suffix = 'mwas_bmi_wo_exercise_cov'

    y_gen_f = lambda subjects_df, species_set: robust_features_loader(['bmi'], COVS_DF_P, subjects_df)
    covariate_gen_f = lambda subjects_df: extra_covs_features_loader(COVS_DF_P,
                                                                     exercise_cols+['age', 'gender'], ['age', 'gender'],
                                                                     subjects_df)

    output_cols = ['N', 'Coef', 'Pval', 'Coef_025', 'Coef_975', 'log10_RA_Pval', 'log10_RA_Coef']


def merge_mwases():
    bonferroni = 0.05 / (40 * 4 * 2)
    cov_groups = ['drugs', 'diet', 'exercise']
    covs_mwas_dir = FIGS_DIR.joinpath("MWAS")

    df = pd.read_csv(covs_mwas_dir.joinpath(f"mb_gwas_w_{cov_groups[0]}.csv"), index_col=[0,1,2,3])
    summary_df = pd.DataFrame(index=df.index, columns=[f'changed_w_{g}_covs' for g in cov_groups])

    for g in cov_groups:
        w_mwas = pd.read_csv(covs_mwas_dir.joinpath(f"mb_gwas_w_{g}.csv"), index_col=[0,1,2,3])
        wo_mwas = pd.read_csv(covs_mwas_dir.joinpath(f"mb_gwas_wo_{g}.csv"), index_col=[0,1,2,3])
        merged = w_mwas.merge(wo_mwas[['Pval', 'N']], how='left', left_index=True, right_index=True, suffixes=('', '_wo_extra_covs'))
        merged = merged.drop(columns=['Y_Bonferroni', 'Global_Bonferroni']).sort_index()
        assert merged['N'].equals(merged['N_wo_extra_covs'])

        merged['sig_wo'] = merged['Pval_wo_extra_covs'] < bonferroni
        merged['sig_w'] = merged['Pval'] < bonferroni
        merged[f'changed_w_{g}_covs'] = (merged['sig_wo'] & ~merged['sig_w'])
        summary_df[f'changed_w_{g}_covs'] = (merged['sig_wo'] & ~merged['sig_w'])

        merged.to_csv(covs_mwas_dir.joinpath(f"w_and_wo_{g}_covs.csv"))

    summary_df.to_csv(covs_mwas_dir.joinpath(f"covs_mwas_summary.csv"))


if __name__ == '__main__':
    # create_covariates_df()
    # covariates_correlations_mat()


    sethandlers(file_dir=config.log_dir)
    os.chdir('/net/mraid08/export/genie/LabData/Analyses/lironza/jobs')
    run_mwas(diet_covs_BMI_MWAS)
    # run_mwas(wo_exercise_covs_BMI_MWAS)