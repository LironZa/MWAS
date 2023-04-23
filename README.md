# MWAS
Associating bacterial SNPs in the human gut microbiome with host traits.

* README.md — this file.

* MBSNPAnalyses.py — the MWAS framework. Loads the phenotypes and the genetic data, and computes their associations. Calls statutils.py. 

* MWAS.py — generates an MWAS instance and runs it. Calls MBSNPAnalyses.py.

* mwas_w_robust_samples.py — runs the MWAS analysis for BMI, as described in the manuscript. Sets the parameters for the run, chooses the y and the covariates for the regression. Calls MWAS.py.

* mwas_extra_bmi_covs_10K.py — runs the MWAS analysis with the additional diet, exercise and medication covariates.

* statutils.py — helper function for MBSNPAnalyses.py. Computes the regression.

* MWASClumping.py — applies clumping to the MWAS results. Plots the histograms in Extended Data Fig. 2.

* species_composition_phenotype.py — associates species relative abundance (RA) with host phenotype. Visualizes the fraction of species that associate with BMI by RA and that do not out of the species in which we found BMI-associated SNPs, Fig. 4.

* custom_volcano_plot.py — creates the volcano plot in Extended Data Fig. 3.

* maf_plot.py — creates the boxplots that compare the phenotypes of participants with the major allele dominant, an alternative allele, or without the species (e.g. Fig. 2b).

* manhattan_plot.py — draws the Manhattan plots visualizing the MWAS results. Used for generating Fig. 2a.

* mwas_paper_fig_utils.py — helper functions for the various visualizations.

* phenotype_plots.py — visualize the phenotype distributions in the cohort, Fig. 1b and Extended Data Fig. 8.

* snps_overview.py — plots the distribution of detected SNPs, as in Extended Data Fig. 1. Plots the number of BMI-associated SNPs in each species, Fig. 3.

* clumped_qq.py — plots the q-q plots, Extended Data Fig. 5. 

* power_analysis — estimates the statistical power to discover similar BMI-SNP associations based on sample size. Plots the results, Extended Data Fig. 6. Estimates the required sample size to discover the associations in a different cohort. 

* mwas_replication_w_Lifelines.py — runs the replication MWAS analysis.

* replication_randomization — runs the randomization analysis to estimate the p-value of the replication results. Plots the results, Extended Data Fig. 7. Plots the comparison between the estimated coefficients of SNPs in the discovery and replication cohorts, Fig. 5.

* enrichment.py — tests the enrichment of energy metabolism function in SNPs.

* gene_map_plot.py — plots genes on a zoomed-in version of the Manhattan plot, Fig. 6.
