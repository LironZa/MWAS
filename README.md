# MWAS
Associating bacterial SNPs in the human gut microbiome with host traits.

* README.md — this file.

* MBSNPAnalyses.py — the MWAS framework. Loads the phenotypes and the genetic data, and computes their associations. Calls statutils.py. 

* MWAS.py — generates an MWAS instance and runs it. Calls MBSNPAnalyses.py.

* mwas_w_robust_samples.py — runs the MWAS analysis for BMI, as described in the manuscript. Sets the parameters for the run, chooses the y and the covariates for the regression. Calls MWAS.py.

* statutils.py — helper function for MBSNPAnalyses.py. Computes the regression.

* MWASClumping.py — applies clumping to the MWAS results. Plots the histograms in Extended Data Fig. 2.

* species_composition_phenotype.py — associates species relative abundance with host phenotype.

* custom_volcano_plot.py — creates the volcano plot in Extended Data Fig. 3.

* maf_plot.py — creates the boxplots that compare the phenotypes of participants with the major allele dominant, an alternative allele, or without the species (e.g. Fig. 3).

* manhattan_plot.py — draws the Manhattan plots visualizing the MWAS results. Used for generating Fig. 2.

* mwas_paper_fig_utils.py — helper functions for the various visualizations.

* phenotype_plots.py — visualize the phenotype distributions in the cohort, Fig. 1b.

* snps_overview.py — plots the distribution of detected SNPs, as in Extended Data Fig. 1.
