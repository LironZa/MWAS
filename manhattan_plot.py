"""Draws Manhattan plots for SNP GWAS data."""
from typing import List
import matplotlib as mpl
import numpy as np
import pandas as pd
from LabData.DataAnalyses.MBSNPs import InterpreterUtils
from LabData.DataAnalyses.MBSNPs.taxonomy import draw_taxonomy_stripes, sort_by_taxonomy
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Iterable, Callable, Tuple
from matplotlib.lines import Line2D
from matplotlib import offsetbox, rcParams

_DRAW_SPECIES_XTICKS = False
_COLOR_MAP = plt.get_cmap('winter')
_NUM_COLORS = 3

COG_CAT_2_LABEL = InterpreterUtils.COG_categories_labels
COG_CAT_2_VAL = {mt: mi for mi, mt in enumerate(COG_CAT_2_LABEL)}


def _get_gene_groups_dicts(annot_col):
    if annot_col == 'COG cat':
        lab_dict = COG_CAT_2_LABEL
        func_dict = COG_CAT_2_VAL
        color_dict = {gs: InterpreterUtils.COG_labels_colors[gl] for gs, gl in COG_CAT_2_LABEL.items()}
    return lab_dict, func_dict, color_dict


def add_snps_annots(ax, xx_dict, annots_fs, annot_df, phen, num_annots=12, bon_thresh=.05, opt_by='clump'):
    from LabData.DataAnalyses.MBSNPs.Plots.MWASPaper.mwas_paper_fig_utils import choose_top_snps, MWASPaperColors
    from adjustText import adjust_text

    c = MWASPaperColors()

    # choose SNPs to annotate
    if 'GeneRelation' in annot_df.columns:
        rel_annot_df = annot_df.loc[((annot_df['GeneRelation'] == 'Current') & annot_df['gene'].notnull())].reset_index()
    else:
        rel_annot_df = annot_df.loc[((annot_df['GeneDistance'] > -1) & annot_df['gene'].notnull())].reset_index()

    if 'Global_Bonferroni' in annot_df.columns:
        rel_annot_df = rel_annot_df.loc[rel_annot_df['Global_Bonferroni'] < bon_thresh]

    try:
        top_snps_annots = choose_top_snps(rel_annot_df, opt_by=opt_by, n=num_annots)[[
            'Species', 'Contig', 'Position', 'Pval', 'Coef', 'gene']]
    except:
        top_snps_annots = choose_top_snps(rel_annot_df, opt_by='Pval', n=num_annots)[[
            'Species', 'Contig', 'Position', 'Pval', 'Coef', 'gene', 'post_clumping']]

    print(top_snps_annots)
    for i, r in top_snps_annots.iterrows():
        top_snps_annots.loc[i, 'xx'] = float(xx_dict[r['Species']].xs(r['Contig'], level='Contig').xs(r['Position'], level='Position'))
    tags = [ax.text(snp_row['xx'], snp_row['Pval'], snp_row['gene'].split("_")[0], fontsize=annots_fs, color=c.red, weight="bold")
            for _, snp_row in top_snps_annots.iterrows()]

    # now get ALL top SNPs, so the lables will avoid overlapping with them
    all_sig_snps = annot_df.sort_values('Pval').head(int(len(annot_df)/10)).reset_index()[['Species', 'Contig', 'Position', 'Pval']]
    if len(all_sig_snps) == 0:
        return
    for i, r in all_sig_snps.iterrows():
        all_sig_snps.loc[i, 'xx'] = float(xx_dict[r['Species']].xs(r['Contig'], level='Contig').xs(r['Position'], level='Position'))
    adjust_text(tags, x=all_sig_snps['xx'].values, y=all_sig_snps['Pval'].values,
                ax=ax, arrowprops={'arrowstyle': '-', 'color': c.red})
    return


def _match_gene_to_group(gene_row, annot_col, color_dict):
    try:
        val = color_dict[gene_row[annot_col]]
    except KeyError:
        val = color_dict['else']
    return val


def _bonferroni_cutoff(df: pd.DataFrame, val=0.05) -> float:
    """Returns the p-value cutoff for bonferroni 0.05."""
    filtered = df[df['Global_Bonferroni'] <= val]['Pval']
    return None if filtered.empty else max(filtered)


def _ticks_to_xx(ticks: List[float]) -> List[float]:
    """Converts coordinates of middles of intervals to coordinates of the ends
    of those intervals. Assumes intervals have no gaps between them."""
    result = [ticks[0] * 2]
    for i in range(1, len(ticks)):
        result.append(result[i - 1] + ticks[i] - ticks[i - 1])
    return result


def _map_snp_types(df):
    ### since some SNPs are unannotated (contig too short for genes?) they are treated the same as the intergenic
    marker = df['NonSymMutation'].map({False: 'syn', True: 'nonsyn'})
    marker.loc[df['GeneDistance'] < 0] = 'inter'
    marker.loc[df['GeneDistance'].isnull()] = 'inter'  # these are the unannotated SNPs
    marker.loc[(df['GeneDistance'] >= 0) & (df['feature'] != 'CDS')] = 'RNA'
    return marker


def draw_annotated_manhattan_plot(df: pd.DataFrame, out_file: str = None, figsize=(8.2, 4.3), dpi=300,
                                  pval_col='Global_FDR', pval_cutoff=.001,
                                  color_by_func=False, annot_col='COG cat', dicts=None,
                                  inter_black=False, canc_alpha=False, ppt=False, summ_txt=False,
                                  tax_txt=True, tax_all_ticks=False, phenotype=None,
                                  annot_df=None, label_genes=False,
                                  marker_shape_by_type=True, draw_type_legend=True,
                                  legend_colors=True) -> None:
    if summ_txt:
        summary_text = str(len(df)) + ' SNPs in ' + str(len(set(df.index.get_level_values('Species')))) + \
            ' genomes\n' + 'N = ' + str(int(df['N'].min())) + ' to ' + str(int(df['N'].max())) + \
            ' samples per SNP\n' + str((df['Global_Bonferroni'] <= 0.05).sum()) + ' SNPs passed 0.05 Bonferroni cutoff'

    bonn_5_line = 0.05 / (df['Global_Bonferroni'] / df['Pval']).max() if 'Global_Bonferroni' in df.columns else 0.05 / len(df)
    df = df.loc[df['Pval'] <= pval_cutoff]
    # df.loc[:, pval_col] = df[pval_col].replace(to_replace=0, value=1e-300)
    if len(df) < 2:
        return
    df[pval_col] = df[pval_col].replace(to_replace=0, value=1e-300)

    xx_dict = {}
    if dicts is None and color_by_func:
        lab_dict, func_dict, color_dict = _get_gene_groups_dicts(annot_col)
    elif color_by_func:
        lab_dict, func_dict, color_dict = dicts

    # sort by taxonomy
    species = sort_by_taxonomy(set(df.index.unique('Species')))

    grouped = df.groupby('Species')
    num_points = 0
    if len(df) > 100000:
        gap = 20  # between species
    else:
        gap = 3
    black = (0.0, 0.0, 0.0, .7)

    if ppt:
        legend_fs = 7
        tags_fs = 7
        figsize = (10, 4.3)
        axis_labels_fs = 8
        if color_by_func:
            ms_factor = 1.1
        else:
            ms_factor = .5
        lg_ms = 5
        dpi = 500
    else:
        legend_fs = 6
        tags_fs = 6
        axis_labels_fs = 7
        lg_ms = 3.5
        ms_factor = .5
        dpi = 500

    markers = {'inter': '*', 'syn': 'o', 'nonsyn': 'o', 'RNA': 'v'}
    tp_labels = {'inter': 'Intergenic or unknown', 'syn': 'Protein coding: synonymous',
                 'nonsyn': 'Protein coding: non-synonymous', 'RNA': 'Non-protein coding (rRNA, tRNA etc.)'}

    fig = plt.figure(figsize=figsize, dpi=dpi)

    gs = GridSpec(6, 1, hspace=.1)
    ax_man: plt.Axes = fig.add_subplot(gs[:5, :])
    ax_tax: plt.Axes = fig.add_subplot(gs[5, :])

    if color_by_func:
        iterations_list = [0, 1]
    else:
        iterations_list = [0]

    for iter_ind in iterations_list:
        x_max = 0
        ticks = []
        actual_species = []
        x_rights = []
        for i, sp in enumerate(species):
            if sp not in grouped.groups:
                continue
            actual_species.append(sp)
            species_df = grouped.get_group(sp).sort_values('Position')  # because the df was sorted by p in MWASInterpreter
            pvals = species_df[pval_col]
            n = len(pvals)
            x_width = n ** (1/2)  # reduce width differences between different species
            to_draw = pvals.to_frame('pvals')
            to_draw['x'] = range(n)

            if marker_shape_by_type:
                tp_annots = _map_snp_types(grouped.get_group(sp))

            xx = (to_draw['x'] / n) * x_width + x_max
            yy = to_draw['pvals']
            markersizes = -np.log10(pvals)
            if label_genes:
                xx_dict[sp] = xx

            if not color_by_func:
                color = 'black'
                tp_facecolors = {'inter': color, 'syn': 'none', 'nonsyn': color, 'RNA': color}
                edgecolors = color
                if marker_shape_by_type:
                    for tp in markers.keys():
                        msizes = .5 * ms_factor * markersizes.loc[tp_annots == tp]
                        ax_man.scatter(xx.loc[tp_annots == tp], yy.loc[tp_annots == tp],
                                       marker=markers[tp], s=msizes, facecolor=tp_facecolors[tp],
                                       edgecolor=edgecolors, alpha=.45, linewidths=1)
                else:
                    msizes = .5 * ms_factor * markersizes
                    ax_man.scatter(xx, yy, marker='o', s=msizes, facecolor='black', edgecolor=None, alpha=.45, linewidths=1)

            if color_by_func:
                to_draw['color_g'] = species_df.apply(_match_gene_to_group,
                                                      axis=1,
                                                      args=(annot_col, color_dict))  # maps gene groups to colors
                greys = (to_draw['color_g'] == color_dict['else'])
                colored = (to_draw['color_g'] != color_dict['else'])
                color_group = [greys, colored][iter_ind]  # to draw in two layers
                for tp in markers.keys():
                    draw_inds = ((tp_annots == tp) & color_group)
                    if tp == 'syn':
                        edgecolors = to_draw['color_g'].loc[draw_inds]
                        facecolor = 'none'
                    elif tp == 'inter' and inter_black:
                        facecolor = black
                        edgecolors = black
                    elif (tp == 'nonsyn' or tp == 'RNA') or (tp == 'inter' and not inter_black):
                        edgecolors = to_draw['color_g'].loc[draw_inds]
                        facecolor = edgecolors
                    msizes = ms_factor * markersizes.loc[draw_inds]
                    if canc_alpha:
                        alpha = None
                    else:
                        alpha = .55
                    ax_man.scatter(xx.loc[draw_inds], yy.loc[draw_inds],
                                   marker=markers[tp], s=msizes, facecolor=facecolor,
                                   edgecolor=edgecolors, linewidth=msizes ** .2, alpha=alpha)

            ticks.append(x_max + x_width / 2)
            # add gap and move to the locations of the next species
            x_rights.append(x_max + x_width + gap/2)
            x_max += x_width + gap
            num_points += n

    # legend for snp types
    legend_handles = []
    for st, sm in markers.items():
        if st == 'syn':
            fillstyle = 'none'
        else:
            fillstyle = 'full'
        hand = mpl.lines.Line2D([], [], axes=ax_man, color='black', fillstyle=fillstyle,
                                marker=sm, linestyle='None', markersize=lg_ms, label=tp_labels[st])
        legend_handles.append(hand)
    if draw_type_legend and marker_shape_by_type:
        first_legend = ax_man.legend(handles=legend_handles, loc='lower right', fontsize=legend_fs,
                                     frameon=False, ncol=4, bbox_to_anchor=(1, 1), labelspacing=0.15, handletextpad=0.05)
        ax_man.add_artist(first_legend)

    # add legends for colors
    if color_by_func and legend_colors:
        legend_handles = []
        for gg, gc in color_dict.items():
            if gg in list(df[annot_col].unique()) + ['else']:
                hand = mpl.lines.Line2D([], [], axes=ax_man, color=gc, marker='o', alpha=.75, linestyle='None',
                                        markersize=lg_ms, label=lab_dict[gg])
                legend_handles.append(hand)
        ax_man.legend(handles=legend_handles, loc='upper right', title='gene group', fontsize=legend_fs,
                      title_fontsize=legend_fs, framealpha=0.3, labelspacing=0.8, frameon=True)

    ax_man.axhline(bonn_5_line, ls='--', color='red', lw=1)
    if dicts is None and ppt:
        ax_man.text(x=0, y=bonn_5_line, s='Bonferroni<0.05', color='red', size=legend_fs-2,
                    weight='bold', va='bottom', ha='right')  # or: x=x_max+gap and ha='left'

    if summ_txt:
        ax_man.text(x=100, y=ax_man.get_ylim()[1]*5, s=summary_text, ha='left', va='top',
                    size=legend_fs, color='midnightblue')

    ax_man.set_yscale('log')
    min_pval = min(df[pval_col].min(), bonn_5_line)
    min_pval_in_fig = min_pval * 10 ** (0.1 * np.log10(min_pval)) # Add 10% white space above
    ax_man.set_ylim([1, min_pval_in_fig])
    ticks_major = -2
    ax_man.set_yticks([10**i for i in np.arange(0, np.log10(min_pval_in_fig), ticks_major)])
    ax_man.set_ylabel('P-value', fontsize=axis_labels_fs)#, labelpad=2)
    ax_man.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_man.tick_params(labelsize=axis_labels_fs, axis='y')
    ax_man.set_xlim(-5, x_max + 5)
    draw_taxonomy_stripes(ax=ax_tax, xx=x_rights, species=actual_species, levels=[1, 2, 4, 5, 6],
                          add_text=tax_txt, labels_fs=axis_labels_fs, all_ticks=tax_all_ticks)
    ax_tax.tick_params(labelsize=axis_labels_fs, axis='y')
    ax_tax.tick_params(labelsize=axis_labels_fs, axis='x')
    ax_tax.set_xlim(-5, x_max + 5)
    if label_genes:
        add_snps_annots(ax_man, xx_dict, tags_fs, annot_df, phenotype, num_annots=40)

    plt.savefig(out_file, dpi=dpi, bbox_inches='tight', pad_inches=.2)
    plt.close()

