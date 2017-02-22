
import os
import logging
import configparser
from collections import Counter

import numpy
import pandas

from trueimpactdataset_analysis.data.data_loader import DataLoader
from trueimpactdataset_analysis.classification.citation_classifier import (
    CitationClassifier
)

__author__ = 'robodasha'
__email__ = 'damirah@live.com'


def load_config():
    """
    :return:
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config


def df_to_table(res_df, output_fpath):
    """
    :param res_df:
    :param output_fpath:
    :return:
    """
    res_df['tn (%)'] = res_df.tn / res_df.total
    res_df['tp (%)'] = res_df.tp / res_df.total
    res_df['fn (%)'] = res_df.fn / res_df.total
    res_df['fp (%)'] = res_df.fp / res_df.total
    res_df['tn_best (%)'] = res_df.tn_best / res_df.total
    res_df['tp_best (%)'] = res_df.tp_best / res_df.total
    res_df['fn_best (%)'] = res_df.fn_best / res_df.total
    res_df['fp_best (%)'] = res_df.fp_best / res_df.total

    decimals = pandas.Series(
        {'accuracy': 4, 'accuracy_best': 4, 'baseline': 4,
         'tn (%)': 2, 'tp (%)': 2, 'fn (%)': 2, 'fp (%)': 2, 'tn_best (%)': 2,
         'tp_best (%)': 2, 'fn_best (%)': 2, 'fp_best (%)': 2})
    columns = ['thresholds', 'threshold_best', 'accuracy', 'accuracy_best',
               'baseline', 'tn', 'tp', 'fn', 'fp', 'tn (%)', 'tp (%)', 'fn (%)',
               'fp (%)', 'tn_best', 'tp_best', 'fn_best', 'fp_best',
               'tn_best (%)', 'tp_best (%)', 'fn_best (%)', 'fp_best (%)',
               'total']

    tot_tot = res_df.total.sum()
    tn_sum = res_df.tn.sum()
    tn_sum_p = tn_sum / tot_tot
    tp_sum = res_df.tp.sum()
    tp_sum_p = tp_sum / tot_tot
    fn_sum = res_df.fn.sum()
    fn_sum_p = fn_sum / tot_tot
    fp_sum = res_df.fp.sum()
    fp_sum_p = fp_sum / tot_tot

    tn_best_sum = res_df.tn_best.sum()
    tn_best_sum_p = tn_best_sum / tot_tot
    tp_best_sum = res_df.tp_best.sum()
    tp_best_sum_p = tp_best_sum / tot_tot
    fn_best_sum = res_df.fn_best.sum()
    fn_best_sum_p = fn_best_sum / tot_tot
    fp_best_sum = res_df.fp_best.sum()
    fp_best_sum_p = fp_best_sum / tot_tot

    totals = ['-', '-',
              (tn_sum + tp_sum) / tot_tot,
              (tn_best_sum + tp_best_sum) / tot_tot,
              '-',
              tn_sum, tp_sum, fn_sum, fp_sum,
              tn_sum_p, tp_sum_p, fn_sum_p, fp_sum_p,
              tn_best_sum, tp_best_sum, fn_best_sum, fp_best_sum,
              tn_best_sum_p, tp_best_sum_p, fn_best_sum_p, fp_best_sum_p,
              tot_tot]

    all_s = pandas.Series(totals, index=columns)
    res_df[columns].round(decimals).to_html(output_fpath)
    return res_df, all_s


def print_latex(res_df, totals):
    """
    :param res_df:
    :param totals:
    :return:
    """
    for row in res_df.iterrows():
        print('\t\t{}'.format(row[0]), '& {:.4f}'.format(row[1].accuracy),
              '& {:.4f}'.format(row[1].accuracy_best),
              '& {:.4f}'.format(row[1].baseline),
              '&', row[1].threshold_best, '&',
              row[1].tn, '&', row[1].tp, '&', row[1].fn, '&', row[1].fp, '&',
              row[1].total, '\\\\')
        print('\t\t\hline')
    print('\t\t\hline')
    print('\t\t\\textbf{All}', '& {:.4f}'.format(totals.accuracy),
          '& {:.4f}'.format(totals.accuracy_best),
          '& - ',
          '&', totals.threshold_best, '&',
          totals.tn, '&', totals.tp, '&', totals.fn, '&', totals.fp, '&',
          totals.total, '\\\\')
    print('\t\t\hline')
    return


def classify_citations_basic():
    """
    :return:
    """
    config = load_config()
    logger = logging.getLogger(__name__)

    logger.info('Loading data')
    papers_df = DataLoader(config['trueid']['metadata']).load_papers()
    logger.info('Done loading data')

    accuracy_all, threshold_all, cm_all = CitationClassifier().train(
        papers_df.citations_gs.tolist(), papers_df.seminal.tolist(),
        optimal=True)

    accuracy_cv, thresholds, cm = CitationClassifier().test(
        papers_df.citations_gs.tolist(), papers_df.seminal.tolist())

    logger.info('Accuracy on all: {}, threshold: {}'.format(
        accuracy_all, threshold_all))
    logger.info('Confusion matrix:\n{}'.format(pandas.DataFrame(
        cm_all, columns=['Pred 0', 'Pred 1'], index=['True 0', 'True 1'])))
    logger.info('Cross-validation accuracy: {}, thresholds: {}'.format(
        accuracy_cv, Counter(thresholds).most_common()))
    logger.info('Confusion matrix:\n{}'.format(pandas.DataFrame(
        cm, columns=['Pred 0', 'Pred 1'], index=['True 0', 'True 1'])))
    return


def classify_readership_basic():
    """
    :return:
    """
    config = load_config()
    logger = logging.getLogger(__name__)

    logger.info('Loading data')
    metadata_dir = config['trueid']['metadata']
    dl = DataLoader(metadata_dir)
    papers_df = dl.load_papers()
    mendeley_df = dl.load_mendeley_metadata()
    logger.info('Done loading data')

    papers_df['reader_count'] = mendeley_df.reader_count
    papers_df.loc[pandas.isnull(papers_df.reader_count), 'reader_count'] = 0
    papers_df.reader_count = papers_df.reader_count.astype(int)

    accuracy_all, threshold_all, cm_all = CitationClassifier().train(
        papers_df.reader_count.tolist(), papers_df.seminal.tolist(),
        optimal=True)

    accuracy_cv, thresholds, cm = CitationClassifier().test(
        papers_df.reader_count.tolist(), papers_df.seminal.tolist())

    logger.info('Accuracy on all: {}, threshold: {}'.format(
        accuracy_all, threshold_all))
    logger.info('Confusion matrix:\n{}'.format(pandas.DataFrame(
        cm_all, columns=['Pred 0', 'Pred 1'], index=['True 0', 'True 1'])))
    logger.info('Cross-validation accuracy: {}, thresholds: {}'.format(
        accuracy_cv, Counter(thresholds).most_common()))
    logger.info('Confusion matrix:\n{}'.format(pandas.DataFrame(
        cm, columns=['Pred 0', 'Pred 1'], index=['True 0', 'True 1'])))
    return


def classify_citations_discipline():
    """
    :return:
    """
    config = load_config()
    logger = logging.getLogger(__name__)

    logger.info('Loading data')
    papers_df = DataLoader(config['trueid']['metadata']).load_papers()
    logger.info('Done loading data')

    logger.info('Converting res. category to numerical labels')
    papers_df.citations_gs.tolist()
    research_area_category = pandas.factorize(papers_df.research_area)
    papers_df['research_area_category'] = research_area_category[0]

    logger.info('Preparing classification data')
    citations = numpy.array(papers_df.citations_gs.tolist())
    labels = numpy.array(papers_df.seminal.tolist())
    groups = numpy.array(papers_df.research_area_category.tolist())
    logger.info('Got {} citation values, {} labels, {} group labels'.format(
        len(citations), len(labels), len(groups)))

    logger.info('Finding \'other\' category')
    other_cat = list(research_area_category[1]).index('Other')
    logger.info('Will skip {} values'.format(
        len(numpy.where(groups == other_cat)[0])))

    cc = CitationClassifier()
    results = []

    for category in numpy.unique(groups):
        if category == other_cat:
            continue
        good_idxs = numpy.where(groups == category)
        good_labels = labels[good_idxs]
        good_citations = citations[good_idxs]
        # need at least 3 papers representing both groups
        if len(good_idxs[0]) <= 3 \
                or len(numpy.where(good_labels == True)[0]) <= 1 \
                or len(numpy.where(good_labels == False)[0]) <= 1:
            continue
        accuracy_best, threshold_best, cm_best = cc.train(
            good_citations.tolist(), good_labels.tolist(),
            print_progress=False, optimal=True)
        accuracy, thresholds, cm = cc.test(
            good_citations.tolist(), good_labels.tolist())
        results.append({
            'res_area': research_area_category[1][category],
            'thresholds': Counter(thresholds).most_common(),
            'threshold_best': threshold_best,
            'accuracy': accuracy,
            'accuracy_best': accuracy_best,
            'baseline': cc.baseline(good_labels.tolist()),
            'tn': cm[0, 0], 'tp': cm[1, 1], 'fn': cm[1, 0], 'fp': cm[0, 1],
            'tn_best': cm_best[0, 0], 'tp_best': cm_best[1, 1],
            'fn_best': cm_best[1, 0], 'fp_best': cm_best[0, 1],
            'total': len(good_idxs[0])})

    res_df = pandas.DataFrame(results)
    res_df.set_index('res_area', drop=True, inplace=True)
    output_fpath = os.path.join(
        config['paths']['out'], 'results_citations_discipline.html')
    res_df, totals = df_to_table(res_df, output_fpath)
    print_latex(res_df, totals)
    return


def classify_readership_discipline():
    """
    :return:
    """
    config = load_config()
    logger = logging.getLogger(__name__)

    logger.info('Loading data')
    dl = DataLoader(config['trueid']['metadata'])
    papers_df = dl.load_papers()
    mendeley_df = dl.load_mendeley_metadata()
    logger.info('Done loading data')

    papers_df['reader_count'] = mendeley_df.reader_count
    papers_df.loc[pandas.isnull(papers_df.reader_count), 'reader_count'] = 0
    papers_df.reader_count = papers_df.reader_count.astype(int)

    logger.info('Converting res. category to numerical labels')
    research_area_category = pandas.factorize(papers_df.research_area)
    papers_df['research_area_category'] = research_area_category[0]

    logger.info('Preparing classification data')
    readership = numpy.array(papers_df.reader_count.tolist())
    labels = numpy.array(papers_df.seminal.tolist())
    groups = numpy.array(papers_df.research_area_category.tolist())
    logger.info('Got {} citation values, {} labels, {} group labels'.format(
        len(readership), len(labels), len(groups)))

    logger.info('Finding \'other\' category')
    other_cat = list(research_area_category[1]).index('Other')
    logger.info('Will skip {} values'.format(
        len(numpy.where(groups == other_cat)[0])))

    cc = CitationClassifier()
    results = []

    for category in numpy.unique(groups):
        if category == other_cat:
            continue
        good_idxs = numpy.where(groups == category)
        good_labels = labels[good_idxs]
        good_readership = readership[good_idxs]
        # need at least 3 papers representing both groups
        if len(good_idxs[0]) <= 3 \
                or len(numpy.where(good_labels == True)[0]) <= 1 \
                or len(numpy.where(good_labels == False)[0]) <= 1:
            continue
        accuracy_best, threshold_best, cm_best = cc.train(
            good_readership.tolist(), good_labels.tolist(),
            print_progress=False, optimal=True)
        accuracy, thresholds, cm = cc.test(
            good_readership.tolist(), good_labels.tolist())
        results.append({
            'res_area': research_area_category[1][category],
            'thresholds': Counter(thresholds).most_common(),
            'threshold_best': threshold_best,
            'accuracy': accuracy,
            'accuracy_best': accuracy_best,
            'baseline': cc.baseline(good_labels.tolist()),
            'tn': cm[0, 0] if cm.shape[0] >= 2 and cm.shape[1] >= 2 else None,
            'tp': cm[1, 1] if cm.shape[0] >= 2 and cm.shape[1] >= 2 else None,
            'fn': cm[1, 0] if cm.shape[0] >= 2 and cm.shape[1] >= 2 else None,
            'fp': cm[0, 1] if cm.shape[0] >= 2 and cm.shape[1] >= 2 else None,
            'tn_best': cm_best[0, 0], 'tp_best': cm_best[1, 1],
            'fn_best': cm_best[1, 0], 'fp_best': cm_best[0, 1],
            'total': len(good_idxs[0])})

    res_df = pandas.DataFrame(results)
    res_df.set_index('res_area', drop=True, inplace=True)
    output_fpath = os.path.join(
        config['paths']['out'], 'results_readership_discipline.html')
    res_df, totals = df_to_table(res_df, output_fpath)
    print_latex(res_df, totals)
    return


def classify_citations_year():
    """
    :return:
    """
    config = load_config()
    logger = logging.getLogger(__name__)

    logger.info('Loading data')
    papers_df = DataLoader(config['trueid']['metadata']).load_papers()
    logger.info('Done loading data')

    logger.info('Preparing classification data')
    citations = numpy.array(papers_df.citations_gs.tolist())
    labels = numpy.array(papers_df.seminal.tolist())
    years = numpy.array(papers_df.year.tolist())
    logger.info('Got {} citation values, {} labels, {} year labels'.format(
        len(citations), len(labels), len(years)))

    cc = CitationClassifier()
    results = []

    for year in sorted(numpy.unique(years)):
        print(year)
        print(years)
        good_idxs = numpy.where(years == year)
        print(good_idxs)
        good_labels = labels[good_idxs]
        good_citations = citations[good_idxs]
        # need at least 4 papers, two of each type
        if len(good_idxs[0]) <= 3 \
                or len(numpy.where(good_labels == True)[0]) <= 1 \
                or len(numpy.where(good_labels == False)[0]) <= 1:
            continue
        accuracy_best, threshold_best, cm_best = cc.train(
            good_citations.tolist(), good_labels.tolist(),
            print_progress=False, optimal=True)
        accuracy, thresholds, cm = cc.test(
            good_citations.tolist(), good_labels.tolist())
        results.append({
            'year': year,
            'thresholds': Counter(thresholds).most_common(),
            'threshold_best': threshold_best,
            'accuracy': accuracy,
            'accuracy_best': accuracy_best,
            'baseline': cc.baseline(good_labels.tolist()),
            'tn': cm[0, 0], 'tp': cm[1, 1], 'fn': cm[1, 0], 'fp': cm[0, 1],
            'tn_best': cm_best[0, 0], 'tp_best': cm_best[1, 1],
            'fn_best': cm_best[1, 0], 'fp_best': cm_best[0, 1],
            'total': len(good_idxs[0])})

    res_df = pandas.DataFrame(results)
    res_df.set_index('year', drop=True, inplace=True)
    output_fpath = os.path.join(
        config['paths']['out'], 'results_citations_year.html')
    res_df, totals = df_to_table(res_df, output_fpath)
    print_latex(res_df, totals)
    return


def classify_readership_year():
    """
    :return:
    """
    config = load_config()
    logger = logging.getLogger(__name__)

    logger.info('Loading data')
    dl = DataLoader(config['trueid']['metadata'])
    papers_df = dl.load_papers()
    mendeley_df = dl.load_mendeley_metadata()
    logger.info('Done loading data')

    papers_df['reader_count'] = mendeley_df.reader_count
    papers_df.loc[pandas.isnull(papers_df.reader_count), 'reader_count'] = 0
    papers_df.reader_count = papers_df.reader_count.astype(int)

    logger.info('Preparing classification data')
    readership = numpy.array(papers_df.reader_count.tolist())
    labels = numpy.array(papers_df.seminal.tolist())
    years = numpy.array(papers_df.year.tolist())
    logger.info('Got {} citation values, {} labels, {} years'.format(
        len(readership), len(labels), len(years)))

    cc = CitationClassifier()
    results = []

    for year in sorted(numpy.unique(years)):
        good_idxs = numpy.where(years == year)
        good_labels = labels[good_idxs]
        good_readership = readership[good_idxs]
        # need at least 4, two of each type
        if len(good_idxs[0]) <= 3 \
                or len(numpy.where(good_labels == True)[0]) <= 1 \
                or len(numpy.where(good_labels == False)[0]) <= 1:
            continue
        accuracy_best, threshold_best, cm_best = cc.train(
            good_readership.tolist(), good_labels.tolist(),
            print_progress=False, optimal=True)
        accuracy, thresholds, cm = cc.test(
            good_readership.tolist(), good_labels.tolist())
        results.append({
            'year': year,
            'thresholds': Counter(thresholds).most_common(),
            'threshold_best': threshold_best,
            'accuracy': accuracy,
            'accuracy_best': accuracy_best,
            'baseline': cc.baseline(good_labels.tolist()),
            'tn': cm[0, 0] if cm.shape[0] >= 2 and cm.shape[1] >= 2 else None,
            'tp': cm[1, 1] if cm.shape[0] >= 2 and cm.shape[1] >= 2 else None,
            'fn': cm[1, 0] if cm.shape[0] >= 2 and cm.shape[1] >= 2 else None,
            'fp': cm[0, 1] if cm.shape[0] >= 2 and cm.shape[1] >= 2 else None,
            'tn_best': cm_best[0, 0], 'tp_best': cm_best[1, 1],
            'fn_best': cm_best[1, 0], 'fp_best': cm_best[0, 1],
            'total': len(good_idxs[0])})

    res_df = pandas.DataFrame(results)
    res_df.set_index('year', drop=True, inplace=True)
    output_fpath = os.path.join(
        config['paths']['out'], 'results_readership_year.html')
    res_df, totals = df_to_table(res_df, output_fpath)
    print_latex(res_df, totals)
    return


def experiments():
    """
        All statistics we run for our paper
        :return:
        """
    logger = logging.getLogger(__name__)
    logger.info('====================')
    logger.info('Citations aggregate:')
    logger.info('====================')
    classify_citations_basic()
    logger.info('=========================')
    logger.info('Citations per discipline:')
    logger.info('=========================')
    classify_citations_discipline()
    logger.info('===================')
    logger.info('Citations per year:')
    logger.info('===================')
    classify_citations_year()
    logger.info('=====================')
    logger.info('Readership aggregate:')
    logger.info('=====================')
    classify_readership_basic()
    logger.info('==========================')
    logger.info('Readership per discipline:')
    logger.info('==========================')
    classify_readership_discipline()
    logger.info('====================')
    logger.info('Readership per year:')
    logger.info('====================')
    classify_readership_year()
    return
