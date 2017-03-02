
import os
import csv
import logging
import configparser

import numpy
import pandas
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

from trueimpactdataset_analysis.data.data_loader import DataLoader

__author__ = 'robodasha'
__email__ = 'damirah@live.com'


def load_config():
    """
    :return:
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config


def normalize_name(name):
    """
    :param name:
    :return:
    """
    name_normalized = ''
    parts = name.split(',')
    for first_name in parts[1].strip().split(' '):
        name_normalized += first_name[0]
    name_normalized += parts[0].strip()
    return name_normalized.lower()


def analysis():
    """
    All stats we used in our paper
    :return:
    """
    config = load_config()
    logger = logging.getLogger(__name__)

    output_dir = config['paths']['out']
    dl = DataLoader(config['trueid']['metadata'])
    papers_df = dl.load_papers()
    responses_df = dl.load_responses()
    mendeley_df = dl.load_mendeley_metadata()
    authors_all = [auth for auths in papers_df.authors.str.split(';')
                   for auth in auths]
    logger.info('All authors: {}'.format(len(authors_all)))

    authors_unique = {normalize_name(name) for name in authors_all}
    logger.info('Unique author names: {}'.format(len(authors_unique)))

    # ======================================================================== #

    plt.style.use('ggplot')
    matplotlib.rcParams.update({'font.size': 7})
    bins = numpy.arange(int(min(papers_df.year)), max(papers_df.year) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))

    seminal_papers = papers_df.seminal == True
    survey_papers = papers_df.seminal == False
    non_empty_years = pandas.notnull(papers_df.year)
    logger.info('Seminal papers age stats:\n{}'.format(
        papers_df[seminal_papers & non_empty_years].year.describe()))
    logger.info('Survey papers age stats:\n{}'.format(
        papers_df[survey_papers & non_empty_years].year.describe()))
    logger.info('Overall age stats:\n{}'.format(
        papers_df[non_empty_years].year.describe()))

    a_heights, a_bins = numpy.histogram(
        papers_df[seminal_papers & non_empty_years].year.tolist(), bins=bins)
    b_heights, b_bins = numpy.histogram(
        papers_df[survey_papers & non_empty_years].year, bins=bins)

    p1 = ax.bar(a_bins[:-1], a_heights, facecolor='cornflowerblue')
    p2 = ax.bar(b_bins[:-1], b_heights, bottom=a_heights, facecolor='seagreen')

    ax.set_xlim((int(min(papers_df.year)), int(max(papers_df.year))))
    ax.set_ylabel('Number of papers')
    ax.set_xlabel('Year')
    ax.set_title('')
    ax.legend((p1[0], p2[0]), ('Seminal', 'Survey'), loc='upper left')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'papers_per_year.pdf')
    fig.savefig(output_path)

    # ======================================================================== #

    logger.info('Seminal papers citations stats:\n{}'.format(
        papers_df[seminal_papers].citations_gs.describe()))
    logger.info('Survey papers citations stats:\n{}'.format(
        papers_df[survey_papers].citations_gs.describe()))

    # ======================================================================== #

    ids = responses_df[responses_df.research_area == 'Other'].index.tolist()
    logger.info('Papers labeled as "Other":\n{}'.format(
        '\n'.join(responses_df[responses_df.index.isin(ids)].topic.tolist())))

    # ======================================================================== #

    bins = len(papers_df.research_area.unique())
    fig, ax = plt.subplots(figsize=(7, 5))

    cnts_overal = papers_df.research_area.value_counts()
    cnts_seminal = papers_df[seminal_papers].research_area.value_counts()
    cnts_survey = papers_df[survey_papers].research_area.value_counts()

    p1 = ax.barh(
        numpy.arange(bins), cnts_seminal[cnts_overal.index].values[::-1],
        facecolor='cornflowerblue')
    p2 = ax.barh(
        numpy.arange(bins),
        cnts_survey[cnts_overal.index].values[::-1],
        left=cnts_seminal[cnts_overal.index].values[::-1],
        facecolor='seagreen')

    ax.set_xlabel('Number of papers')
    ax.set_title('')
    ax.set_yticks(numpy.arange(bins) + 0.40)
    ax.set_yticklabels(cnts_overal.index[::-1])
    ax.set_ylim((0, bins))
    ax.legend((p1[0], p2[0]), ('Seminal', 'Survey'), loc='lower right')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'disciplines.pdf')
    fig.savefig(output_path)

    # ======================================================================== #

    papers_df['reader_count'] = mendeley_df.reader_count
    papers_df.loc[pandas.isnull(papers_df.reader_count), 'reader_count'] = 0
    papers_df.reader_count = papers_df.reader_count.astype(int)
    non_empty_readers = pandas.notnull(papers_df.reader_count)
    logger.info('Mendeley seminal papers reader stats:\n{}'.format(
        papers_df[seminal_papers & non_empty_readers].reader_count.describe()))
    logger.info('Mendeley survey papers reader stats:\n{}'.format(
        papers_df[survey_papers & non_empty_readers].reader_count.describe()))

    # ======================================================================== #

    citations_wos_df = pandas.read_csv(
        os.path.join(config['trueid']['metadata'], 'citations_wos.csv'))
    citations_wos_df.set_index('id', inplace=True, drop=True)
    citations_wos_df['citations_gs'] = pandas.Series(papers_df.citations_gs)
    citations_wos_df['seminal'] = pandas.Series(papers_df.seminal)
    logger.info('WOS citations stats:\n{}'.format(
        citations_wos_df.groupby('seminal').describe()))

    logger.info('Pearson correlation GS & WOS:\n{}'.format(
        scipy.stats.pearsonr(
            citations_wos_df.citations_wos, citations_wos_df.citations_gs)))
    logger.info('Spearman correlation GS & WOS:\n{}'.format(
        scipy.stats.spearmanr(
            citations_wos_df.citations_wos, citations_wos_df.citations_gs)))

    logger.info('Seminal pearson correlation GS & WOS:\n{}'.format(
        scipy.stats.pearsonr(
            citations_wos_df[citations_wos_df.seminal == True].citations_wos,
            citations_wos_df[citations_wos_df.seminal == True].citations_gs)))
    logger.info('Seminal spearman correlation GS & WOS:\n{}'.format(
        scipy.stats.spearmanr(
            citations_wos_df[citations_wos_df.seminal == True].citations_wos,
            citations_wos_df[citations_wos_df.seminal == True].citations_gs)))

    logger.info('Survey pearson correlation GS & WOS:\n{}'.format(
        scipy.stats.pearsonr(
            citations_wos_df[citations_wos_df.seminal == False].citations_wos,
            citations_wos_df[citations_wos_df.seminal == False].citations_gs)))
    logger.info('Survey spearman correlation GS & WOS:\n{}'.format(
        scipy.stats.spearmanr(
            citations_wos_df[citations_wos_df.seminal == False].citations_wos,
            citations_wos_df[citations_wos_df.seminal == False].citations_gs)))

    return
