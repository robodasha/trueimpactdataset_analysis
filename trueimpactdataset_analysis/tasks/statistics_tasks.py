
import logging
import configparser

import numpy
from scipy import stats
import pandas

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


def citations_ttest():
    """
    :return:
    """
    config = load_config()
    logger = logging.getLogger(__name__)

    logger.info('Loading data')
    dl = DataLoader(config['trueid']['metadata'])
    papers_df = dl.load_papers()
    responses_df = dl.load_responses()
    logger.info('Done loading data')

    nonull_ids = papers_df[
        pandas.notnull(papers_df.citations_gs)].index.tolist()
    pairs = responses_df.seminal_id.isin(nonull_ids) \
            & responses_df.survey_id.isin(nonull_ids)
    seminal_ids = responses_df[pairs].seminal_id.tolist()
    survey_ids = responses_df[pairs].survey_id.tolist()

    citations_seminal = papers_df[
        papers_df.index.isin(seminal_ids)].citations_gs.tolist()
    citations_survey = papers_df[
        papers_df.index.isin(survey_ids)].citations_gs.tolist()

    logger.info('Got {} seminal and {} survey citations'.format(
        len(citations_seminal), len(citations_survey)))

    # stats.ttest_ind performs two tailed t-test, to obtain a p-value
    # for one tailed t-test we need to divide p by 2
    result = stats.ttest_ind(citations_seminal, citations_survey)
    logger.info('Independent t-test: p={}'.format(result[1]/2))
    return


def citations_ttest_discipline():
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

    logger.info('Preparing data')
    citations = numpy.array(papers_df.citations_gs.tolist())
    labels = numpy.array(papers_df.seminal.tolist())
    groups = numpy.array(papers_df.research_area_category.tolist())
    logger.info('Got {} citation values, {} labels, {} group labels'.format(
        len(citations), len(labels), len(groups)))

    logger.info('Finding \'other\' category')
    other_cat = list(research_area_category[1]).index('Other')
    logger.info('Will skip {} values'.format(
        len(numpy.where(groups == other_cat)[0])))

    results = []
    skipped_categories = []
    for category in numpy.unique(groups):
        if category == other_cat:
            continue
        good_idxs = numpy.where(groups == category)
        good_labels = labels[good_idxs]
        good_citations = citations[good_idxs]
        seminal_citations = good_citations[numpy.where(good_labels == True)]
        survey_citations = good_citations[numpy.where(good_labels == False)]
        if len(good_idxs[0]) <= 3 \
                or len(numpy.where(good_labels == True)[0]) <= 1 \
                or len(numpy.where(good_labels == False)[0]) <= 1:
            logger.info('Insufficient number of samples for {}'.format(
                research_area_category[1][category]))
            skipped_categories.append(research_area_category[1][category])
            continue

        logger.info('Got {} seminal and {} survey citations'.format(
            len(seminal_citations), len(survey_citations)))

        # stats.ttest_ind performs two tailed t-test, to obtain a p-value
        # for one tailed t-test we need to divide p by 2
        result = stats.ttest_ind(seminal_citations, survey_citations)
        logger.info('Independent t-test: p={}'.format(result[1] / 2))

        results.append({
            'res_area': research_area_category[1][category],
            'p': result[1]/2,
            'total': len(good_idxs[0])})

    results_df = pandas.DataFrame(results)
    results_df.set_index('res_area', inplace=True, drop=True)
    logger.info('Results:\n{}'.format(results_df))
    logger.info('Total number of samples: {}'.format(sum(results_df.total)))
    logger.info('Skipped categories: {}'.format(skipped_categories))
    return


def citations_ttest_year():
    """
    :return:
    """
    config = load_config()
    logger = logging.getLogger(__name__)

    logger.info('Loading data')
    papers_df = DataLoader(config['trueid']['metadata']).load_papers()
    logger.info('Done loading data')

    logger.info('Preparing data')
    citations = numpy.array(papers_df.citations_gs.tolist())
    labels = numpy.array(papers_df.seminal.tolist())
    years = numpy.array(papers_df.year.tolist())
    logger.info('Got {} citation values, {} labels, {} years'.format(
        len(citations), len(labels), len(years)))

    results = []
    skipped_categories = []
    for year in sorted(numpy.unique(years)):
        good_idxs = numpy.where(years == year)
        good_labels = labels[good_idxs]
        good_citations = citations[good_idxs]
        seminal_citations = good_citations[numpy.where(good_labels == True)]
        survey_citations = good_citations[numpy.where(good_labels == False)]
        if len(good_idxs[0]) <= 3 \
                or len(numpy.where(good_labels == True)[0]) <= 1 \
                or len(numpy.where(good_labels == False)[0]) <= 1:
            logger.info('Insufficient number of samples for {}'.format(year))
            skipped_categories.append(year)
            continue

        logger.info('Got {} seminal and {} survey citations'.format(
            len(seminal_citations), len(survey_citations)))

        # stats.ttest_ind performs two tailed t-test, to obtain a p-value
        # for one tailed t-test we need to divide p by 2
        result = stats.ttest_ind(seminal_citations, survey_citations)
        logger.info('Independent t-test: p={}'.format(result[1] / 2))

        results.append({
            'year': year,
            'p': result[1]/2,
            'total': len(good_idxs[0])})

    results_df = pandas.DataFrame(results)
    results_df.set_index('year', inplace=True, drop=True)
    logger.info('Results:\n{}'.format(results_df))
    logger.info('Total number of samples: {}'.format(sum(results_df.total)))
    logger.info('Skipped categories: {}'.format(skipped_categories))
    return


def readership_ttest():
    """
    :return:
    """
    config = load_config()
    logger = logging.getLogger(__name__)

    logger.info('Loading data')
    dl = DataLoader(config['trueid']['metadata'])
    mendeley_df = dl.load_mendeley_metadata()
    responses_df = dl.load_responses()
    logger.info('Done loading data')

    pairs = pandas.notnull(responses_df.seminal_id) \
            & pandas.notnull(responses_df.survey_id)
    seminal_ids = responses_df[pairs].seminal_id.tolist()
    survey_ids = responses_df[pairs].survey_id.tolist()

    seminal_ids_all = responses_df.seminal_id.tolist()
    survey_ids_all = responses_df.survey_id.tolist()

    logger.info('Found {} seminal and {} survey papers in Mendeley'.format(
        sum(mendeley_df.index.isin(seminal_ids_all)),
        sum(mendeley_df.index.isin(survey_ids_all))))

    readership_seminal = mendeley_df[
        mendeley_df.index.isin(seminal_ids)].reader_count.tolist()
    readership_survey = mendeley_df[
        mendeley_df.index.isin(survey_ids)].reader_count.tolist()

    # papers which are missing in mendeley have 0 readers
    if len(readership_seminal) < sum(pairs):
        readership_seminal.extend(
            numpy.zeros(sum(pairs) - len(readership_seminal)))
    if len(readership_survey) < sum(pairs):
        readership_survey.extend(
            numpy.zeros(sum(pairs) - len(readership_survey)))

    logger.info('Got {} seminal and {} survey citations'.format(
        len(readership_seminal), len(readership_survey)))

    # stats.ttest_ind performs two tailed t-test, to obtain a p-value
    # for one tailed t-test we need to divide p by 2
    result = stats.ttest_ind(readership_seminal, readership_survey)
    logger.info('Independent t-test: p={}'.format(result[1] / 2))
    return


def readership_ttest_discipline():
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

    logger.info('Preparing data')
    readership = numpy.array(papers_df.reader_count.tolist())
    labels = numpy.array(papers_df.seminal.tolist())
    groups = numpy.array(papers_df.research_area_category.tolist())
    logger.info('Got {} readership values, {} labels, {} group labels'.format(
        len(readership), len(labels), len(groups)))

    logger.info('Finding \'other\' category')
    other_cat = list(research_area_category[1]).index('Other')
    logger.info('Will skip {} values'.format(
        len(numpy.where(groups == other_cat)[0])))

    results = []
    skipped_categories = []
    for category in numpy.unique(groups):
        if category == other_cat:
            continue
        good_idxs = numpy.where(groups == category)
        good_labels = labels[good_idxs]
        good_readership = readership[good_idxs]
        seminal_readership = good_readership[numpy.where(good_labels == True)]
        survey_readership = good_readership[numpy.where(good_labels == False)]
        # need at least 3 papers representing both groups
        if len(good_idxs[0]) <= 3 \
                or len(numpy.where(good_labels == True)[0]) <= 1 \
                or len(numpy.where(good_labels == False)[0]) <= 1:
            logger.info('Insufficient number of samples for {}'.format(
                research_area_category[1][category]))
            skipped_categories.append(research_area_category[1][category])
            continue

        logger.info('Got {} seminal and {} survey citations'.format(
            len(seminal_readership), len(survey_readership)))

        # stats.ttest_ind performs two tailed t-test, to obtain a p-value
        # for one tailed t-test we need to divide p by 2
        result = stats.ttest_ind(seminal_readership, survey_readership)
        logger.info('Independent t-test: p={}'.format(result[1] / 2))

        results.append({
            'res_area': research_area_category[1][category],
            'p': result[1]/2,
            'total': len(good_idxs[0])})

    results_df = pandas.DataFrame(results)
    results_df.set_index('res_area', inplace=True, drop=True)
    logger.info('Results:\n{}'.format(results_df))
    logger.info('Total number of samples: {}'.format(sum(results_df.total)))
    logger.info('Skipped categories: {}'.format(skipped_categories))
    return


def readership_ttest_year():
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

    logger.info('Preparing data')
    readership = numpy.array(papers_df.reader_count.tolist())
    labels = numpy.array(papers_df.seminal.tolist())
    years = numpy.array(papers_df.year.tolist())
    logger.info('Got {} readership values, {} labels, {} years'.format(
        len(readership), len(labels), len(years)))

    results = []
    skipped_categories = []
    for year in sorted(numpy.unique(years)):
        good_idxs = numpy.where(years == year)
        good_labels = labels[good_idxs]
        good_readership = readership[good_idxs]
        seminal_readership = good_readership[numpy.where(good_labels == True)]
        survey_readership = good_readership[numpy.where(good_labels == False)]
        # need at least 3 papers representing both groups
        if len(good_idxs[0]) <= 3 \
                or len(numpy.where(good_labels == True)[0]) <= 1 \
                or len(numpy.where(good_labels == False)[0]) <= 1:
            logger.info('Insufficient number of samples for {}'.format(year))
            skipped_categories.append(year)
            continue

        logger.info('Got {} seminal and {} survey citations'.format(
            len(seminal_readership), len(survey_readership)))

        # stats.ttest_ind performs two tailed t-test, to obtain a p-value
        # for one tailed t-test we need to divide p by 2
        result = stats.ttest_ind(seminal_readership, survey_readership)
        logger.info('Independent t-test: p={}'.format(result[1] / 2))

        results.append({
            'year': year,
            'p': result[1]/2,
            'total': len(good_idxs[0])})

    results_df = pandas.DataFrame(results)
    results_df.set_index('year', inplace=True, drop=True)
    logger.info('Results:\n{}'.format(results_df))
    logger.info('Total number of samples: {}'.format(sum(results_df.total)))
    logger.info('Skipped categories: {}'.format(skipped_categories))
    return


def statistics():
    """
    All statistics we run for our paper
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info('=================')
    logger.info('Citations t-test:')
    logger.info('=================')
    citations_ttest()
    logger.info('================================')
    logger.info('Citations per discipline t-test:')
    logger.info('================================')
    citations_ttest_discipline()
    logger.info('==========================')
    logger.info('Citations per year t-test:')
    logger.info('==========================')
    citations_ttest_year()
    logger.info('==================')
    logger.info('Readership t-test:')
    logger.info('==================')
    readership_ttest()
    logger.info('=================================')
    logger.info('Readership per discipline t-test:')
    logger.info('=================================')
    readership_ttest_discipline()
    logger.info('===========================')
    logger.info('Readership per year t-test:')
    logger.info('===========================')
    readership_ttest_year()
    return
