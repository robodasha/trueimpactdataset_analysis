
import os
import logging

import pandas

__author__ = 'robodasha'
__email__ = 'damirah@live.com'


class DataLoader(object):

    def __init__(self, metadata_dir):
        self._logger = logging.getLogger(__name__)
        self._metadata_dir = metadata_dir

    def _load_df(self, df_path, index_col=None):
        """
        :param df_path:
        :return:
        """
        self._logger.info('Loading {}'.format(df_path))
        df = pandas.read_csv(df_path)
        if index_col is not None:
            df.set_index(index_col, inplace=True, drop=True)
        self._logger.info('Done, got {} items'.format(len(df)))
        return df

    def load_papers(self):
        """
        :return:
        """
        papers_path = os.path.join(self._metadata_dir, 'papers.csv')
        responses_path = os.path.join(self._metadata_dir, 'responses.csv')

        papers_df = self._load_df(papers_path, index_col='id')
        responses_df = self._load_df(responses_path, index_col='id')

        self._logger.info('Adding column \'seminal\' to papers DataFrame')
        seminal_ids = [int(x) for x in responses_df.seminal_id
                       if pandas.notnull(x)]
        papers_df['seminal'] = [True if idx in seminal_ids else False
                                for idx in papers_df.index]
        self._logger.info('Done, got {} seminal papers'.format(
            sum(papers_df.seminal)))

        self._logger.info('Adding column \'research_area\' to papers DF')
        seminal_papers = pandas.notnull(responses_df.seminal_id)
        survey_papers = pandas.notnull(responses_df.survey_id)
        research_area = dict(zip(
            responses_df[seminal_papers].seminal_id,
            responses_df[seminal_papers].research_area))
        research_area.update(dict(zip(
            responses_df[survey_papers].survey_id,
            responses_df[survey_papers].research_area)))
        papers_df['research_area'] = pandas.Series(research_area)

        return papers_df

    def load_responses(self):
        """
        :return:
        """
        responses_path = os.path.join(self._metadata_dir, 'responses.csv')
        responses_df = self._load_df(responses_path, index_col='id')
        return responses_df

    def load_mendeley_metadata(self):
        """
        :return:
        """
        mendeley_path = os.path.join(
            self._metadata_dir, 'mendeley_metadata.csv')
        responses_path = os.path.join(self._metadata_dir, 'responses.csv')

        mendeley_df = self._load_df(mendeley_path, index_col='Unnamed: 0')
        responses_df = self._load_df(responses_path, index_col='id')

        self._logger.info('Adding column \'seminal\' to Mendeley DataFrame')
        seminal_ids = [int(x) for x in responses_df.seminal_id
                       if pandas.notnull(x)]
        mendeley_df['seminal'] = [True if idx in seminal_ids else False
                                  for idx in mendeley_df.index]
        self._logger.info('Done, got {} seminal papers in mendeley'.format(
            sum(mendeley_df.seminal)))

        self._logger.info('Adding column \'research_area\' to mendeley DF')
        seminal_papers = pandas.notnull(responses_df.seminal_id)
        survey_papers = pandas.notnull(responses_df.survey_id)
        research_area = dict(zip(
            responses_df[seminal_papers].seminal_id,
            responses_df[seminal_papers].research_area))
        research_area.update(dict(zip(
            responses_df[survey_papers].survey_id,
            responses_df[survey_papers].research_area)))
        mendeley_df['research_area'] = pandas.Series(research_area)

        return mendeley_df
