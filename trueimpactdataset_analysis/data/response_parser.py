
import os
import csv
import codecs
import logging

import pandas

__author__ = 'robodasha'
__email__ = 'damirah@live.com'


class DataParser(object):

    # mapping of columns in the responses.csv file
    _MAPPING = {
        'response': {
            0: 'timestamp', 1: 'research_area', 2: 'research_area_details',
            3: 'interests', 4: 'years_since_phd', 5: 'num_pubs', 20: 'topic',
            21: 'comment'
        },
        'seminal': {
            6: 'user_input', 7: 'uri', 8: 'title', 9: 'authors', 10: 'year',
            11: 'citations_gs', 12: 'note'
        },
        'survey': {
            13: 'user_input', 14: 'uri', 15: 'title', 16: 'authors', 17: 'year',
            18: 'citations_gs', 19: 'note'
        }
    }

    def __init__(self, responses_path, metadata_dir):
        """
        :param responses_path:
        :param metadata_dir:
        """
        self._logger = logging.getLogger(__name__)
        self._responses_path = responses_path
        self._metadata_dir = metadata_dir

    def _get_columns(self, row, output_list, columns, additional_values=None,
                     required=2):
        """
        :param row: one row of the input CSV
        :param output_list: the list to which the processed data will
                            be appended (this is used to generate an id/key)
        :param columns: which columns of the input CSV should be used
        :param additional_values: any additional values (such as a key/id
                                  pointing to another table) to be added
                                  to the data
        :param required: how many values must be set at least for the data
                         to not be considered empty, if the result has less
                         than required values, nothing is appended to the
                         output list and None is returned
        :return: tuple -- id and the data that was appended to the output list
        """
        item = [len(output_list)]
        if additional_values is not None:
            item.extend(additional_values)
        item.extend([row[x] if row[x] != '-' and len(row[x]) > 0 else None
                     for x in sorted(columns.keys())])
        if sum(x is not None for x in item) >= required:
            output_list.append(item)
            return item[0], item
        else:
            return None, None

    def _list_to_df(self, cols, data):
        """
        :param cols:
        :param data:
        :return:
        """
        df = pandas.DataFrame(data, columns=cols)
        df.set_index('id', inplace=True, drop=True)
        return df

    def parse_responses(self):
        """
        :return: tuple(pandas.DataFrame, pandas.DataFrame) of papers
                 and responses
        """
        papers = []
        resps = []
        mapping = DataParser._MAPPING

        self._logger.info('Reading file {}'.format(self._responses_path))
        with codecs.open(self._responses_path, encoding='ISO-8859-1') as fp:
            reader = csv.reader(fp)
            for idx, row in enumerate(reader):
                # skip header
                if idx == 0:
                    continue

                seminal_id, seminal_paper = self._get_columns(
                    row, papers, mapping['seminal'], [idx], 5)
                survey_id, survey_paper = self._get_columns(
                    row, papers, mapping['survey'], [idx], 5)
                if seminal_id is None and survey_id is None:
                    continue
                self._get_columns(
                    row, resps, DataParser._MAPPING['response'],
                    [seminal_id, survey_id])

        self._logger.info('Done reading data, converting to DataFrames')

        responses_df = self._list_to_df(
            ['id', 'seminal_id', 'survey_id']
            + [item[1] for item in sorted(mapping['response'].items())],
            resps)

        papers_df = self._list_to_df(
            ['id', 'row_id']
            + [item[1] for item in sorted(mapping['seminal'].items())],
            papers)

        self._logger.info('Got {0} responses and {1} papers'.format(
            len(responses_df), len(papers_df)))
        self._logger.info('Seminal: {}'.format(
            sum([x[1] is not None for x in resps])))
        self._logger.info('Survey: {}'.format(
            sum([x[2] is not None for x in resps])))

        self._logger.info('Appending column DOI')
        is_doi = pandas.notnull(papers_df.uri) \
                 & papers_df['uri'].str.startswith('dx.doi.org/')
        papers_df['doi'] = pandas.DataFrame(
            papers_df[is_doi]['uri'].str.split('dx.doi.org/').tolist(),
            columns=['pre', 'doi'], index=papers_df[is_doi].index).doi
        self._logger.info('Got {} DOIs'.format(
            sum(pandas.notnull(papers_df.doi))))

        return papers_df, responses_df

    def save_data(self, papers_df, responses_df):
        """
        :param papers_df:
        :param responses_df:
        :return:
        """
        self._logger.info('Saving papers')
        papers_df.to_csv(
            os.path.join(self._metadata_dir, 'papers.csv'),
            columns=['user_input', 'uri', 'doi', 'title',
                     'authors', 'year', 'citations_gs'])
        self._logger.info('Done saving papers')
        self._logger.info('Saving responses')
        responses_df.to_csv(
            os.path.join(self._metadata_dir, 'responses.csv'))
        self._logger.info('Done saving responses')
        return
