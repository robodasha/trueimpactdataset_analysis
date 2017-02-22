
import os
import logging

import pandas
from research_papers.tools.mendeley_resolver import MendeleyResolver as Mendeley

__author__ = 'robodasha'
__email__ = 'damirah@live.com'


class MendeleyResolver(object):

    def __init__(self, metadata_dir, mendeley_id, mendeley_secret):
        """
        :param metadata_dir:
        :param mendeley_id:
        :param mendeley_secret:
        """
        self._logger = logging.getLogger(__name__)
        self._metadata_dir = metadata_dir
        self._mendeley_id = mendeley_id
        self._mendeley_secret = mendeley_secret

    def _get_doc_metadata(self, mendeley, uri, title, year):
        """
        :param mendeley:
        :param uri:
        :param title:
        :param year:
        :return:
        """
        doc = None
        if isinstance(uri, str) and uri.startswith('dx.doi.org/'):
            doi = uri.replace('dx.doi.org/', '')
            doc = mendeley.get_document_by_doi(doi)
        elif title is not None and year is not None:
            doc = mendeley.get_document_by_title_and_year(title, year)
        return doc

    def _format_authors(self, authors):
        """
        :param authors:
        :return:
        """
        authors_string = ''
        if authors is not None:
            for author in authors:
                authors_string += '{}, {}; '.format(
                    author.last_name, author.first_name)
        return authors_string.strip().strip(';')

    def get_mendeley_metadata(self, papers_df):
        """
        :param papers_df:
        :return: dictionary of {paper_id: metadata}
        """
        self._logger.info('Resolving paper metadata from Mendeley')

        mendeley = Mendeley(self._mendeley_id, self._mendeley_secret)
        fields = Mendeley.FIELDS

        metadata = {}

        for row in papers_df.iterrows():
            uri = row[1]['uri']
            title = row[1]['title']
            year = row[1]['year']
            doc = self._get_doc_metadata(mendeley, uri, title, year)
            if doc:
                doc_meta = []
                for field in fields:
                    if hasattr(doc, field):
                        attr = getattr(doc, field)
                        if field == 'authors':
                            doc_meta.append(self._format_authors(attr))
                        else:
                            doc_meta.append(attr)
                    else:
                        doc_meta.append(None)
                metadata[row[0]] = doc_meta

        metadata_df = pandas.DataFrame.from_dict(metadata, orient='index')
        metadata_df.columns = fields
        self._logger.info('Number of retrieved documents: {}'.format(
            len(metadata_df)))

        metadata_df = metadata_df.join(
            metadata_df['identifiers'].apply(pandas.Series))
        del metadata_df['identifiers']

        return metadata_df

    def save_mendeley_metadata(self, mendeley_df):
        """
        :param mendeley_df:
        :return: None
        """
        out_path = os.path.join(self._metadata_dir, 'mendeley_metadata.csv')
        self._logger.info('Saving Mendeley metadata in {}'.format(out_path))
        mendeley_df.to_csv(out_path)
        self._logger.info('Done saving Mendeley metadata')
        return
