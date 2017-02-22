
import os
import logging
import configparser

from trueimpactdataset_analysis.data.data_loader import DataLoader
from trueimpactdataset_analysis.data.response_parser import DataParser
from trueimpactdataset_analysis.data.mendeley_resolver import MendeleyResolver

__author__ = 'robodasha'
__email__ = 'damirah@live.com'


def load_config():
    """
    :return:
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config


def process_raw_responses():
    """
    :return:
    """
    config = load_config()
    logger = logging.getLogger(__name__)

    responses_dir = config['trueid']['responses']
    responses_path = os.path.join(responses_dir, 'responses.csv')
    out_dir = config['trueid']['metadata']

    logger.info('Parsing responses in {}'.format(responses_path))
    logger.info('Storing output in {}'.format(out_dir))

    parser = DataParser(responses_path, out_dir)
    papers_df, responses_df = parser.parse_responses()
    parser.save_data(papers_df, responses_df)
    return


def get_mendeley_metadata():
    """
    :return:
    """
    config = load_config()
    logger = logging.getLogger(__name__)

    out_dir = config['trueid']['metadata']

    logger.info('Will store metadata in {}'.format(out_dir))

    ml = MendeleyResolver(
        out_dir, config['mendeley']['client_id'], config['mendeley']['secret'])

    papers_df = DataLoader(config['trueid']['metadata']).load_papers()

    mendeley_df = ml.get_mendeley_metadata(papers_df)
    ml.save_mendeley_metadata(mendeley_df)
    return
