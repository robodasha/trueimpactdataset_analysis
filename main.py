"""
Project main
"""

import sys
import logging

from trueimpactdataset_analysis.logutils import LogUtils
from trueimpactdataset_analysis.tasks.data_tasks import (
    process_raw_responses,
    get_mendeley_metadata,
)
from trueimpactdataset_analysis.tasks.analysis_tasks import (
    analysis
)
from trueimpactdataset_analysis.tasks.statistics_tasks import (
    statistics
)
from trueimpactdataset_analysis.tasks.classification_tasks import (
    experiments
)

__author__ = 'robodasha'
__email__ = 'damirah@live.com'


def exit_app():
    """
    Exit app
    :return: None
    """
    print('Exiting')
    exit()


def menu():
    """
    Just print menu
    :return: None
    """
    print('\nPossible actions:')
    print('=================')
    for key in sorted(menu_actions):
        print('{0}: {1}'.format(key, menu_actions[key].__name__))
    print('Please select option(s)')
    # enable selecting multiple actions which will be run in a sequence
    actions = [i.lower() for i in list(sys.stdin.readline().strip())]
    exec_action(actions)
    return


def exec_action(actions):
    """
    Execute selected action
    :param actions:
    :return: None
    """
    if not actions:
        menu_actions['x']()
    else:
        print('\nSelected the following options: \n{0}'.format(
            [(key, menu_actions[key].__name__)
             if key in menu_actions
             else (key, 'invalid action')
             for key in actions]))
        for action in actions:
            if action in menu_actions:
                menu_actions[action]()
            else:
                pass
    menu()
    return


menu_actions = {
    '0': process_raw_responses,
    '1': get_mendeley_metadata,
    '2': analysis,
    '3': statistics,
    '4': experiments,
    'x': exit_app,
    'y': menu,
}


if __name__ == '__main__':
    LogUtils.setup_logging()
    logger = logging.getLogger(__name__)
    logger.info('Application started')
    try:
        menu()
    except Exception as e:
        logger.exception(e, exc_info=True)

