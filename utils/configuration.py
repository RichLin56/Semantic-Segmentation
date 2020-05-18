#!/usr/bin/env python

"""configuration.py: Contains functions for loading a config file as dict."""

__author__ = "Richard Lindenpuetz"
__email__ = "richard.lindenpuetz@rwth-aachen.de"
__license__ = "MIT"
__version__ = "1.0.0"

import logging

import commentjson as json


def load_config_json(config_file, config_group=None):
    __logger = logging.getLogger(name=__name__)
    __logger.info(
        'Loading config file from this path: {}...'.format(config_file))
    config_dict = {}
    with open(config_file, 'r') as f:
        j_doc = json.load(f)

        # iterate over groups
        for group, group_dict in j_doc.items():
            group_dict = iterate_dict(group_dict)

            config_dict[group] = group_dict

    if config_group is not None:
        __logger.info('Loaded Config_Group: {}:\n{}'.format(
            config_group, json.dumps(config_dict[config_group], indent=4)))
        return config_dict[config_group]
    else:
        __logger.info('Loaded Config_Dict:\n{}'.format(
            json.dumps(config_dict, indent=4)))
        return config_dict


def load_funcs_by_name(module, config_dict):
    funcs = []
    for key in config_dict:
        func_dict = config_dict[key]
        if not isinstance(func_dict, dict):
            continue
        if 'ACTIVATE' in func_dict.keys():
            if func_dict['ACTIVATE'] is False:
                continue
            func_dict.pop('ACTIVATE')
        if hasattr(module, key):
            func = getattr(module, key)
            funcs.append(func(**func_dict))
    if len(funcs) == 0:
        return None
    else:
        return funcs


def iterate_dict(dictionary):
    for key, val in dictionary.items():
        if isinstance(val, dict):
            dictionary[key] = iterate_dict(dictionary[key])
        else:
            # set attributes with value 'None' to None
            if val == 'None':
                dictionary[key] = None
            if val == 'True':
                dictionary[key] = True
            if val == 'False':
                dictionary[key] = False
    return dictionary


if __name__ == '__main__':
    pass

"""
@history
__version__ = "1.0.0" -> basic functionality
"""
