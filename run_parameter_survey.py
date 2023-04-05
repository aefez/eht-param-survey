# run_paramsurvey.py
# runs sgr A* imaging survey in parallel, using the paramsurvey package
#
#    Copyright (C) 2020 Antonio Fuentes et al.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import range
from builtins import object

import itertools
import os
import yaml
import sys
import time
import glob
import shutil

import yaml
import json
import re
import numpy as np

import paramsurvey
import paramsurvey.params

import pandas as pd

# FIX PICKLE's VESRSION to 4 (not 5 newly used in python 3.8)
#import pickle
#pickle.HIGHEST_PROTOCOL = 5

# This is necessary to fix bug in yaml where e.g. 1e6 is interpreted as a string
# see https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number 
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

###############################################################################
# SETUP
###############################################################################

# This defines the ParameterSurvey class -- swap out for Sgr A* / M87,  etc
from parameter_survey import ParameterSurvey, make_results_hdf5

# This defines the default .yaml file where parameters are defined
YAML_DEFAULT = 'static'

# Hides print statements  from script
HIDEPRINT=False

NCORES=1

###############################################################################
# MAIN
###############################################################################

def main(test=False, hdf5_only=False, hdf5_skip=False, save_hdf5=False,
         offset=0, yamlfile=YAML_DEFAULT):
    
    """Run parameter survey

     This script uses the configurations in "params_sgra_survey.yml" as
     the default, and then look for "sgra_survey.yml" at different
     locations such as "/etc/sgra_survey.yml", "~/sgra_survey.yml", and
     "./sgra_survey.yml" to reconfig the survey.

     These additional YAML files are useful to specify per-machine (using
     "/etc/sgra_survey.yml" or "~/sgra_survey.yml") and per-survey (using
     "./sgra_survey.yml") setups.

     This script also accepts a single argument "--dry-run" or "--test"
     to print out udpated `params` without actually running the survey.
    """

    print('\n... READY FOR A WONDERFUL PARAMETER SURVEY??? ...\n')
    paramsurvey.init(backend='multiprocessing', ncores=NCORES)  # 'multiprocessing', 'ray'
    rank = 0  # Only one process runs all of this code

    # Load in the top level yaml file with parameter survey specifications
    path = ['/etc', '~', '.', './paramfiles']
    ext  = ['yaml', 'yml']

    params0 = {}
    n = os.path.splitext(os.path.basename(yamlfile))[0]
    for p in path:
        for x in ext:
            full = os.path.expanduser('{}/{}.{}'.format(p, n, x))
            if not os.path.exists(full):
                continue
            with open(full, 'r') as f:
                if rank==0:
                    print('Loading `params` from "{}"'.format(full))
                params0.update(yaml.load(f, Loader=loader))
            break

    # if the top level yaml file directs to another, load that
    # but overwrite any parameters defined in the top level file 
            
    if 'paramfile' in params0.keys():
        params = {}
        n = os.path.splitext(os.path.basename(params0['paramfile']))[0]
        for p in path:
            for x in ext:
                full = os.path.expanduser('{}/{}.{}'.format(p, n, x))
                if not os.path.exists(full):
                    continue
                with open(full, 'r') as f:
                    if rank==0:
                        print('Loading further `params` from "{}"'.format(full))
                    params.update(yaml.load(f, Loader=loader))
                    params.update(params0)
                    del params['paramfile']
                break
    else:
        params = params0

    # survey parameters
    subdir           = params.pop('subdir',           False)
    save_uvfits      = params.pop('save_uvfits',      False)
    save_imgsums     = params.pop('save_imgsums',     False)
    save_stats       = params.pop('save_stats',       False)
    ehproc           = params.pop('ehproc',              -1)
    ttype            = params.pop('ttype',         'direct')
    hideprint        = params.pop('hideprint',    HIDEPRINT)
    paramhdf5        = params.pop('paramhdf5',        False)
    
    # input and output paths
    inpath  = params.pop('inpath',  './')
    outpath = params.pop('outpath', './')

    #################################################
    # parameters specifically for sgr A*...
    run_static       = params.pop('run_static'      , True)
    run_sw           = params.pop('run_sw'          , False)
    run_di           = params.pop('run_di'          , False)

    # data properties
    top_path_files     = params.pop('top_path_files',  None)
    #################################################

    # Get top images from top_path_file
    if isinstance(top_path_files, list): top_path_files = top_path_files[0]
    
    params.update({'top_path': [False]})
    if top_path_files and not run_static:
        top_images = []
        with open(top_path_files) as f:
            for line in f:
                top_images.append(line.rstrip('\n'))   
        params.update({'top_path': top_images})
    
    # If dataset is normalized, replace total flux with 1
    if params['normalized'][0]: params.update({'zbl_tot': 1.0})
        
    # make all remaining parameters into list
    for key in params.keys():
        if not isinstance(params[key],list):
            params[key] = [params[key]]

    # List parameters if test (--dry-run or --test flag)
    if rank==0:
        print('\nehproc:', ehproc)
        print('\nsave_uvfits:', save_uvfits)
        print('save_imgsums:', save_imgsums)
        print('save_stats:', save_stats)
        print('\nrun_static:', run_static)
        print('run_sw:', run_sw)
        print('run_di:', run_di)
        print('\nmodels:', params['model'],'\n')
        print('parameters (w/o topset priors):')
        try:
            from pprint import pprint
            pprint(params)
        except:
            print(params)
    if test:
        exit(0)
    
###############################################################################
    # Load/save params from/to HDF5 part
###############################################################################

    if paramhdf5 and not save_hdf5:
        # load dataframe
        print('\n... LOADING DATAFRAME {} ...'.format(os.path.basename(paramhdf5)))
        psets = pd.read_hdf(paramhdf5, 'parameters')
        psets['netcal'], psets['LMTcal'], psets['JCMTcal'] = False, False, False
        
    else:
        # create all parameter combinations
        psets = paramsurvey.params.product(params, infer_category=False)
        
        # scattering and psd noise parameters might be duplicated, drop these rows
        len0 = len(psets)
        psets.loc[psets.deblurr == False, ['ref_type', 'ref_scale']] = False, 1.
        psets.loc[psets.psd_noise == False, ['psd_a', 'psd_u0', 'psd_b', 'psd_c']] = 1., 1., 1., 1.
        psets = psets.drop_duplicates(ignore_index=True)    
        len1 = len(psets)
        
        if len0 != len1:
            print('\n... {} DUPLICATED COMBINATIONS RELATED TO SCATTERING AND PSD NOISE WERE REMOVED ...\n'\
                  .format(len0-len1))

        # add epoch to psets
        psets['epoch'] = np.array([psets['model'][i].split('_')[1] for i in range(len(psets))])        
        
        # this code expects a running counter, so add it to the psets
        psets['id'] = np.array(['{:08}'.format(x+offset) for x in range(len(psets))])
        
        if save_hdf5:
            # export dataframe to hdf5 file
            print('\n... EXPORTING DATAFRAME TO HDF5 ...')
            columns = [col for col in list(psets.columns)
                       if col not in list(psets.columns[0:49])
                       +['top_path', 'epoch', 'id']] #drop dynamic imaging parameters
            psets.drop(columns=columns).to_hdf(paramhdf5, 'parameters', mode='w', complevel=9)
            
            print('\n That took {} minutes'.format((time.time() - start)/60.))
            exit(0)
    
    if rank==0:
        print('\n... IDENTIFIED %i PARAMETER COMBINATIONS ...\n' % len(psets))

###############################################################################
    # Survey part
###############################################################################

    if not hdf5_only:
        user_kwargs = {
            'n': len(psets),
            'inpath': inpath,
            'outpath': outpath,
            'subdir': subdir,
            'ehproc': ehproc,
            'ttype': ttype,
            'hideprint': hideprint,
            'run_static': run_static,
            'run_sw': run_sw,
            'run_di': run_di,
            'save_uvfits': save_uvfits,
            'save_imgsums': save_imgsums,
            'save_stats': save_stats,
        }

        paramsurvey.map(_run_survey, psets, user_kwargs=user_kwargs)

    if hdf5_skip:
        return

###############################################################################
    # HDF5 part
###############################################################################

    if rank==0:
        print('\n... COMPILING RESULTS INTO HDF5 ...\n')
        print(outpath)

        try:
            make_results_hdf5(savepath[:-1], model +  '_paramstats')
        except:
            print('FAILED TO MAKE HDF5!')

        print('\n... FINISHED SURVEY FOR %s ...\n' % model)
        sys.stdout.flush()
     
###############################################################################
# Helpers
###############################################################################

class HiddenPrints:
    """Suppresses printing from the loop function
    """

    def __init__(self, hideprint=False): 
        self.hideprint=hideprint

    def __enter__(self):
        if self.hideprint:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        else:
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hideprint:
            sys.stdout.close()
            sys.stdout = self._original_stdout
        else:
            pass

def _run_survey(pset, system_kwargs, user_kwargs):
    for name in ('inpath', 'outpath', 'subdir', 'ttype',
                 'save_uvfits', 'save_imgsums', 'save_stats',
                 'run_static','run_sw','run_di'):
        pset[name] = user_kwargs[name]

    ehproc = user_kwargs['ehproc']
    hideprint = user_kwargs['hideprint']
#    i = int(pset['id'])

    with HiddenPrints(hideprint=hideprint):
        survey = ParameterSurvey(pset, ehproc)
        survey.run()

    return

###############################################################################
# RUN
###############################################################################

if __name__ == '__main__':
    test = False
    hdf5_only = False
    hdf5_skip = False
    save_hdf5 = False
    offset = 0
    yamlfile = YAML_DEFAULT
    for arg in sys.argv[1::]:
        if arg == '--dry-run' or arg == '--test':
            test=True
        if arg == '--hdf5only':
            hdf5_only=True
        if arg == '--hdf5skip':
            hdf5_skip=True
        if arg == '--savehdf5':
            save_hdf5=True
        if arg == '--offset':
            offset=int(sys.argv[sys.argv.index(arg) + 1])
        if os.path.splitext(arg)[-1] in ['.yaml', '.yml']:
            yamlfile = os.path.splitext(arg)[0] 
    
    start = time.time()
    main(test=test, hdf5_only=hdf5_only, hdf5_skip=hdf5_skip,
         save_hdf5=save_hdf5, offset=offset, yamlfile=yamlfile)
    print('\n That took {} minutes'.format((time.time() - start)/60.))
    
