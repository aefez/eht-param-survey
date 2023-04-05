import re
import glob
import pandas as pd
import os
from tqdm.notebook import tqdm

class SurveySummary(object):
    """A class for top set parameter selection.
    
    This class loads all _params.txt and _stats.txt files, either one by one or
    from a table containing all parameters combinations and stats. Then, it
    creates a DataFrame with all the data and filters it according to the
    metric threshold specified, returning a DataFrame with the top set
    parameters.
    
    Attributes:
        inpath (str): path to _params.txt and _stats.txt files.
        outpath (str): path to results folder.
        models (list): list of synthetic models used for parameter selection.
        param_list (list): list of parameters used in the top set selection.
        metric (str): metric used as selection criteria.
        model_thres (dict): dictionary of synthetic models and their
            corresponding metric threshold.
        param_files (list): list of paths to _params.txt files.
        table_all (dataframe): dataframe containing all synthetic models and
            parameters.
        table_top (dataframe): dataframe containing the intersection of
            parameters which produce reconstructions with metrics above the
            threshold.
            
    """
    
    def __init__(self, inpath, outpath, models, param_list, metric,
                 model_thres):

        self.inpath = inpath
        self.outpath = outpath
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
        self.models = models
        self.param_list = param_list
        self.metric = metric
        self.model_thres = model_thres
        self.param_files = None
        self.table_all = None
        self.table_top = None

    def load_files(self):
        """Load _params.txt files.

        """
        
        print('\n---> Getting list of table files')

        # filenames = {}
        # for model in self.models:
        #     files = self.inpath + model + '_*params.txt'
        #     filenames.update({model: glob.glob(files)})

        self.param_files = glob.glob(self.inpath+'/*_params.txt')

        print('\n{} table files were found'.format(len(self.param_files)))

    def make_table(self):
        """Make a DataFrame containing all parameters.
        
        """
        
        print('\n---> Creating table with all parameters')

        param_names = ['file', 'ttype', 'scattered_truth', 'deblur',
                       'noise_floor', 'zbl_tot', 'prepared', 'scanavg',
                       'netcal', 'LMTcal', 'zbl_frac', 'fov', 'npixels',
                       'xmaj', 'xmin', 'xpa', 'sys_noise_imaging',
                       'niter_static', 'priortype', 'fwhm', 'blurfrac',
                       'maxit', 'diag_closure', 'amp', 'cphase', 'logcamp',
                       'simple', 'l1', 'tv', 'tv2', 'flux', 'rgauss',
                       'epsilon_tv', 'niter_sw', 'warp_method', 'numLinIters',
                       'interiorPriors', 'reassign_apxImgs',
                       'variance_img_diff', 'powerDropoff', 'covfrac',
                       'sw_amp', 'sw_cphase', 'sw_logcamp', 'sw_bs',
                       'metric_di', 'alpha_dt_rel', 'alpha_dI_rel']
                       
        stat_names  = ['file', 'chisq_vis_descattered', 'chisq_vis_scattered',
                       'chisq_amp_descattered', 'chisq_amp_scattered',
                       'chisq_cphase_descattered', 'chisq_cphase_scattered',
                       'chisq_logcamp_descattered', 'chisq_logcamp_scattered',
                       'chisq_camp_descattered', 'chisq_camp_scattered',
                       'nxcorr_descattered', 'nxcorr_scattered',
                       'nrmse_descattered', 'nrmse_scattered',
                       'rssd_descattered', 'rssd_scattered']
        
        #print(len(param_names), len(stat_names))

        param_list_ = [param
                       for param in self.param_list
                       if param not in ['avgtime', 'sys_noise']]

        df_list = []
        for f in tqdm(self.param_files):

            m = re.search('^(.+?)_[a-zA-Z]+?cal.+?$', os.path.basename(f))
            if not m:
                print('Failed to get model name from "{}"'\
                      .format(os.path.basename(f)))
                continue

            model = m.group(1)
            if model not in self.models:
                print('Unknown model "{}"'.format(model))
                continue

            print('model:', model)

            ptab = pd.read_table(f, sep=' ', names=param_names)
            stab = pd.read_table(f[:-10]+'sw_stats.txt', sep=' ',
                                 names=stat_names)
            if not ptab.file.is_unique:
                raise ValueError('File names in {} are not unique'.format(f))
            if not stab.file.is_unique:
                raise ValueError('File names in {} are not unique'\
                                 .format(f[:-10]+'sw_stats.txt'))

            ptab = ptab.sort_values(by=['file'], ignore_index=True)
            stab = stab.sort_values(by=['file'], ignore_index=True)
            if not all(ptab.file.str[:-10] == stab.file.str[:-12]):
                raise ValueError('params and stats files do not match')

            df = pd.DataFrame(ptab[param_list_]) # create a new data frame
            df['metric']    = stab[self.metric]
            df['avgtime']   = ptab.file.str[-27:-23]
            df['sys_noise'] = pd.to_numeric(ptab.file.str[-22:-20])*0.01
            df['model']     = model
            df['id']        = ptab.file.str[-16:-11]
            df['pfile']     = ptab['file']
            df['sfile']     = stab['file']

            df_list.append(df)

        self.table_all = pd.concat(df_list)

    def filter_table(self):
        """Make a DataFrame containing the top set parameters.
        
        This method finds the intersection of parameters which produce
        reconstructions with metrics above the threshold.
  
        """
        
        print('\n---> Filtering table')

        tbs = []
        models = []
        for model in tqdm(self.models):
            threshold = self.model_thres[model]
            conds = 'metric > @threshold and model == @model'
            tb_ = self.table_all.query(conds)
            print('Model "{}" has {} matches'.format(model, len(tb_)))
            if len(tb_) > 0:
                tbs.append(tb_)
                models.append(model)

        tbs_merge = [tbs[0]]
        for i in range(len(tbs)-1):
            tb_new = pd.merge(tbs_merge[i], tbs[i+1], how='inner',
                              on=self.param_list,
                              suffixes=('_'+models[i],
                                        '_'+models[i+1]))
            tbs_merge.append(tb_new)

        cols = ['metric', 'model', 'id']
        tb_final = tbs_merge[-1].rename(columns={col: col+'_'+models[-1]
                                        for col in cols})
        self.table_top = tb_final
        self.table_top.to_csv(self.outpath+'table_top.txt', sep=' ')
        #with open(self.outpath+'table_top.md', 'w') as tb:
        #    tb.write(self.table_top.to_markdown())

    def run(self):
        """Run the class.
  
        """
        self.load_files()
        self.make_table()
        self.filter_table()

        print('\n---> Done!\n')
