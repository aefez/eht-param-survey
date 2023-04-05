# paramsurvey.py
# parameter survey class for Sgr A* static and dynamical imaging surveys
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

import os
import sys
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import glob
import h5py
import csv
import contextlib

import preimcal # the new sgra pre-processing pipeline

class HiddenPrints:
    """Suppresses printing from the loop function
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

with HiddenPrints():
    import ehtim as eh
    import ehtim.imaging.dynamical_imaging as di
    from ehtim.imaging import starwarps as sw
    import ehtim.scattering as so

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

class ParameterSurvey(object):
    """
    Defines & executes parameter survey 
    """

    def __init__(self, paramset, ehproc):

        for param in paramset:
            setattr(self, param, paramset[param])
        self.paramset = {param: paramset[param] for param in paramset 
                         if param not in ['inpath', 'outpath', 'save_stats',
                                          'save_uvfits', 'save_imgsums',
                                          'run_static', 'run_di', 'run_sw']}

        # defaults
        # TODO add more?
        if not hasattr(self, 'top_path'): self.top_path=False
        if not hasattr(self, 'blur_top_path'): self.blur_top_path=0.

        if not hasattr(self, 'save_uvfits'): self.save_uvfits=False
        if not hasattr(self, 'save_imgsums'): self.save_imgsums=False
        if not hasattr(self, 'save_stats'): self.save_stats=False

        if not hasattr(self, 'run_static'): self.run_static=False
        if not hasattr(self, 'run_di'): self.run_di=False
        if not hasattr(self, 'run_sw'): self.run_sw=False

        if not hasattr(self, 'ttype'): self.ttype='direct'
        if not hasattr(self, 'solint'): self.solint=0
        if not hasattr(self, 'gaintol'): self.gaintol=(0.2,0.2)
        if not hasattr(self, 'epsilon_tv'): self.epsilon_tv=1.e-10
        if not hasattr(self, 'stop'): self.stop=1.e-6
        
        # extend outpath
        if self.subdir: self.outpath += self.model + '/'
        
        # make outdir
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)

        #  other attributes
        self.ehproc = ehproc
        self.obsfile = self.inpath + self.model + '.uvfits'
        self.fov = self.fov * eh.RADPERUAS
        self.fwhm = self.fwhm * eh.RADPERUAS
        self.xmaj = self.xmaj * eh.RADPERUAS
        self.xmin = self.xmin * eh.RADPERUAS
        self.xpa = self.xpa * np.pi/180.0
        self.zbl = self.zbl_frac * self.zbl_tot
        self.LZgauss_flux = self.LZgauss_flux_frac * self.zbl_tot
        
        if self.model.split('_')[2] in ['double', 'elliptical', 'ring+hs', 'simple']:
            name = '_'.join(self.model.split('_')[2:4])
        else: name = self.model.split('_')[2]
        if name == 'SGRA': name = self.model.split('_')[0]
        if self.merge: band = 'lo+hi'
        else: band = 'lo'
        if self.deblurr: sct = 'dsct'
        else: sct = 'sct'
        self.outfile = '{}_{}_{}_{}_{}'.format(name, self.epoch, band, self.id, sct)

        self.res = None
        self.initimage = None

        self.obs_nopsd = None
        self.obs_psd = None

        self.out_static = None
        self.obs_static_nopsd = None
        self.obs_static_psd = None

        self.movie_sw= None
        self.obs_movie_sw = None

        self.movie_di = None
        self.obs_movie_di = None
        
        self.psd_noise_false = False
        
        self.gaintol = (float(self.gaintol.split('_')[0]),
                        float(self.gaintol.split('_')[1]))
        
    def save_params(self):
        """
        Save the parameter combination to a text file
        """
        df = pd.DataFrame.from_dict(self.paramset, orient='index',
                                    columns=[self.outfile])

        df.to_csv(self.outpath+self.outfile+'_params.txt', sep=' ',
                  header=False)

    def load_data(self, band='LO'):
        """
        Load and inititalize the data set (netcal/selfcal/sys noise)
        """

        if band == 'HI':
            if '_LO_' in self.obsfile: obsfile = self.obsfile.replace('_LO_', '_HI_')
            else: obsfile = self.obsfile.replace('_lo_', '_hi_')
        else: obsfile = self.obsfile
        
        # yaml puts None as 'None'
        if self.lcarr == 'None': self.lcarr = None
        
        # don't apply refractive noise floor if deblurr=False
        if not self.deblurr: self.ref_type = False

        print('\n Loading initial datafile')
        self.input_obs1 = eh.obsdata.load_uvfits(obsfile)

        ##############TODO: light curve normalization as survey parameter##########
        ############################################################################

        if self.run_sw:
            ############## Obs files for dynamic imaging -- no psd noise ###########
            inputset_nopsd = ['input_obs1', 'normalized','deblurred', 'lcarr',
                              'ehproc', 'LMTcal', 'LZgauss_size_uas', 'JCMTcal',
                              'tint', 'syserr', 'ref_type', 'ref_scale', 'deblurr',
                              'psd_noise_false', 'psd_a', 'psd_u0', 'psd_b', 'psd_c']

            print('\n Preparing {} band Data without Noise'.format(band))
            inputs = [getattr(self, param) for param in inputset_nopsd]
            obs_nopsd = preimcal.preim_pipeline(*inputs)
            obs_nopsd = obs_nopsd.switch_polrep('stokes')

        ############## Obs files for static imaging -- with psd noise ##############
        inputset_psd = ['input_obs1', 'normalized','deblurred',  'lcarr',
                        'ehproc', 'LMTcal', 'LZgauss_size_uas', 'JCMTcal',
                        'tint', 'syserr', 'ref_type', 'ref_scale', 'deblurr',
                        'psd_noise', 'psd_a', 'psd_u0', 'psd_b', 'psd_c']

        print('\n Preparing {} band Data with Noise'.format(band))
        inputs = [getattr(self, param) for param in inputset_psd]
        obs_psd = preimcal.preim_pipeline(*inputs)
        obs_psd = obs_psd.switch_polrep('stokes')
        
        if self.run_sw: return obs_nopsd, obs_psd
        else: return obs_psd
        
    def merge_data(self, obs_lo, obs_hi):
        """
        Merge LO and HI band data
        """
        # Add a slight offset to avoid mixed closure products
        obs_hi.data['time'] += 0.00002718
        
        print('\n Merging LO and HI band data')
        obs_lohi = obs_lo.copy()
        obs_lohi.data = np.concatenate([obs_lo.data, obs_hi.data])
        
        return obs_lohi

    def load_groundtruth(self):
        """
        Load and inititalize the ground truth movie
        """
        try:    
            # TODO: if deblurred do nxcorr with noscattering groundtruth and
            #       if nodeblurred with scattering groundtruth?
            if self.deblurr: name = 'noscattering'
            else: name = 'scattering'

            # TODO: can we make the search for the hdf5 more generic?
            # TODO: can we just load average image?
            truth_path = glob.glob(self.inpath+'_'.join(self.model.split('_')[0:3])
                                   +'*_'+name+'.fits')
            if len(truth_path) > 0: truth_path = truth_path[0]
            else: truth_path = self.obsfile[:-35] + name + '.fits'
#            self.truth_mov = eh.movie.load_hdf5(truth_path)
#            self.truth_avg = self.truth_mov.avg_frame()
            self.truth_avg = eh.image.load_fits(truth_path)
            self.truth_avg = self.truth_avg.regrid_image(self.fov, self.npixels)

        except (FileNotFoundError, OSError, IOError) as error:
            return -1

    def init_image(self):
        """
        Set up initial & prior image
        """
        self.res = self.obs_psd.res()
        emptyprior = eh.image.make_square(self.obs_psd, self.npixels, self.fov)
        
        if self.priortype == 'gauss':
            gaussprior = emptyprior.add_gauss(self.zbl, (self.fwhm, self.fwhm,
                                                         0, 0, 0))
            self.initimg = gaussprior.copy()
        elif self.priortype == 'disk':
            tophat = emptyprior.add_tophat(self.zbl, self.fwhm/2.0)
            tophat = tophat.blur_circ(self.res)
            self.initimg = tophat.copy()
        elif self.priortype == 'ring':
            ringim = emptyprior.add_ring_m1(self.zbl, 0, 26*eh.RADPERUAS,
                                            0, 10*eh.RADPERUAS)
            ringim = ringim.blur_circ(self.res)
            self.initimg = ringim.copy()
        elif self.priortype == 'asringim_left':
            asringim = emptyprior.add_ring_m1(self.zbl, 0.999, 26*eh.RADPERUAS,
                                              np.pi/3., 10*eh.RADPERUAS)
            asringim = asringim.blur_circ(self.res)
            self.initimg = asringim.copy()
        elif self.priortype == 'asringim_right':
            asringim2 = emptyprior.add_ring_m1(self.zbl, 0.999,
                                               26*eh.RADPERUAS, np.pi*2./3.,
                                               10*eh.RADPERUAS)
            asringim2 = asringim2.blur_circ(self.res)
            self.initimg = asringim2.copy()
            
    def opti_static(self):
        """
        Run static imaging on the sgra data
        """

        # new noise model (broken power law)
        obs_psd = self.obs_psd.copy()
        
        if self.run_sw:
            # data without psd noise
            obs_nopsd = self.obs_nopsd.copy()
                
        # check if we can load an already saved image
        # only load from top_path if static imaging is not run
        if self.top_path and not self.run_static: 
            load_path = self.top_path
        else:
            load_path = self.outpath + self.outfile + '.fits'

        try:
            out_static = eh.image.load_fits(load_path)
            print('\n   STATIC IMAGE LOAD PATH: ', load_path)
            print("   FOUND STATIC IMAGE FITS -- SKIPPING IMAGING!")

            if(out_static.rf != obs_nopsd.rf):
                out_static.rf = obs_nopsd.rf
            if(out_static.ra != obs_nopsd.ra or out_static.dec != obs_nopsd.dec):
                out_static.ra = obs_nopsd.ra
                out_static.dec = obs_nopsd.dec

            # regrid to correct resolution and pixel number
            out_static = out_static.regrid_image(self.fov, self.npixels)

        except:
            if not self.run_static: 
                print("run_static=False, and can't find static output!")
                return (None,None,None)

            print("STATIC IMAGING")

            # specify  data terms
            data_term = {}
            if hasattr(self,'vis') and self.vis != 0.:
                data_term['vis'] = self.vis
            if hasattr(self,'amp') and self.amp != 0.:
                data_term['amp'] = self.amp
            if hasattr(self, 'diag_closure') and self.diag_closure is True:
                if hasattr(self,'logcamp_diag') and self.logcamp_diag != 0.:
                    data_term['logcamp_diag'] = self.logcamp
                if hasattr(self,'cphase_diag') and self.cphase_diag != 0.:
                    data_term['cphase_diag'] = self.cphase
            else:
                if hasattr(self,'logcamp') and self.logcamp != 0.:
                    data_term['logcamp'] = self.logcamp
                if hasattr(self,'cphase') and self.cphase != 0.:
                    data_term['cphase'] = self.cphase

            # specify regularizer terms
            reg_term = {}
            if hasattr(self,'simple') and self.simple != 0.:
                reg_term['simple'] = self.simple
            if hasattr(self,'tv2') and self.tv2 != 0.:
                reg_term['tv2'] = self.tv2
            if hasattr(self,'tv') and self.tv != 0.:
                reg_term['tv'] = self.tv
            if hasattr(self,'l1') and self.l1 != 0.:
                reg_term['l1'] = self.l1
            if hasattr(self,'flux') and self.flux != 0.:
                reg_term['flux'] = self.flux
            if hasattr(self,'rgauss') and self.rgauss != 0.:
                reg_term['rgauss'] = self.rgauss
            
            # set up imager and convergence function
            imgr = eh.imager.Imager(obs_psd, self.initimg, prior_im=self.initimg,
                                    flux=self.zbl, 
                                    data_term=data_term, reg_term=reg_term,
                                    maxit=self.maxit, norm_reg=True,
                                    epsilon_tv=self.epsilon_tv,
                                    ttype=self.ttype,
                                    major=self.xmaj, 
                                    minor=self.xmin, 
                                    PA=self.xpa,
                                    stop=self.stop)
            
            # run imager
            def converge(major=self.niter_static, blur_frac=self.blurfrac):
                for repeat in range(major):
                    imgr.make_image_I(show_updates=False)
                    init = imgr.out_last().blur_circ(blur_frac*self.res)
                    imgr.init_next = init

            converge(blur_frac=self.blurfrac)
            out_static = imgr.out_last()

            # save
            out_static.save_fits(self.outpath + self.outfile + '.fits')

        # RESCALE output to total flux #TODO make an option
        if self.rescale_static:
            print('   RESCALING TOTAL FLUX')
            out_static.imvec *= self.zbl / out_static.total_flux()

        # selfcal
        if self.selfcal:
            if self.run_sw:
                obs_static_nopsd = eh.selfcal(obs_nopsd, out_static, method='both',
                                              ttype=self.ttype, solution_interval=self.solint,
                                              processes=self.ehproc)
                obs_static_psd = obs_psd.copy()

            if self.run_static:
                obs_static_psd = eh.selfcal(obs_psd, out_static, method='both', ttype=self.ttype,
                                            solution_interval=self.solint, processes=self.ehproc)
        else: 
            if self.run_sw: obs_static_nopsd = obs_nopsd.copy()
            obs_static_psd = obs_psd.copy()

        # save output
        if self.save_uvfits:
            if self.run_sw:
                obs_static_nopsd.save_uvfits(self.outpath + self.outfile +
                                             '_static_nopsd.uvfits')
            if self.run_static:
                obs_static_psd.save_uvfits(self.outpath + self.outfile + '.uvfits')

        if self.run_sw: return out_static, obs_static_nopsd, obs_static_psd
        else: return out_static, obs_static_psd
        
    def opti_starwarps(self, obs_static, out_static):
        """
        Set up and run starwarps
        """

        try: # try to load existing starwarps movie
            movie = eh.movie.load_hdf5(self.outpath + self.outfile + '_sw.hdf5')
            print("   FOUND STARWARPS MOVIE -- SKIPPING IMAGING!")

        except:
            if not self.run_sw: 
                print("run_sw=False, and can't find sw output!")
                return (None,None)

            # initalize list
            imCov, meanImg, initImg = [], [], []
            
            # blur the  prior/initial image
            if self.blur_sw_prior>0.: 
                out_static = out_static.blur_circ(self.blur_sw_prior*eh.RADPERUAS)
            
            # set the mean and initial image
            meanImg.append(out_static) 
            initImg.append(out_static)
        
            # set covariance
            imCov.append(sw.gaussImgCovariance_2(meanImg[0],
                                                 powerDropoff=self.powerDropoff,
                                                 frac=self.covfrac))
            
            # make the covariance matrix that says how much variation there
            # should be between frames in time
            noiseCov_img = np.eye(self.npixels**2) * self.variance_img_diff

            # initialize the flowbasis and get the initTheta which says how to
            # specify no motion for the specified flow basis
            init_x, init_y, flowbasis_x, flowbasis_y, initTheta = \
                sw.affineMotionBasis_noTranslation(meanImg[0])
            
            # split observation into scans
            # TODO: multiple types of obs splitting?
            obs_list = obs_static.split_obs(t_gather=self.tint)

            # set the data term
            data_term = {}
            if hasattr(self,'sw_amp') and self.sw_amp != 0.:
                data_term['amp'] = self.sw_amp
            if hasattr(self,'sw_logcamp') and self.sw_logcamp != 0.:
                data_term['logcamp'] = self.sw_logcamp
            if hasattr(self,'sw_cphase') and self.sw_cphase != 0.:
                data_term['cphase'] = self.sw_cphase
            if hasattr(self,'sw_bs') and self.sw_bs != 0.:
                data_term['bs'] = self.sw_bs
            if hasattr(self,'sw_vis') and self.sw_vis != 0.:
                data_term['vis'] = self.sw_vis
            if hasattr(self,'sw_flux') and self.sw_flux != 0.:
                data_term['flux'] = self.sw_flux

            # iterate after self-calibrating the data
            for iter in range(0, self.niter_sw):
                frames, expVal_t_t, expVal_tm1_t, loglikelihood, apxImgs = \
                    sw.computeSuffStatistics(meanImg, imCov,
                                obs_list, noiseCov_img,
                                initTheta, init_x, init_y,
                                flowbasis_x, flowbasis_y, initTheta,
                                method=self.warp_method,
                                measurement=data_term,
                                init_images=initImg, 
                                lightcurve = self.flux_list,
                                interiorPriors=self.interiorPriors,
                                numLinIters=self.numLinIters,
                                compute_expVal_tm1_t=False)
            
                # asign the time corresponding to each frame
                for i in range(len(frames)):
                    frames[i].time = obs_list[i].data['time'][0]
                    frames[i].mjd  = obs_list[i].mjd
            
                movie = eh.movie.merge_im_list(frames)
                movie.reset_interp(interp='linear', bounds_error=False)
                if self.niter_sw > 1:
                    obs_movie = eh.selfcal(obs_static, movie, method='both', ttype=self.ttype,
                                           solution_interval=self.solint, gain_tol=self.gaintol, 
                                           processes=self.ehproc)
                else:
                    obs_movie = obs_static.copy()
            
                # set the obs for the next iteration of imaging with the new
                # self-calibrated data
                obs_list = di.split_obs(obs_movie)

            # save output
            movie.save_hdf5(self.outpath + self.outfile + '_sw.hdf5')

        obs_movie = obs_static.copy()

        if self.save_uvfits:
            obs_movie.save_uvfits(self.outpath + self.outfile + '_sw.uvfits')

        return movie, obs_movie
    
    def opti_dynamical(self, obs, movie_init, di_prior):
        import ehtim.imaging.dynamical_imaging as di

        try: # try to load existing di movie
            movie = eh.movie.load_hdf5(self.outpath + self.outfile + '_di.hdf5')
            print("   FOUND DI MOVIE -- SKIPPING IMAGING!")
        except:
            if not self.run_di: 
                print("run_di=False, and can't find di output!")
                return (None,None)

            # Deal with potentially negative values and nans in starwarps
            movie_init.frames = np.nan_to_num(movie_init.frames)
            movie_init.frames = movie_init.frames * (movie_init.frames > 0.0) + 1e-30

            # split observation into scans
            # TODO: multiple types of obs splitting? 
            obs.add_scans()
            obs_list = obs.split_obs() #scan_gather=True << causes a weird bug

            # parameters
            metric = self.metric_di
            maxit = self.maxit # TODO: same as in static imaging
            # Print the regularizer values for a given movie
            B_dt = np.zeros((di_prior.ydim,di_prior.xdim))
            print("Parameters for initial movie")
            print("Rdt: %4.8f"%di.Rdt(np.array([i.imvec for i in movie_init.im_list()]), 
                                      B_dt, metric))
            print("RdI: %4.8f"%di.RdI(np.array([i.imvec for i in movie_init.im_list()]), 
                                      metric))

            # Determine the dynamical imaging weights
            B_dt = np.zeros((di_prior.ydim,di_prior.xdim))
            alpha_dI = self.alpha_dI_rel/di.RdI(np.array([i.imvec for i in movie_init.im_list()]),
                                                metric)
            alpha_dt = self.alpha_dt_rel/di.Rdt(np.array([i.imvec for i in movie_init.im_list()]),
                                                B_dt, metric)

            # Run dynamical imaging
            # TODO -- more regularizers!
            if self.di_vis !=0:
                movie = di.dynamical_imaging(obs, movie_init.blur_circ(obs.res()/2, 5), di_prior, 
                                           flux_List=self.flux_list, alpha_flux=1e6, 
                                           R_dI={'alpha':alpha_dI, 'metric':metric},
                                           R_dt={'alpha':alpha_dt, 'metric':metric, 'sigma_dt':0.0},
                                           d1 = 'vis',     alpha_d1=self.di_vis, 
                                           d2 = 'cphase',  alpha_d2=self.di_cphase, 
                                           d3 = 'logcamp', alpha_d3=self.di_logcamp, 
                                           entropy1='tv2',    alpha_s1=self.di_tv2,  
                                           entropy2='simple', alpha_s2=self.di_simple, 
                                           #entropy2='rgauss',  alpha_s2=self.rgauss,
                                           #major=self.xmaj, minor=self.xmin, PA=self.xpa,
                                           update_interval=20, maxit = maxit)
            else:
                movie = di.dynamical_imaging(obs, movie_init.blur_circ(obs.res()/2, 5), di_prior, 
                                           flux_List=self.flux_list, alpha_flux=1e6, 
                                           R_dI={'alpha':alpha_dI, 'metric':metric},
                                           R_dt={'alpha':alpha_dt, 'metric':metric, 'sigma_dt':0.0},
                                           d1 = 'amp',     alpha_d1=self.di_amp, 
                                           d2 = 'cphase',  alpha_d2=self.di_cphase, 
                                           d3 = 'logcamp', alpha_d3=self.di_logcamp, 
                                           entropy1='tv2',    alpha_s1=self.di_tv2,  
                                           entropy2='simple', alpha_s2=self.di_simple, 
                                           #entropy2='rgauss',  alpha_s2=self.rgauss,
                                           #major=self.xmaj, minor=self.xmin, PA=self.xpa,
                                           update_interval=20, maxit = maxit)


            movie.reset_interp(interp='linear', bounds_error=False)

            # save output
            movie.save_hdf5(self.outpath + self.outfile + '_di.hdf5')

        if self.selfcal:
            obs_movie = eh.selfcal(obs, movie, method='both', ttype=self.ttype,
                                   solution_interval=self.solint, gain_tol=self.gaintol,
                                   processes=self.ehproc)
        else:
            obs_movie = obs.copy()

        if self.save_uvfits:
            obs_movie.save_uvfits(self.outpath + self.outfile + '_di.uvfits')

        return movie, obs_movie

    def save_image(self, out_static, obs_static, obs):
        """
        Save the output image
        """
        
        # display
        out_static.display(show=False, has_title=False,
                           export_pdf=self.outpath + self.outfile + '_static.png')

        # diagnostic pdf
        eh.imgsum(out_static, obs_static, obs, 
                  self.outpath + self.outfile + '_static.pdf',
                  title='imgsum', ttype=self.ttype,
                  processes=self.ehproc)

    def save_movie(self, movie, obs_movie, obs, name):
        """
        Save the output movie
        """
        # export an mp4
        movie.export_mp4(out=self.outpath+self.outfile+name+'.gif',
                         label_time=True, fps=10)

        # save diagnositic pdf
        eh.imgsum(movie, obs_movie, obs, self.outpath + self.outfile
                  + name + '.pdf', title='imgsum', ttype=self.ttype,
                  processes=self.ehproc)
    
    def save_statistics(self, stattype='static'):
        """
        Save summary statistics
        """

        if stattype=='static':
            #run static compraisons with noise
            obs = self.obs_static_psd  
            out = self.out_static
            out_avg = self.out_static
        elif stattype=='sw':
            obs = self.obs_movie_sw
            out = self.movie_sw
            out_avg = self.movie_sw.avg_frame()
        elif stattype=='di':
            obs = self.obs_movie_di
            out = self.movie_di
            out_avg = self.movie_di.avg_frame()
        else:
            raise Exception("stattype not recognized in  save_statistics!")

        # define metrics
        chisqs = ['vis', 'amp', 'cphase', 'logcamp', 'camp']
        index = ['chisq_' + chisq for chisq in chisqs]
        columns = ['obs']
        
        metric = [obs.chisq(out, dtype=chisq) for chisq in chisqs]

        # Try to add image comparisons if we can find an hdf5 file
        if hasattr(self, 'truth_avg'):   

            index.append('nxcorr')
            index.append('nrmse')
            index.append('rssd')

            nxcorr, nrmse, rssd = \
                self.truth_avg.compare_images(out_avg)[0]

            metric.append(nxcorr)
            metric.append(nrmse)
            metric.append(rssd)

        # Wrap up metrics and save        
        metric_dict = {ind: metr for ind, metr in zip(index, metric)}
        
        df = pd.DataFrame.from_dict(metric_dict, orient='index',
                                    columns=columns)

        df.to_csv(self.outpath+self.outfile + '_stats.txt', sep=' ')

    def setup(self):
        """initialize for imaging"""

        # Load data
        if self.run_sw:
            obs_nopsd_lo, obs_psd_lo = self.load_data(band='LO')
            if self.merge:
                obs_nopsd_hi, obs_psd_hi = self.load_data(band='HI')
            
        else:
            obs_psd_lo = self.load_data(band='LO')
            if self.merge:
                obs_psd_hi = self.load_data(band='HI')
            
        # Merge data
        if self.merge:
            if self.run_sw:
                self.obs_nopsd = self.merge_data(obs_nopsd_lo, obs_nopsd_hi)    
            self.obs_psd = self.merge_data(obs_psd_lo, obs_psd_hi)
            
        else:
            if self.run_sw:
                self.obs_nopsd = obs_nopsd_lo.copy()
            self.obs_psd = obs_psd_lo.copy()

        # Set up static imaging initialization and prior
        self.init_image()

        # Load the truth if you can
        self.load_groundtruth()

        # Get the lightcurve
        # TODO CREATE OPTION TO LOAD LIGHTCURVE FROM FILE

        self.flux_list = get_lightcurve(obs_psd_lo)


    def run_static_imaging(self):
        """
        Run static imaging parameter survey step
        """

        # Run static imaging
        if self.run_sw:
            self.out_static, self.obs_static_nopsd, self.obs_static_psd = self.opti_static()
        else:
            self.out_static, self.obs_static_psd = self.opti_static()

    def run_sw_imaging(self):
        """
        Run stawrwarps parameter survey step
        """

        # Run starwarps on scattered & deblurred
        if self.run_sw:
            self.movie_sw, self.obs_movie_sw = \
                self.opti_starwarps(self.obs_static_nopsd, self.out_static)

            # Add small brightness to mask negative values
            ep = np.median(self.movie_sw.frames)*1.e-8
            self.movie_sw.frames[self.movie_sw.frames<=ep] = ep

    def run_di_imaging(self):
        """
        Run dynamical imaging parameter survey
        """

        # Run dynamic imaging
        if self.run_di:
            self.movie_di, self.obs_movie_di = \
                self.opti_dynamical(self.obs_movie_sw, self.movie_sw, self.out_static)

    def save_figures(self, figtype='static'):
        """
        Save imgsums and mp4
        """
        
        if figtype=='static':
            self.save_image(self.out_static, self.obs_static_psd, self.obs_psd)

        elif figtype=='sw':
            try:
                self.save_movie(self.movie_sw, self.obs_movie_sw, self.obs_nopsd, '_sw')
            except:
                print("Warning! Couldn't save mp4 and imgsum for sw movies!")

        elif figtype=='di':
            try:
                self.save_movie(self.movie_di, self.obs_movie_di, self.obs_nopsd, '_di')
            except:
                print("Warning! Couldn't save mp4 and imgsum for di movies!")
        else:
            raise Exception("figtype not recognized in  save_figures!")

    def run(self):
        """Run the whole survey"""
        
        try:
            # Load pre-computed images and data
            self.out_static = eh.image.load_fits(self.outpath+self.outfile+'.fits')
            self.obs_static_psd = eh.obsdata.load_uvfits(self.outpath+self.outfile+'.uvfits')
            self.load_groundtruth()
            
            # TODO: blur reconstructed and ground-truth image with a common beam?
            
            # Compute and save statistics
            if self.save_stats and self.run_static:
                self.save_statistics(stattype='static')
                
        except:
            # Save parameter file
            self.save_params()

            # Setup 
            self.setup()
            
            # Static imaging
            print('\n    STATIC IMAGING')
            self.run_static_imaging() # merged load function from previous run into opti_static
            if self.save_stats and self.run_static:
                self.save_statistics(stattype='static')
            if self.save_imgsums and self.run_static:
                self.save_figures(figtype='static')

            # Starwarps
            print('\n    STARWARPS')
            self.run_sw_imaging()  # merged load function from previous run into opti_sw
            if self.save_stats and self.run_sw:
                self.save_statistics(stattype='sw')
            if self.save_imgsums and self.run_sw:
                self.save_figures(figtype='sw')

            # Dynamic imaging
            print('\n    DYNAMIC IMAGING')
            self.run_di_imaging() # merged load function from previous run into opti_di
            if self.save_stats and self.run_di:
                self.save_statistics(stattype='di')
            if self.save_imgsums and self.run_di:
                self.save_figures(figtype='di')


def get_lightcurve(obs):
    obs.add_scans()
    obs_split = obs.split_obs(scan_gather=False)

    # univariate spline interpolation
    alltimes = np.array([obs_split[j].data['time'][0] for j in range(len(obs_split))])
    with contextlib.redirect_stdout(None):
        allfluxes = np.array(
            [np.median(obs_split[j].flag_uvdist(uv_max=0.01e9).unpack('amp')['amp']) 
             for j in range(len(obs_split))])

    idxsort = np.argsort(alltimes)
    alltimes = alltimes[idxsort]
    allfluxes = allfluxes[idxsort]

    mask = np.isnan(allfluxes)
    maskedtimes = alltimes[~mask]
    maskedfluxes = allfluxes[~mask]

    spl = interp.UnivariateSpline(maskedtimes, maskedfluxes, ext=3)
    spl.set_smoothing_factor(1e-10)
    spl_times = alltimes
    spl_fluxes = spl(spl_times)
    flux_list = list(spl_fluxes)

    return flux_list

    
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#TODO better error catching missing files!
def make_results_hdf5(direc, outname):
    """
    Compiles all results into hdf5
    """

    stattypes_all = ['static','sw','di']

    # find all parameter files
    files = np.sort(glob.glob(direc + '/*_params.txt'))
    n = len(files)
    print("found %i parameter files in %s"%(n,direc))

    # create base hdf5
    oname = direc + '/' + outname + '.hdf5'
    if os.path.exists(oname):
        os.remove(oname)
    fout = h5py.File(oname,'w')

    # setup file name array
    dat0 = fout.create_dataset('files', (n,), h5py.string_dtype())

    # setup stats array using first file as template for keys 
    stattypes = []
    ddicts = []
    for stattype in stattypes_all:
        fname = os.path.basename(files[0])[:-11]
        fname_stats = direc + '/' + fname + '_' + stattype + '_stats.txt'
        if os.path.exists(fname_stats):
            stats = np.loadtxt(files[0],str,skiprows=1)
            keys_stats = stats[:,0]

            sname = stattype + '_stats'
            gp1 = fout.create_group(sname + '_dsct')
            dat1_dict = {key : gp1.create_dataset(key, (n,), 'f') for key in keys_stats}

            gp11 = fout.create_group(sname + '_sct')
            dat11_dict = {key : gp11.create_dataset(key, (n,), 'f') for key in keys_stats}

            stattypes.append(stattype)
            ddicts.append([dat1_dict,dat11_dict])

    # setup parameter array using first file as template for keys
    gp2 = fout.create_group('params')
    fname_params = files[0]
    fp = open(fname_params,'r')
    reader = csv.reader(fp, delimiter=' ')
    dat2_dict = {}
    keys_params = []
    for row in reader:
        key = row[0]
        val = row[1]
        if val == 'True' or val == 'False':
            dtype = bool
        else:
            try:
                val = float(val)
                dtype = 'f'
            except:
                dtype = h5py.string_dtype()
        dat2_dict[key] = gp2.create_dataset(key, (n,), dtype)
        keys_params.append(key)
    fp.close()

    # load  and  save for all files in survey
    for idx, f in enumerate(files):
    
        # save base file name 
        fname = os.path.basename(f)[:-11]
        dat0[idx] = fname

        # load parameters & save to hdf5
        fname_params = direc + '/' + fname + '_params.txt'
        fp = open(fname_params,'r')
        reader = csv.reader(fp, delimiter=' ')
        for row in reader:
            key = row[0]
            val = row[1]
            if val == 'True' or val == 'False':
                val = bool(val)
            elif val[0] == '[':
                val = val[1:-1]
            else:
                try: val = float(val)
                except: pass

            if key in keys_params:
                dat2_dict[key][idx] = val
        fp.close()

        # load stats and save to hdf5
        for kk, stattype in enumerate(stattypes):
            fname_stats = direc + '/' + fname + '_' + stattype + '_stats.txt'
            stats = np.loadtxt(fname_stats,str,skiprows=1)
            dat1_dict = ddicts[kk][0]
            dat11_dict = ddicts[kk][1]

            for d in stats:
                #print(d)
                key = d[0]
                val_dsct = float(d[1])
                val_sct = float(d[2])
                if key in keys_stats:
                    dat1_dict[key][idx] = val_dsct
                    dat11_dict[key][idx] = val_sct

    # close hdf5
    fout.close()
    return

