#!/usr/bin/env python
"""
Code related to the Sirenas survey.
"""
from __future__ import print_function

import os,sys
import logging
import copy
from collections import OrderedDict as odict

import pandas as pd
import numpy as np
import fitsio

from obztak.field import FieldArray, SISPI_DICT, SEP
from obztak.survey import Survey
from obztak.scheduler import Scheduler
from obztak.tactician import Tactician

from obztak.utils.projector import cel2gal, angsep
from obztak.utils import constants
from obztak.utils import fileio
from obztak.utils.constants import BANDS,SMASH_POLE,CCD_X,CCD_Y,STANDARDS,DECAM
from obztak.utils.constants import COLORS, CMAPS
from obztak.utils.date import datestring, setdefaults, nite2utc,utc2nite,datestr

NAME    = 'SIRENAS'
PROGRAM = NAME.lower()
PROPID  = '2019A-0305' # TODO: amend
PROPOSER = 'Soares-Santos'
BANDS = ["u",'g','r','i','z',"M4112", "M4376", "M4640", "M4904", "M5168"]  
TILINGS = [1,2,3,4,5,6,7,8,9,10]
DONE = -1

# TODO: Need to have a discussion about what kinds of mini-surveys we would want, and what the t_eff and FWHM limits should be
TEFF_MIN_BEAR = pd.DataFrame(dict(FILTER=BANDS[0:5],TEFF=[0.4,0.4,0.4,0.4,0.4])) # ugriz, teff threshold at 0.4
TEFF_MIN_O4 = pd.DataFrame(dict(FILTER=BANDS[0:5],TEFF=[0.4,0.4,0.4,0.4,0.4])) # ugriz, teff threshold at 0.4
TEFF_MIN_O5 = pd.DataFrame(dict(FILTER=BANDS[-5:],TEFF=[0.4,0.4,0.4,0.4,0.4])) # medium bands, teff threshold at 0.4

FWHM_BEAR = 1 # Arcsec
FWHM_O4 = 1
FWHM_O5 = 1 


class SirenasSurvey(Survey):
    """ Survey sublcass for SIRENAS. """

    """ Instantiate the relevant nights and half nights"""

    nights_2023B = [
        ['2023/08/23','first'], #DDT
        ['2023/09/01','full'],
        ['2023/10/24','full'],
        ['2023/11/01','second'],
        ['2023/11/02','second'],
        ['2023/11/03','second'],
        ['2023/12/23','full'],
        ['2023/12/29','full'],
        ['2024/01/01','second'],
        ['2024/01/19','first'],
        ['2024/01/20','first'],
        ['2024/01/28','full'],

    ]

    extra_nights = []

    nights = nights_2023B \
             + extra_nights

    def prepare_fields(self, infile=None, outfile=None, plot=True, **kwargs):
        """ Create the list of fields to be targeted by this survey.

        Parameters:
        -----------
        infile : File containing all possible field locations.
        outfile: Output file of selected fields
        plot   : Create an output plot of selected fields.

        Returns:
        --------
        fields : A FieldArray of the selected fields.
        """

        if infile is None:
            infile = fileio.get_datafile('decam-tiles-bliss-v1.fits.gz')
        logging.info("Reading tiles from: %s"%os.path.basename(infile))
        data = fitsio.read(infile)

        deep_fields  = self.create_deep_fields(data)
        mc_fields    = self.create_mc_fields(data)
        wide_fields  = self.create_wide_fields(data)
        extra_fields = self.create_extra_fields(data)

        # Overlap fields
        sel  = (~np.in1d(extra_fields.unique_id,wide_fields.unique_id))
        sel &= (~np.in1d(extra_fields.unique_id,mc_fields.unique_id))
        ## Mark overlap fields as DONE
        #extra_fields['PRIORITY'][~sel] = DONE
        # Remove overlap fields
        extra_fields = extra_fields[sel]

        fields = wide_fields + mc_fields + deep_fields + extra_fields

        logging.info("Masking bright stars...")
        mask = self.bright_stars(fields['RA'],fields['DEC'])
        fields['PRIORITY'][mask] = DONE

        # Can't go lower
        mask = fields['DEC'] < -89.0
        fields['PRIORITY'][mask] = DONE


	""" TODO: Better understand this, and see if we have any exclusion zones for Sirenas """
        # Exclusion
        exclude  = [#pole
            '399-01-g','399-01-r','399-01-i',
            '399-02-g','399-02-r','399-02-i',
            '399-04-g','399-04-r','399-04-i',
            '436-01-i','437-01-i',
            '417-03-g','417-03-r','417-03-i',
        ]
        exclude += [ # Orion Nebula
            '5696-01-i','5696-02-i','5696-03-i','5696-04-i',
            '5760-01-i','5760-02-i','5760-03-i','5760-04-i',
            '5761-01-i','5761-02-i','5761-03-i','5761-04-i',
            '5780-01-i','5780-02-i','5780-03-i','5780-04-i',
            '5781-01-i','5781-02-i','5781-03-i','5781-04-i',
            '5782-01-i','5782-02-i','5782-03-i','5782-04-i',
            '5783-01-i','5783-02-i','5783-03-i','5783-04-i',
            '5798-01-i','5798-02-i','5798-03-i','5798-04-i',
            '5799-01-i','5799-02-i','5799-03-i','5799-04-i',
        ]
        exclude += [# flame nebula
            '5716-01-g','5716-01-i'
        ]
        exclude += [# rho ophiuchi nebula
        ]
        exclude += [ # MC poles
            '14110-01-g','14110-02-g','14110-03-g','14110-04-g',
            '14110-01-r','14110-02-r','14110-03-r','14110-04-r',
            '14110-01-i','14110-02-i','14110-03-i','14110-04-i',
            '15464-01-i','15464-02-i','15464-03-i','15464-04-i',
            '15465-01-i','15465-02-i','15465-03-i','15465-04-i',
            '15484-03-i',
        ]
        exclude += [ # Other
            '7246-01-r','7264-01-i',
            '7246-01-i','7246-02-i','7246-02-i',
            '12253-03-i','14238-03-i',
            '14241-03-i','14255-03-i','14256-03-i','14257-03-i',
            '14258-03-i','15465-04-g',
        ]
        fields['PRIORITY'][np.in1d(fields.unique_id,exclude)] = DONE

        if plot:
            import pylab as plt
            import skymap.survey
            plt.ion()

            sel = [fields['PRIORITY'] > 0]

            plt.figure()
            smap = skymap.survey.MaglitesSkymap()
            smap.draw_fields(fields[sel],alpha=0.3,edgecolor='none')
            smap.draw_des(color='r')
            smap.draw_milky_way()
            smap.draw_smash()

            plt.figure()
            smap = skymap.survey.SurveyMcBryde()
            smap.draw_fields(fields[sel],alpha=0.3,edgecolor='none')
            smap.draw_des(color='r')
            smap.draw_milky_way()
            smap.draw_smash()

            if outfile:
                plt.savefig(os.path.splitext(outfile)[0]+'.png',bbox_inches='tight')
            if not sys.flags.interactive:
                plt.show(block=True)

        if outfile:
            print("Writing %s..."%outfile)
            fields.write(outfile)

        return fields

    @classmethod
    def update_covered_fields(cls, fields):
        """ Update the priority of covered fields. """
        fields = copy.deepcopy(fields)
        frac, depth = cls.covered(fields)
        done = (fields['PRIORITY'] == DONE)
        print("Found %i exposures already done."%done.sum())

        bear= np.char.endswith(fields['PROGRAM'],'-bear')
        teff_min_bear = pd.DataFrame(fields).merge(TEFF_MIN_BEAR,on='FILTER',how='left').to_records()['TEFF']
        covered_bear = depth > teff_min_bear*fields['TILING']*fields['EXPTIME']
        done_bear = bear & covered_bear
        print('Found %i BEAR exposures newly done.'%(done_bear & ~done).sum())

        O4 = np.char.endswith(fields['PROGRAM'],'-O4')
        teff_min_O4 = pd.DataFrame(fields).merge(TEFF_MIN_O4,on='FILTER',how='left').to_records()['TEFF']
        covered_O4 = depth > teff_min_O4*fields['TILING']*fields['EXPTIME']
        done_O4 = O4 & covered_O4
        print('Found %i O4 exposures newly done.'%(done_O4 & ~done).sum())

        O5 = np.char.endswith(fields['PROGRAM'],'-O5')
        teff_min_O5 = pd.DataFrame(fields).merge(TEFF_MIN_O5,on='FILTER',how='left').to_records()['TEFF']
        covered_O5 = depth > teff_min_O5*fields['TILING']*fields['EXPTIME']
        done_O5 = O5 & covered_O5
        print('Found %i O5 exposures newly done.'%(done_O5 & ~done).sum())

        fields['PRIORITY'][done_bear & ~done] = DONE
        fields['PRIORITY'][done_O4 & ~done] = DONE
        fields['PRIORITY'][done_O5 & ~done] = DONE

        return fields

    def create_bear_fields(self, data, plot=False):
        """ Create the bear field observations """
        logging.info("Creating BEAR fields...")
        BANDS = ['g','i','z','r','u']
        EXPTIME = [90,90,90,90,90]
        TILINGS = [4,4,4,4,4]
        TEFF_MIN = TEFF_MIN_BEAR

        nhexes = len(np.unique(data['TILEID']))
        nbands = len(BANDS)

        nfields = len(data)*nbands

        logging.info("  Number of hexes: %d"%nhexes)
        logging.info("  Filters: %s"%BANDS)
        logging.info("  Exposure time: %s"%EXPTIME)
        logging.info("  Tilings: %s"%TILINGS)

        fields = FieldArray(nfields)
        fields['PROGRAM'] = PROGRAM+'-bear'
        fields['HEX'] = np.repeat(data['TILEID'],nbands)
        fields['TILING'] = np.repeat(data['PASS'],nbands)
        fields['RA'] = np.repeat(data['RA'],nbands)
        fields['DEC'] = np.repeat(data['DEC'],nbands)

        fields['FILTER'] = np.tile(BANDS,len(data))
        fields['EXPTIME'] = np.tile(EXPTIME,len(data))
        fields['PRIORITY'] = fields['TILING']

        sel = self.footprintBEAR(fields['RA'],fields['DEC']) # Bear footprint?
        sel &= (~self.footprintMilkyWay(fields['RA'],fields['DEC'])) # Avoiding milky way
        sel &= (~self.footprintDES(fields['RA'],fields['DEC'])) # Avoiding DES fields
        #sel &= (~self.footprintSMASH(fields['RA'],fields['DEC'],angsep=0.75*DECAM))
        #sel &= (~self.footprintmc(fields['RA'],fields['DEC']))
        # Avoid DEEP fields? yes.
        sel &= (~self.footprintDEEP(fields['RA'],fields['DEC']))

        fields = fields[sel]

        # Covered fields
        frac, depth = self.covered(fields)
        teffmin = pd.DataFrame(fields).merge(TEFF_MIN,on='FILTER',how='left').to_records()['TEFF']
        fields['PRIORITY'][depth > teffmin*fields['TILING']*fields['EXPTIME']] = DONE

        if plot: self.plot_depth(fields,depth,'sirenas-bear-%s-gt%i.png')

        logging.info("Number of target fields: %d"%len(fields))

        outfile = 'sirenas-bear-fields.fits.fz'
        logging.info("Writing %s..."%outfile)
        fields.write(outfile,clobber=True)

        return fields

""" TODO: Decide fields for bear survey """

""" TODO: Decide on how to manage O4 and O5 fields """
    
""" TODO: Revise bear footprint selection """
    @staticmethod
    def footprintBEAR(ra,dec):
        """ Select extra exposures for BEAR survey """
        ra,dec = np.copy(ra), np.copy(dec)

        # Between S82 and SPT
        sel1  = (dec >= -40) & (dec <= -15)
        sel1 &= ((ra >= 280) & (ra <= 360)) | (ra < 10)
        # Close to the Galactic plane
        sel2 = (dec >= -20) & (dec <= -5)
        sel2 &= (ra >= 240) & (ra <= 280)

        # Either of the two
        sel = sel1 | sel2

        return sel

""" TODO: Revise O4 footprint selection """
    @staticmethod
    def footprintEXTRA(ra,dec):
        """ Selecting wide-field exposures plane """
        ra,dec = np.copy(ra), np.copy(dec)
        sel = (dec < 30)
        return sel

""" TODO: Revise O5 footprint selection """
    @staticmethod
    def footprintEXTRA(ra,dec):
        """ Selecting wide-field exposures plane """
        ra,dec = np.copy(ra), np.copy(dec)
        sel = (dec < 30)
        return sel



    @staticmethod
    def bright_stars(ra,dec):
        """ Load bright star list """
        ra,dec = np.copy(ra), np.copy(dec)
        sel = np.zeros(len(ra),dtype=bool)
        #filename = fileio.get_datafile('famous-bright-stars.csv')
        filename = fileio.get_datafile('bsc5p-bright-stars.csv')
        targets = fileio.read_csv(filename).to_records()
        for t in targets:
            sel |= (angsep(t['ra'],t['dec'],ra,dec) < t['radius'])
        return sel


    @staticmethod
    def covered(fields, percent=85., dirname=None, basename=None):
        """
        Determine which fields haven't been previously covered by DECam

        Parameters:
        -----------
        fields : field information
        percent: fraction of the field that is covered

        Returns:
        --------
        frac, depth : selection of fields and coverage fraction
        """
        import healpy as hp
        # These maps are SUM(teff * exptime)
        #if not dirname: dirname = '/Users/kadrlica/delve/observing/v2/maps/20230204'
        #if not dirname: dirname = '/Users/kadrlica/delve/observing/v2/maps/20230824'
        #if not dirname: dirname = '/Users/kadrlica/delve/observing/v2/maps/20231013'
        if not dirname: dirname = '/Users/kadrlica/delve/observing/v2/maps/20231226'
        if not basename: basename = 'decam_sum_expmap_%s_n1024.fits.gz'

        logging.info("Loading maps from: %s"%dirname)

        sel = np.ones(len(fields),dtype=bool)
        frac  = np.zeros(len(fields),dtype=float)
        depth = np.zeros(len(fields),dtype=float)
        ra,dec,band=fields['RA'],fields['DEC'],fields['FILTER']

        for b in np.unique(band):
            idx = (band==b)
            filename = os.path.join(dirname,basename%b)
            logging.info("Reading %s..."%os.path.basename(filename))
            skymap = hp.read_map(filename,verbose=False)

            nside = hp.get_nside(skymap)
            vec = hp.ang2vec(np.radians(90.-dec[idx]),np.radians(ra[idx]))

            f,d = [],[]
            for i,v in enumerate(vec):
                print('\r%s/%s'%(i+1,len(vec)),end="")
                sys.stdout.flush()
                pix = hp.query_disc(nside,v,np.radians(constants.DECAM))

                # Find effective exposure time that is achieved over
                # the specified fraction of the exposure e.g., where
                # 75% of the pixels have a larger SUM(teff * exptime)
                d.append(np.percentile(skymap[pix],100-percent))

                # Find the fraction covered at the achieved depth
                # (I don't think this adds anything)
                f.append((skymap[pix] >= d[-1]).sum()/float(len(pix)))

            print()
            frac[idx] = np.array(f)
            depth[idx] = np.array(d)

        return frac,depth

    def plot_depth(self, fields, depth, outbase, proj='mcbryde', **kwargs):
        import skymap, skymap.survey
        import pylab as plt
        bands = np.unique(fields['FILTER'])
        ra,dec = fields['RA'],fields['DEC']
        for b in bands:
            sel = fields['FILTER']==b
            for d in np.unique(fields[sel]['EXPTIME']*fields[sel]['TILING'])[:-1]:
                plt.figure()
                if proj == 'mcbryde': smap = skymap.McBrydeSkymap()
                elif proj == 'maglites': smap = skymap.survey.MaglitesSkymap()
                smap.scatter(*smap(ra[sel],dec[sel]),c=depth[sel],vmax=d,edgecolor='none',s=3)
                smap.draw_lmc(fc='none')
                smap.draw_smc(fc='none')
                plt.colorbar()
                plt.savefig(outbase%(b,d),bbox_inches='tight')
                plt.close()

""" Start TODO again here"""
class DelveFieldArray(FieldArray):
    PROGRAM  = PROGRAM
    PROPID   = PROPID
    PROPOSER = PROPOSER

    SISPI_DICT = copy.deepcopy(SISPI_DICT)
    SISPI_DICT["program"] = PROGRAM
    SISPI_DICT["propid"] = PROPID
    SISPI_DICT["proposer"] = PROPOSER

    OBJECT_FMT = NAME.upper() + ' field'+SEP+' %s'
    SEQID_FMT  = NAME.upper() + ' scheduled'+SEP+' %(DATE)s'
    BANDS = BANDS

    @classmethod
    def query(cls, **kwargs):
        """ Generate the database query.

        Parameters:
        -----------
        kwargs : Keyword arguments to fill the query.

        Returns:
        --------
        query  : The query string.
        """
        defaults = dict(propid=cls.SISPI_DICT['propid'], limit='',
                        object_fmt = cls.OBJECT_FMT%'')
        #date_column = 'date' or 'to_timestamp(utc_beg)'
        defaults['date_column'] = 'date'
        kwargs = setdefaults(kwargs,copy.deepcopy(defaults))

        query ="""
        SELECT object, seqid, seqnum, telra as RA, teldec as dec,
        expTime, filter,
        to_char(%(date_column)s, 'YYYY/MM/DD HH24:MI:SS.MS') AS DATE,
        COALESCE(airmass,-1) as AIRMASS, COALESCE(moonangl,-1) as MOONANGLE,
        COALESCE(ha, -1) as HOURANGLE, COALESCE(slewangl,-1) as SLEW, PROGRAM
        --2019B-1014: Felipe Olivares
        --2022B-780972: Ferguson
        --2023A-343956: Ferguson 
        FROM exposure
        WHERE propid in ('%(propid)s','2019A-0305','2019B-1014','2022B-780972','2023A-343956')
        and exptime > 89
        and discard = False and delivered = True and flavor = 'object'
        and object LIKE '%(object_fmt)s%%'
        and object NOT LIKE '%%Peg4%%'
        and object NOT LIKE '%%LMi%%'
        and object NOT LIKE '%%dr2_%%'
        and object NOT LIKE '%%dr3_%%'
        and object NOT LIKE '%%p6next_%%'
        and id NOT IN (967215)
        -- Disk corruption
        and id NOT IN (1029209, 1029212, 1029213, 1029214)
        -- Bad build
        and id NOT IN (1221800, 1221801)
        -- and id NOT IN (860597, 860598, 860599, 860600, 860601, 860602)
        -- Mirror compressed air on 20201025
        -- and id NOT BETWEEN 948781 and 948795
        -- Cloudy nite with lots of qc_teff = nan
        and NOT (id BETWEEN 1025565 and 1025876 and qc_teff is null)
        -- DEEP on 20230712 with poor seeing
        and NOT (id BETWEEN 1220911 and 1220934 and qc_fwhm > 1.2)
        and id NOT IN (1222202, 1222204, 1222205)
        and (
             (COALESCE(qc_teff,-1) NOT BETWEEN 0 and 0.3
             AND COALESCE(qc_fwhm,1) BETWEEN 0.5 and 1.5)
             OR %(date_column)s  > (now() - interval '14 hours')
        )
        ORDER BY %(date_column)s %(limit)s
        """%kwargs
        return query


class DelveScheduler(Scheduler):
    _defaults = odict(list(Scheduler._defaults.items()) + [
        ('tactician','coverage'),
        ('windows',fileio.get_datafile("delve-windows.csv.gz")),
        ('targets',fileio.get_datafile("delve-target-fields.csv.gz")),
    ])

    FieldType = DelveFieldArray


class DelveTactician(Tactician):
    CONDITIONS = odict([
        (None,       [1.0, 2.0]),
        ('wide',     [1.0, 1.6]),
        ('deep',     [1.0, 1.4]),
        ('mc',       [1.0, 1.8]),
        ('gw',       [1.0, 2.0]),
        ('extra',    [1.0, 1.6]),
        ('delver',   [1.0, 1.3]),
    ])

    def __init__(self, *args, **kwargs):
        super(DelveTactician,self).__init__(*args,**kwargs)
        #Default to mode 'wide' if no mode in kwargs
        self.mode = kwargs.get('mode','wide')

    @property
    def viable_fields(self):
        viable = super(DelveTactician,self).viable_fields
        viable &= (self.fields['PRIORITY'] >= 0)
        return viable

    def skybright_select(self):
        """Select fields based on skybrightness and band.

        Parameters:
        -----------
        None

        Returns:
        --------
        sel : boolean selection
        """
        sel = np.ones(len(self.fields),dtype=bool)

        if (self.sun.alt > -0.28):
            # i,z if Sun altitude > -16 deg
            sel &= (np.char.count('iz',self.fields['FILTER'].astype(str)) > 0)
        elif (self.moon.phase >= 50) and (self.moon.alt > 0.175):
            # Moon is very bright; only do i,z
            sel &= (np.char.count('iz',self.fields['FILTER'].astype(str)) > 0)
        elif (self.moon.phase >= 30) and (self.moon.alt > 0.0):
            # Moon is moderately full; do r,i
            sel &= (np.char.count('riz',self.fields['FILTER'].astype(str)) > 0)
        elif (self.moon.phase >= 20) and (self.moon.alt > 0.1):
            # Moon is up; do g,r,i
            sel &= (np.char.count('ri',self.fields['FILTER'].astype(str)) > 0)
        else:
            # Moon is faint or down; do g,r,(i)
            sel &= (np.char.count('gri',self.fields['FILTER'].astype(str)) > 0)
        return sel

    @property
    def weight(self):
        """ Calculate the weight from set of programs. """

        if self.mode is None:
            # First priority is deep
            weights = self.weight_deep()
            if self.fwhm < FWHM_DEEP and np.isfinite(weights).sum():
                logging.info("DEEP")
                return weights
            # Then mc
            weights = self.weight_mc()
            if self.fwhm < FWHM_MC and np.isfinite(weights).sum():
                logging.info("MC")
                return weights
            # Then wide
            weights = self.weight_wide()
            if np.isfinite(weights).sum():
                logging.info("WIDE")
                return weights
        elif self.mode == 'deep':
            return self.weight_deep()
        elif self.mode == 'mc':
            return self.weight_mc()
        elif self.mode == 'wide':
            return self.weight_wide()
        elif self.mode == 'gw':
            return self.weight_gw()
        elif self.mode == 'extra':
            return self.weight_extra()
        elif self.mode == 'delver':
            return self.weight_delver()
        else:
            raise ValueError("Unrecognized mode: %s"%self.mode)

        raise ValueError("No viable fields")

    def weight_deep(self):
        """ Calculate the field weight for the WIDE survey.

        Parameters
        ----------
        None

        Returns
        -------
        weight : array of weights per field
        """
        airmass = self.airmass
        moon_angle = self.moon_angle

        sel = self.viable_fields
        sel &= (self.fields['PROGRAM'] == 'delve-deep')

        weight = np.zeros(len(sel))

        # Moon angle constraints
        moon_limit = 30. # + (self.moon.phase/5.)
        sel &= (moon_angle > moon_limit)

        # Sky brightness selection
        sel &= self.skybright_select()
        #sel &= self.fields['FILTER'] == 'z'

        # Airmass cut
        airmass_min, airmass_max = self.CONDITIONS['deep']
        sel &= ((airmass > airmass_min) & (airmass < airmass_max))

        ## Try hard to do high priority fields
        weight += 1e2 * self.fields['PRIORITY']
        ## Weight different fields

        sexB = (self.fields['HEX'] >= 100000) & (self.fields['HEX'] < 100100)
        sel[sexB] = False

        ic5152 = (self.fields['HEX'] >= 100100) & (self.fields['HEX'] < 100200)
        weight[ic5152] += 0.0

        ngc300 = (self.fields['HEX'] >= 100200) & (self.fields['HEX'] < 100300)
        weight[ngc300] += 1e3
        #sel[ngc300] = False

        ngc55 = (self.fields['HEX'] >= 100300) & (self.fields['HEX'] < 100400)
        sel[ngc55] = False

        # Set infinite weight to all disallowed fields
        weight[~sel] = np.inf

        return weight

    def weight_mc(self):
        """ Calculate the field weight for the MC surve.

        Parameters
        ----------
        None

        Returns
        -------
        weight : array of weights per field
        """
        airmass = self.airmass
        moon_angle = self.moon_angle

        sel = self.viable_fields
        sel &= (self.fields['PROGRAM'] == 'delve-mc')

        # DEC cut for LN2 lines
        #sel &= (self.fields['DEC'] > -75)

        weight = np.zeros(len(sel))

        # Moon angle constraints
        moon_limit = 30.
        sel &= (moon_angle > moon_limit)

        # General airmass restrictions
        airmass_min, airmass_max = self.CONDITIONS['mc']
        sel &= ((airmass > airmass_min) & (airmass < airmass_max))

        # Some tweaking for good and bad conditions
        #self.fwhm = 1.2
        #if self.fwhm < 0.9:
        #    # Prefer fields near the pole
        #    weight += 5e2 * (self.fields['DEC'] > -70)
        #elif self.fwhm < 1.0:
        #    # Prefer fields near the pole
        #    weight += 1e2 * (self.fields['DEC'] > -60)
        if self.fwhm > 1.1:
            weight += 5e3 * (airmass - 1.0)**3
        else:
            # Higher weight for higher airmass
            # airmass = 1.4 -> weight = 6.4
            # airmass = 2.0 -> weight = 500
            #weight += 5e2 * (airmass - 1.0)**3
            pass

        # Sky brightness selection
        sel &= self.skybright_select()
        #sel &= np.in1d(self.fields['FILTER'], ['g','r'])
        sel &= self.fields['FILTER'] == ['r']

        # Only a single tiling
        #sel &= (self.fields['PRIORITY'] == 3)

        # Get fields before they set
        #weight += 1.0 * self.hour_angle
        #weight += 0.1 * self.hour_angle

        # Prioritize early tiling fields
        #weight += 3. * 360. * self.fields['PRIORITY'] * (self.fields['TILING'] > 2)
        #weight += 3e4       * (self.fields['TILING'] > 3)
        weight += 1e5       * (self.fields['TILING'] > 3)
        #sel &= (self.fields['TILING'] < 4)

        # Slew weighting
        # slew = 10 deg -> weight = 10^2
        weight += 0.1 * self.slew**3
        # Try hard to do the same field
        weight += 1e5 * (self.slew != 0)

        # Set infinite weight to all disallowed fields
        weight[~sel] = np.inf

        return weight

    def weight_wide(self):
        """ Calculate the field weight for the WIDE survey.

        Parameters
        ----------
        None

        Returns
        -------
        weight : array of weights per field
        """
        airmass = self.airmass
        moon_angle = self.moon_angle

        sel = self.viable_fields
        sel &= (self.fields['PROGRAM'].astype(str) == 'delve-wide')

        weight = np.zeros(len(sel))

        # RA,DEC cuts
        #sel &= (self.fields['DEC'] < -30)
        #sel &= ~((self.fields['RA'] > 120) & (self.fields['RA'] < 220))
        #weight += 1e3 * (self.fields['DEC'] > -30

        # GLON, GLAT cuts
        glon,glat = cel2gal(self.fields['RA'],self.fields['DEC'])
        # Remove southern galactic cap
        #sel &= (glon >= 180)
        #sel &= (glat < 0)
        # Remove bulge region
        sel &= ~( ((glon < 30) | (glon > 330)) & (np.abs(glat) < 15) )

        # Only one tiling
        #sel &= (self.fields['TILING'] <= 3)

        # Sky brightness selection
        sel &= self.skybright_select()

        #if (self.moon.phase <= 10) or (self.moon.alt < 0.0):
        #    sel &= np.in1d(self.fields['FILTER'], ['g'])
        weight += 3e3 * np.in1d(self.fields['FILTER'], ['i'])

        # Airmass cut
        airmass_min, airmass_max = self.CONDITIONS['wide']

        self.fwhm = 1.0
        if True:
            sel &= ((airmass > airmass_min) & (airmass < airmass_max))
        elif self.fwhm <= 0.9:
            sel &= ((airmass > airmass_min) & (airmass < airmass_max))
            weight += 5e1 * (1.0/airmass)**3
        elif self.fwhm <= 1.1:
            sel &= ((airmass > airmass_min) & (airmass < 1.6))
        elif self.fwhm <= 1.4:
            sel &= ((airmass > airmass_min) & (airmass < 1.4))
            weight += 1e2 * (airmass - 1.0)**3
        else:
            sel &= ((airmass > airmass_min) & (airmass < 1.3))
            weight += 1e2 * (airmass - 1.0)**3

        #if self.fwhm <= 1.1:
        #    # Prefer fields near the pole
        #    weight += 5e2 * ( (self.fields['DEC'] > -60) & (self.fields['RA'] > 270) )

        # Higher weight for fields close to the moon (when up)
        # angle = 50 -> weight = 6.4
        # Moon angle constraints (viable fields sets moon_angle > 20.)
        if (self.moon.alt > -0.04) and (self.moon.phase >= 10):
            #moon_limit = np.min(20 + self.moon.phase/2., 40)
            moon_limit = 40. + (self.moon.phase/10.)
            sel &= (moon_angle > moon_limit)

            #weight += 100 * (35./moon_angle)**3
            #weight += 10 * (35./moon_angle)**3
            weight += 1 * (35./moon_angle)**3

        # Higher weight for rising fields (higher hour angle)
        # HA [min,max] = [-53,54] (for airmass 1.4)
        #weight += 10.0 * self.hour_angle
        weight += 1.0 * self.hour_angle
        #weight += 0.1 * self.hour_angle

        # Higher weight for larger slews
        # slew = 10 deg -> weight = 1e2
        #weight += self.slew**2
        weight += 1e0 * self.slew
        #weight += 1e3 * self.slew

        # Higher weight for higher airmass
        # airmass = 1.4 -> weight = 6.4
        weight += 100. * (airmass - 1.0)**3
        #weight += 1e3 * (airmass - 1.0)**2

        # Hack priority near edge of DES S82 (doesn't really work)
        #x = (self.fields['RA'] >= 45) & (self.fields['RA'] <= 100) \
        #    & (self.fields['DEC'] >= -20)
        #self.fields['PRIORITY'][x] = np.minimum(self.fields['PRIORITY'][x],1)

        ## Try hard to do high priority fields
        weight += 1e1 * (self.fields['PRIORITY'] - 1)
        #weight += 1e4 * (self.fields['TILING'] > 3)

        # Set infinite weight to all disallowed fields
        weight[~sel] = np.inf

        return weight

    def weight_extra(self):
        """ Calculate the field weight for the z-band survey.

        Parameters
        ----------
        None

        Returns
        -------
        weight : array of weights per field
        """
        airmass = self.airmass
        moon_angle = self.moon_angle

        sel = self.viable_fields
        sel &= (self.fields['PROGRAM'] == 'delve-extra')

        # Extra restriction due to dome windscreen
        sel &= self.airmass < 1.9

        weight = np.zeros(len(sel))

        # Sky brightness selection
        sel &= self.skybright_select()

        # Select only one band
        #sel &= np.in1d(self.fields['FILTER'], ['g','r','i'])
        #sel &= np.in1d(self.fields['FILTER'], ['g','r'])
        sel &= np.in1d(self.fields['FILTER'], ['i','z'])
        weight += 1e3 * np.in1d(self.fields['FILTER'], ['z'])
        #sel &= np.in1d(self.fields['FILTER'], ['i'])
        #if (self.moon.phase >= 9) and (self.moon.alt > 0.175):
        #    sel &= np.in1d(self.fields['FILTER'], ['i'])
        #if (self.moon.phase >= 70) and (self.moon.alt < 0.0):
        #    sel &= np.in1d(self.fields['FILTER'], ['i'])
        #if (self.moon.phase >= 70) and (self.moon.alt > 0.45):
        #    #sel &= np.in1d(self.fields['FILTER'], ['z'])
        #    weight += 1e3 * np.in1d(self.fields['FILTER'], ['i'])
        #if (self.moon.phase < 90) and (self.moon.alt < 0.4):
        #    #sel &= np.in1d(self.fields['FILTER'], ['i'])
        #    weight += 1e3 * np.in1d(self.fields['FILTER'], ['z'])

        # Select only first tiling
        #sel &= (self.fields['TILING'] <= 2)

        # GLON, GLAT cuts
        glon,glat = cel2gal(self.fields['RA'],self.fields['DEC'])
        #sel &= (glon >= 180)
        #sel &= (glat < 0)
        # Remove bulge region
        sel &= ~( ((glon < 30) | (glon > 330)) & (np.abs(glat) < 15) )
        #sel &= (glat < -12)

        # Select only one region
        #sel &= ((self.fields['DEC'] < -30) | \
        #        ((self.fields['RA'] > 30) & (self.fields['RA'] < 180) & \
        #         (self.fields['DEC'] > 0)))
        #sel &= ((self.fields['DEC'] < -30) | (self.fields['DEC'] > 0) | \
        #        (self.fields['RA'] > 90))
        #sel &= ((self.fields['DEC'] < -30) | (self.fields['DEC'] > 0))
        #sel &= self.fields['DEC'] < 15
        #sel &= (self.fields['DEC'] > 0)
        #sel &= (self.fields['RA'] > 150)
        #weight += 1e3 * (self.fields['DEC'] < 0)

        # Airmass cut
        #self.fwhm = 1.2
        airmass_min, airmass_max = self.CONDITIONS['extra']
        if True:
            sel &= ((airmass > airmass_min) & (airmass < airmass_max))
        elif self.fwhm <= 1.0:
            sel &= ((airmass > airmass_min) & (airmass < airmass_max))
            weight += 5e1 * (1.0/airmass)**3
        elif self.fwhm <= 1.2:
            sel &= ((airmass > airmass_min) & (airmass < 1.5))
            #weight += 5e1 * (1.0/airmass)**3
        else:
            sel &= ((airmass > airmass_min) & (airmass < 1.4))
            weight += 1e2 * (airmass - 1.0)**3

        # Higher weight for higher airmass
        # airmass = 1.4 -> weight = 6.4
        #weight += 100. * (airmass - 1.)**3
        weight += 1e3 * (airmass - 1.)**2

        # Higher weight for fields close to the moon (when up)
        # angle = 50 -> weight = 6.4
        # Moon angle constraints (viable fields sets moon_angle > 20.)
        if (self.moon.alt > -0.04) and (self.moon.phase >= 30):
            moon_limit = 45.0
            sel &= (moon_angle > moon_limit)

            # Use a larger (smaller) weight to increase (decrease) the
            # moon avoidance angle.
            #weight += 100 * (35./moon_angle)**3
            weight += 10 * (35./moon_angle)**3
            #weight += 1 * (35./moon_angle)**3

        # Higher weight for rising fields (higher hour angle)
        # HA [min,max] = [-53,54] (for airmass 1.4)
        weight += 10.0 * self.hour_angle
        #weight += 2.0 * self.hour_angle
        #weight += 0.5 * self.hour_angle

        # Higher weight for larger slews
        # slew = 10 deg -> weight = 1e2
        #weight += self.slew**2
        weight += self.slew
        #weight += 1e3 * self.slew

        ## Try hard to do high priority fields
        weight += 3e1 * (self.fields['PRIORITY'] - 1)
        weight += 1e7 * (self.fields['TILING'] > 3)

        # Set infinite weight to all disallowed fields
        weight[~sel] = np.inf

        return weight

    def weight_delver(self):
        """ Calculate the field weight for the r-band survey.

        Parameters
        ----------
        None

        Returns
        -------
        weight : array of weights per field
        """
        #self.fields.PROPID = '2022B-780972'
        #self.fields.PROPID = '2023A-343956'
        #self.fields.SISPI_DICT["propid"] = self.fields.PROPID

        airmass = self.airmass
        moon_angle = self.moon_angle

        sel = self.viable_fields
        sel &= (self.fields['PROGRAM'] == 'delve-extra')

        weight = np.zeros(len(sel))

        # Sky brightness selection
        sel &= (self.fields['FILTER'] == 'r')

        # GLON, GLAT cuts
        glon,glat = cel2gal(self.fields['RA'],self.fields['DEC'])
        #sel &= (glon >= 180)
        sel &= (glat < -32)
        # Remove bulge region
        #sel &= ~( ((glon < 30) | (glon > 330)) & (np.abs(glat) < 15) )

        # Select region between S82 and SPT
        sel &= (self.fields['DEC'] < -10) & (self.fields['DEC'] > -40)
        sel &= (self.fields['DEC'] < -15)
        #sel &= (self.fields['RA'] > 290)

        # Only first tiling
        #sel &= np.in1d(self.fields['TILING'],[2])

        # Airmass cut
        airmass_min, airmass_max = self.CONDITIONS['delver']
        sel &= ((airmass > airmass_min) & (airmass < airmass_max))

        # Higher weight for fields close to the moon (when up)
        # angle = 50 -> weight = 6.4
        # Moon angle constraints (viable fields sets moon_angle > 20.)
        if (self.moon.alt > -0.04) and (self.moon.phase >= 30):
            moon_limit = 40.0
            sel &= (moon_angle > moon_limit)

            # Use a larger (smaller) weight to increase (decrease) the
            # moon avoidance angle.
            #weight += 100 * (35./moon_angle)**3
            weight += 10 * (35./moon_angle)**3
            #weight += 1 * (35./moon_angle)**3

        # Higher weight for rising fields (higher hour angle)
        # HA [min,max] = [-53,54] (for airmass 1.4)
        #weight += 10.0 * self.hour_angle
        weight += 1.0 * self.hour_angle
        #weight += 0.1 * self.hour_angle

        # Higher weight for larger slews
        # slew = 10 deg -> weight = 1e2
        #weight += self.slew**2
        weight += self.slew
        #weight += 1e3 * self.slew

        # Higher weight for higher airmass
        # airmass = 1.4 -> weight = 6.4
        #weight += 100. * (airmass - 1.)**3
        weight += 1e3 * (airmass - 1.)**2

        ## Try hard to do high priority fields
        weight += 1e1 * (self.fields['PRIORITY'] - 1)
        weight += 1e5 * (self.fields['TILING'] > 3)

        # Set infinite weight to all disallowed fields
        weight[~sel] = np.inf

        return weight

    def weight_gw(self):
        """ Calculate the field weight for the WIDE survey. """
        raise DeprecationWarning

        import healpy as hp
        airmass = self.airmass
        moon_angle = self.moon_angle

        # Reset the exposure time
        self.fields['EXPTIME'] = 90

        if hasattr(self.fields,'hpx'):
            hpx = self.fields.hpx
        else:
            hpx = hp.ang2pix(32,self.fields['RA'],self.fields['DEC'],lonlat=True)
            setattr(self.fields,'hpx',hpx)

        gwpix = np.genfromtxt(fileio.get_datafile('GW150914_hpixels_32.tab'))

        #sel = self.viable_fields
        sel = np.in1d(hpx,gwpix)
        sel &= self.fields['FILTER'] == 'g'

        weight = np.zeros(len(sel))

        # Sky brightness selection

        # Airmass cut
        airmass_min, airmass_max = self.CONDITIONS['gw']
        sel &= ((airmass > airmass_min) & (airmass < airmass_max))

        """
        # Higher weight for fields close to the moon (when up)
        # angle = 50 -> weight = 6.4
        # Moon angle constraints (viable fields sets moon_angle > 20.)
        if (self.moon.alt > -0.04) and (self.moon.phase >= 10):
            #moon_limit = np.min(20 + self.moon.phase/2., 40)
            moon_limit = 40
            sel &= (moon_angle > moon_limit)

            #weight += 100 * (35./moon_angle)**3
            #weight += 10 * (35./moon_angle)**3
            weight += 1 * (35./moon_angle)**3
        """

        # Higher weight for rising fields (higher hour angle)
        # HA [min,max] = [-53,54] (for airmass 1.4)
        #weight += 5.0 * self.hour_angle
        #weight += 1.0 * self.hour_angle
        #weight += 0.1 * self.hour_angle

        # Higher weight for larger slews
        # slew = 10 deg -> weight = 1e2
        weight += self.slew**2
        #weight += self.slew
        #weight += 1e3 * self.slew

        # Higher weight for higher airmass
        # airmass = 1.4 -> weight = 6.4
        weight += 1e3 * (airmass - 1.)**3
        #weight += 1e3 * (airmass - 1.)**2

        ## Try hard to do high priority fields
        #weight += 1e3 * (self.fields['PRIORITY'] - 1)
        #weight += 1e4 * (self.fields['TILING'] > 3)

        # Set infinite weight to all disallowed fields
        weight[~sel] = np.inf

        return weight

    def select_index(self):
        weight = self.weight
        index = np.array([np.argmin(weight)],dtype=int)
        if np.any(~np.isfinite(weight[index])):
            plot = (logging.getLogger().getEffectiveLevel()==logging.DEBUG)
            msg = "Infinite weight selected..."
            logging.warn(msg)
            logging.info(">>> To plot fields enter 'plot=True'")
            logging.info(">>> Enter 'c' to continue")
            import pdb; pdb.set_trace()
            if plot:
                import obztak.utils.ortho, pylab as plt
                airmass = self.CONDITIONS[self.mode][1]
                bmap = obztak.utils.ortho.plotFields(self.completed_fields[-1],self.fields,self.completed_fields,options_basemap=dict(airmass=airmass))
                logging.info(">>> Enter 'c' to continue")
                pdb.set_trace()
            raise ValueError(msg)

        return index
