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
import healpy as hp

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

dataDir = "/Users/sean/Desktop/Repos/obztak-Sirenas/obztak/data/sirenasEventFiles/"
bearEventFile =  dataDir + "bearEvents.csv"
o4EventFile = dataDir + "o4Events.csv"
o5EventFile = dataDir + "o5Events.csv"
skymapDictFile = dataDir + "skymap_mapping.fits.gz"

eventNameDict = {0:"S250119cv",       
		 1:"S240527fv",    
		 2:"S240915b" ,   
		 3:"GW190814" ,   
		 4:"GW170818" ,   
		 5:"GW200311" ,   		
		 6:"GW200208" ,   
                 7:"S240413p" ,   
                 8:"GW190701" ,   
                 9:"GW200224" ,   
                 10:"S250205bk",  
                 11:"S240908bs",
                 12:"S240511i",
                 13:"GW170814",
                 14:"S240922df",
                 15:"GW190503",
		 16:"S240514x",
                 17:"GW200202",
                 18:"S240923ct",
                 19:"S241110br"}

class SirenasSurvey(Survey):
    """ Survey sublcass for SIRENAS. """

    """ Instantiate the relevant nights and half nights"""

# These are some test nights

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
        ['2024/01/28','full']

    ]

    extra_nights = []

    nights = nights_2023B \
             + extra_nights

    eventMappingSkymap,mappingHeader = hp.read_map(skymapDictFile,verbose=False)
    eventMappingNpix = len(eventMappingSkymap)
    eventMappingNside = hp.npix2nside(eventMappingNpix)

    """ 

    FIELDS 
    
    We have three mini-surveys: bear, O4, and O5
    They should proceed sequentially, but optimize ordering of hexes for each mini-survey
    As of now, we are deselecting on the milky way only. We have the option to deselect based on SMASH/DES/DEEP fields, but that is not set to true now.

    Current status: Finished not tested; "That's all" - Phil Collins

    """

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

        bear_fields  = self.create_bear_fields(data)
        O4_fields    = self.create_O4_fields(data)
        O5_fields  = self.create_O5_fields(data)

	""" Comment: Sean is commenting overlaps out, since that is not really what we want to do here. Overlaps are okay, especially for a first pass"""
        # Overlap fields
        # sel  = (~np.in1d(extra_fields.unique_id,wide_fields.unique_id))
        # sel &= (~np.in1d(extra_fields.unique_id,mc_fields.unique_id))
        ## Mark overlap fields as DONE
        #extra_fields['PRIORITY'][~sel] = DONE
        # Remove overlap fields
        # extra_fields = extra_fields[sel]

        fields = bear_fields + O4_fields + O5_fields 

        logging.info("Masking bright stars...")
        mask = self.bright_stars(fields['RA'],fields['DEC'])
        fields['PRIORITY'][mask] = DONE

        # Can't go lower
        mask = fields['DEC'] < -89.0
        fields['PRIORITY'][mask] = DONE


	""" 
	Exclusion zones

	This is in the form of hex#-tile#-band

	Comment: I don't think we have any exclusion zones, beside the milky way for obvious reasons. Therefore, I am commenting these out. 
	"""
        # Exclusion
#         exclude  = [#pole
#             '399-01-g','399-01-r','399-01-i',
#             '399-02-g','399-02-r','399-02-i',
#             '399-04-g','399-04-r','399-04-i',
#             '436-01-i','437-01-i',
#             '417-03-g','417-03-r','417-03-i',
#         ]
#         exclude += [ # Orion Nebula
#             '5696-01-i','5696-02-i','5696-03-i','5696-04-i',
#             '5760-01-i','5760-02-i','5760-03-i','5760-04-i',
#             '5761-01-i','5761-02-i','5761-03-i','5761-04-i',
#             '5780-01-i','5780-02-i','5780-03-i','5780-04-i',
#             '5781-01-i','5781-02-i','5781-03-i','5781-04-i',
#             '5782-01-i','5782-02-i','5782-03-i','5782-04-i',
#             '5783-01-i','5783-02-i','5783-03-i','5783-04-i',
#             '5798-01-i','5798-02-i','5798-03-i','5798-04-i',
#             '5799-01-i','5799-02-i','5799-03-i','5799-04-i',
#         ]
#         exclude += [# flame nebula
#             '5716-01-g','5716-01-i'
#         ]
#         exclude += [# rho ophiuchi nebula
#         ]
#         exclude += [ # MC poles
#             '14110-01-g','14110-02-g','14110-03-g','14110-04-g',
#             '14110-01-r','14110-02-r','14110-03-r','14110-04-r',
#             '14110-01-i','14110-02-i','14110-03-i','14110-04-i',
#             '15464-01-i','15464-02-i','15464-03-i','15464-04-i',
#             '15465-01-i','15465-02-i','15465-03-i','15465-04-i',
#             '15484-03-i',
#         ]
#         exclude += [ # Other
#             '7246-01-r','7264-01-i',
#             '7246-01-i','7246-02-i','7246-02-i',
#             '12253-03-i','14238-03-i',
#             '14241-03-i','14255-03-i','14256-03-i','14257-03-i',
#             '14258-03-i','15465-04-g',
#         ]
#         fields['PRIORITY'][np.in1d(fields.unique_id,exclude)] = DONE

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
#             smap.draw_smash()

            plt.figure()
            smap = skymap.survey.SurveyMcBryde()
            smap.draw_fields(fields[sel],alpha=0.3,edgecolor='none')
            smap.draw_des(color='r')
            smap.draw_milky_way()
#             smap.draw_smash()

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
    
    @classmethod
    def create_bear_fields(self, data, plot=False):
        """ 
	Create the bear field observations 
	At present, I am avoiding the milky way only. 
	We can elect to avoid DES/SMASH/Deep field observations too, but that is disabled (for now)
	"""
        logging.info("Creating BEAR fields...")

        BANDS = ['g','i','z','r','u']
        EXPTIME = 90
        # TILINGS = [4,4,4,4,4] # commenting this out because the number of tilings is variable
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
        # fields['TILING'] = np.repeat(data['PASS'],nbands)
        fields['RA'] = np.repeat(data['RA'],nbands)
        fields['DEC'] = np.repeat(data['DEC'],nbands)

        fields['FILTER'] = np.tile(BANDS,len(data))
        fields['EXPTIME'] = np.tile(EXPTIME,len(data))


        sel = self.footprintBEAR(fields['RA'],fields['DEC']) 
        sel &= (~self.footprintMilkyWay(fields['RA'],fields['DEC'])) # Avoiding milky way
        # sel &= (~self.footprintDES(fields['RA'],fields['DEC'])) # Avoiding DES fields
        #sel &= (~self.footprintSMASH(fields['RA'],fields['DEC'],angsep=0.75*DECAM))
        #sel &= (~self.footprintmc(fields['RA'],fields['DEC']))
        # Avoid DEEP fields? yes.
        # sel &= (~self.footprintDEEP(fields['RA'],fields['DEC']))

        fields = fields[sel]

	fields['TILING'] = self.computeTilings(fields,BANDS) 
        fields['PRIORITY'] = fields['TILING']

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

    @classmethod
    def create_O4_fields(self, data, plot=False):
        """ 
	Create the O4 field observations 
	At present, I am avoiding the milky way only. 
	We can elect to avoid DES/SMASH/Deep field observations too, but that is disabled (for now)
	"""
        logging.info("Creating O4 fields...")
        BANDS = ['g','i','z','r','u']
        EXPTIME = [90,90,90,90,90]
        TILINGS = [4,4,4,4,4]
        TEFF_MIN = TEFF_MIN_O4

        nhexes = len(np.unique(data['TILEID']))
        nbands = len(BANDS)

        nfields = len(data)*nbands

        logging.info("  Number of hexes: %d"%nhexes)
        logging.info("  Filters: %s"%BANDS)
        logging.info("  Exposure time: %s"%EXPTIME)
        logging.info("  Tilings: %s"%TILINGS)

        fields = FieldArray(nfields)
        fields['PROGRAM'] = PROGRAM+'-O4'
        fields['HEX'] = np.repeat(data['TILEID'],nbands)
        fields['TILING'] = np.repeat(data['PASS'],nbands)
        fields['RA'] = np.repeat(data['RA'],nbands)
        fields['DEC'] = np.repeat(data['DEC'],nbands)

        fields['FILTER'] = np.tile(BANDS,len(data))
        fields['EXPTIME'] = np.tile(EXPTIME,len(data))
        fields['PRIORITY'] = fields['TILING']

        sel = self.footprintO4(fields['RA'],fields['DEC']) # Bear footprint?
        sel &= (~self.footprintMilkyWay(fields['RA'],fields['DEC'])) # Avoiding milky way
        # sel &= (~self.footprintDES(fields['RA'],fields['DEC'])) # Avoiding DES fields
        #sel &= (~self.footprintSMASH(fields['RA'],fields['DEC'],angsep=0.75*DECAM))
        #sel &= (~self.footprintmc(fields['RA'],fields['DEC']))
        # Avoid DEEP fields? yes.
        # sel &= (~self.footprintDEEP(fields['RA'],fields['DEC']))

        fields = fields[sel]

        # Covered fields
        frac, depth = self.covered(fields)
        teffmin = pd.DataFrame(fields).merge(TEFF_MIN,on='FILTER',how='left').to_records()['TEFF']
        fields['PRIORITY'][depth > teffmin*fields['TILING']*fields['EXPTIME']] = DONE

        if plot: self.plot_depth(fields,depth,'sirenas-O4-%s-gt%i.png')

        logging.info("Number of target fields: %d"%len(fields))

        outfile = 'sirenas-O4-fields.fits.fz'
        logging.info("Writing %s..."%outfile)
        fields.write(outfile,clobber=True)

        return fields

    @classmethod
    def create_O5_fields(self, data, plot=False):
        """ 
	Create the O5 field observations 
	At present, I am avoiding the milky way only.
	We can elect to avoid DES/SMASH/Deep field observations too, but that is disabled (for now)
	"""
        logging.info("Creating O5 fields...")
        BANDS = ["M4112", "M4376", "M4640", "M4904", "M5168"]
        EXPTIME = [90,90,90,90,90]
        TILINGS = [4,4,4,4,4]
        TEFF_MIN = TEFF_MIN_O5

        nhexes = len(np.unique(data['TILEID']))
        nbands = len(BANDS)

        nfields = len(data)*nbands

        logging.info("  Number of hexes: %d"%nhexes)
        logging.info("  Filters: %s"%BANDS)
        logging.info("  Exposure time: %s"%EXPTIME)
        logging.info("  Tilings: %s"%TILINGS)

        fields = FieldArray(nfields)
        fields['PROGRAM'] = PROGRAM+'-O5'
        fields['HEX'] = np.repeat(data['TILEID'],nbands)
        fields['TILING'] = np.repeat(data['PASS'],nbands)
        fields['RA'] = np.repeat(data['RA'],nbands)
        fields['DEC'] = np.repeat(data['DEC'],nbands)

        fields['FILTER'] = np.tile(BANDS,len(data))
        fields['EXPTIME'] = np.tile(EXPTIME,len(data))
        fields['PRIORITY'] = fields['TILING']

        sel = self.footprintO5(fields['RA'],fields['DEC']) # Bear footprint?
        sel &= (~self.footprintMilkyWay(fields['RA'],fields['DEC'])) # Avoiding milky way
        # sel &= (~self.footprintDES(fields['RA'],fields['DEC'])) # Avoiding DES fields
        #sel &= (~self.footprintSMASH(fields['RA'],fields['DEC'],angsep=0.75*DECAM))
        #sel &= (~self.footprintmc(fields['RA'],fields['DEC']))
        # Avoid DEEP fields? yes.
        # sel &= (~self.footprintDEEP(fields['RA'],fields['DEC']))

        fields = fields[sel]

        # Covered fields
        frac, depth = self.covered(fields)
        teffmin = pd.DataFrame(fields).merge(TEFF_MIN,on='FILTER',how='left').to_records()['TEFF']
        fields['PRIORITY'][depth > teffmin*fields['TILING']*fields['EXPTIME']] = DONE

        if plot: self.plot_depth(fields,depth,'sirenas-O5-%s-gt%i.png')

        logging.info("Number of target fields: %d"%len(fields))

        outfile = 'sirenas-O5-fields.fits.fz'
        logging.info("Writing %s..."%outfile)
        fields.write(outfile,clobber=True)

        return fields
    
    @classmethod
    def computeTilings(self,fields,bands,mode='bear'):
	"""

	Function to compute the tiling numbers for fields in the Sirenas survey
	This function takes in bands and fields within the footprint, and returns the tilings of each field

	Inputs
	------
	fields: A field array of relevant fields, AFTER being filtered by the footprint
	bands: The range of bands that will be observed in the mini-survey.
	mode: Decides the data file for each tiling computation

	Outputs
	-------
	tileArray: The tilings for all fields
	eventIDs: An array of the event IDs for all pointings

	Notes
	-----
	Support should exist for 0 tiles per band
	
	""" 

	# Read in .csv with the per-event per-band exposures
	if mode=='bear':
	    myCSV = fileIO.read_csv(bearEventFile)
	elif mode=='o4':
	    myCSV = fileIO.read_csv(o4EventFile)
	elif mode=='o5':
	    myCSV = fileIO.read_csv(o5EventFile)
	else:
	    # Mode not recognized
	    raise ValueError("Unrecognized mode: %s\n Supported modes are 'bear', 'o4', and 'o5'"%self.mode)
	
	# pull out the relevant column names that compute the number of exposures for each band
	relevantCols = myCSV.columns.values[[x.__contains__("nexp") for x in myCSV.columns.values]]
	# put in a check that all bands are included in the .csv - later problem

	# Create an empty array for all the tilings	
	tileArray = np.array([],dtype=int)
	eventIDs = np.array([],dtype=str)

	for field in fields:	# For each field
		# Compute which event the field is in
		eventID = getEventNameFromSkymap(field["RA"],field["DEC"]) 
		# Call that row in the .csv
		rowDF = myCSV[myCSV["Event ID"]==eventID]
		# Index the row by the relevant columns from above
		fieldtilings = rowDF[relevantCols].values[0]
		# Append the array by the relevant columns
		tileArray = np.append(tileArray,fieldTilings)
		eventIDs = np.append(eventIDs,eventID)
	
	return tileArray,eventIDs

    @classmethod
    def getEventNameFromSkymap(self,ra,dec):
	"""

	Function to extract an event ID from the skymap localization

	Inputs
	------
	ra: the ra of the pointing
	dec: the dec of the pointing

	Outputs
	-------
	eventName: the event name from the pointing

	Notes
	-----


	"""
	theta = 0.5 * np.pi - np.deg2rad(dec)
	phi = np.deg2rad(ra)
	ipix = hp.ang2pix(eventMappingNside, theta, phi)	

	eventNameID = eventMappingSkymap[ipix]

	eventName = eventNameDict[eventNameID]

	return eventName

    """ 
    FOOTPRINTS 

    Current status: Overhaul in progress, need skymaps to proceed further; "You can't hurry love" - Phil Collins

    """
    
""" TODO: Revise bear footprint selection"""
    @staticmethod
    def footprintBEAR(ra,dec):
        """ Select exposures for BEAR survey """
 	""" In general, if we are in the 90% contour, return true. Else, return false"""
        
	ra,dec = np.copy(ra), np.copy(dec)

	sel = # bool based on above criteria

        return sel

""" TODO: Revise O4 footprint selection """
    @staticmethod
    def footprintO4(ra,dec):
        """ 
	Selecting O4 exposures plane 
	To my knowledge, this should mirror the BEAR survey, but I am not 100% sure at this time

	"""
        ra,dec = np.copy(ra), np.copy(dec)
        

	return sel

""" TODO: Revise O5 footprint selection """
    @staticmethod
    def footprintO5(ra,dec):
        """ 
	Selecting O5 exposures plane 
	This should start with the BEAR footprint, and then downselect for the skymaps we care about
	Given the computational cost of that, we should probably just have a separate skymap
	"""

	ra,dec = np.copy(ra), np.copy(dec)
        
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

    """ 

    TODO: Determine if we want to use the covered method, and if so, if we would like to modify the previously covered fields file

    """
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

class SirenasFieldArray(FieldArray):
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

	TODO: revise this query

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


"""

SCHEDULER



Current status: in overhaul

"""

class SirenasScheduler(Scheduler):
    _defaults = odict(list(Scheduler._defaults.items()) + [
        ('tactician','coverage'),
        ('windows',fileio.get_datafile("sirenas-windows.csv.gz")),
        ('targets',fileio.get_datafile("sirenas-target-fields.csv.gz")),
    ])

    FieldType = SirenasFieldArray

"""

TACTICIAN

Must have a discussion about the airmass minimum and maximums - I set them below as a placeholder.

Current status: in overhaul

"""

class SirenasTactician(Tactician):
    CONDITIONS = odict([   # airmass_min, airmass_max
        (None,       [1.0, 2.0]),
        ('bear',     [1.0, 1.6]),
        ('o4',     [1.0, 1.6]),
        ('o5',       [1.0, 1.8])
        ])

    def __init__(self, *args, **kwargs):
        super(SirenasTactician,self).__init__(*args,**kwargs)
        #Default to mode 'bear' if no mode in kwargs
        self.mode = kwargs.get('mode','bear') 


    @property
    def viable_fields(self):
        viable = super(SirenasTactician,self).viable_fields
        viable &= (self.fields['PRIORITY'] >= 0)
        return viable


    """ TODO (low priority): revise this for medium band strategy """
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


    ''' TODO: revise weights based on strategy choices'''
    @property
    def weight(self):
        """ Calculate the weight from set of programs. """

        if self.mode is None:
            # First priority is bear
            weights = self.weight_bear()
            if self.fwhm < FWHM_BEAR and np.isfinite(weights).sum():
                logging.info("BEAR")
                return weights
            # Then o4
            weights = self.weight_o4()
            if self.fwhm < FWHM_O4 and np.isfinite(weights).sum():
                logging.info("O4")
                return weights
            # Then o5
            weights = self.weight_O5()
            if np.isfinite(weights).sum():
                logging.info("O5")
                return weights
        elif self.mode == 'bear':
            return self.weight_bear()
        elif self.mode == 'o4':
            return self.weight_o4()
        elif self.mode == 'o5':
            return self.weight_o5()
        else:
            raise ValueError("Unrecognized mode: %s"%self.mode)

        raise ValueError("No viable fields")


     def weight_bear(self):
        """ Calculate the field weight for the BEAR survey.

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
        sel &= (self.fields['PROGRAM'] == 'sirenas-bear')

        weight = np.zeros(len(sel))

        # Moon angle constraints
        moon_limit = 30. # + (self.moon.phase/5.)
        sel &= (moon_angle > moon_limit)

        # Sky brightness selection
        sel &= self.skybright_select()
        #sel &= self.fields['FILTER'] == 'z'

        # Airmass cut
        airmass_min, airmass_max = self.CONDITIONS['bear']
        sel &= ((airmass > airmass_min) & (airmass < airmass_max))

        ## Try hard to do high priority fields
        weight += 1e2 * self.fields['PRIORITY']
        ## Weight different fields

	"""
	I am removing the weights on certain fields here, since we just want to hit uniform magnitude over the contours
	"""

        # sexB = (self.fields['HEX'] >= 100000) & (self.fields['HEX'] < 100100)
        # sel[sexB] = False

        # ic5152 = (self.fields['HEX'] >= 100100) & (self.fields['HEX'] < 100200)
        # weight[ic5152] += 0.0

        # ngc300 = (self.fields['HEX'] >= 100200) & (self.fields['HEX'] < 100300)
        # weight[ngc300] += 1e3
        # #sel[ngc300] = False

        # ngc55 = (self.fields['HEX'] >= 100300) & (self.fields['HEX'] < 100400)
        # sel[ngc55] = False

        # # Set infinite weight to all disallowed fields
        weight[~sel] = np.inf

        return weight

    def weight_o4(self):
        """ Calculate the field weight for the o4 survey.

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
        sel &= (self.fields['PROGRAM'] == 'sirenas-o4')

        weight = np.zeros(len(sel))

        # Moon angle constraints
        moon_limit = 30. # + (self.moon.phase/5.)
        sel &= (moon_angle > moon_limit)

        # Sky brightness selection
        sel &= self.skybright_select()
        #sel &= self.fields['FILTER'] == 'z'

        # Airmass cut
        airmass_min, airmass_max = self.CONDITIONS['o4']
        sel &= ((airmass > airmass_min) & (airmass < airmass_max))

        ## Try hard to do high priority fields
        weight += 1e2 * self.fields['PRIORITY']
        ## Weight different fields

	"""
	I am removing the weights on certain fields here, since we just want to hit uniform magnitude over the contours
	"""

        # sexB = (self.fields['HEX'] >= 100000) & (self.fields['HEX'] < 100100)
        # sel[sexB] = False

        # ic5152 = (self.fields['HEX'] >= 100100) & (self.fields['HEX'] < 100200)
        # weight[ic5152] += 0.0

        # ngc300 = (self.fields['HEX'] >= 100200) & (self.fields['HEX'] < 100300)
        # weight[ngc300] += 1e3
        # #sel[ngc300] = False

        # ngc55 = (self.fields['HEX'] >= 100300) & (self.fields['HEX'] < 100400)
        # sel[ngc55] = False

        # # Set infinite weight to all disallowed fields
        weight[~sel] = np.inf

        return weight

    def weight_o5(self):
        """ Calculate the field weight for the o5 survey.

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
        sel &= (self.fields['PROGRAM'] == 'sirenas-o5')

        weight = np.zeros(len(sel))

        # Moon angle constraints
        moon_limit = 30. # + (self.moon.phase/5.)
        sel &= (moon_angle > moon_limit)

        # Sky brightness selection
        sel &= self.skybright_select()
        #sel &= self.fields['FILTER'] == 'z'

        # Airmass cut
        airmass_min, airmass_max = self.CONDITIONS['o5']
        sel &= ((airmass > airmass_min) & (airmass < airmass_max))

        ## Try hard to do high priority fields
        weight += 1e2 * self.fields['PRIORITY']
        ## Weight different fields

	"""
	I am removing the weights on certain fields here, since we just want to hit uniform magnitude over the contours
	"""

        # sexB = (self.fields['HEX'] >= 100000) & (self.fields['HEX'] < 100100)
        # sel[sexB] = False

        # ic5152 = (self.fields['HEX'] >= 100100) & (self.fields['HEX'] < 100200)
        # weight[ic5152] += 0.0

        # ngc300 = (self.fields['HEX'] >= 100200) & (self.fields['HEX'] < 100300)
        # weight[ngc300] += 1e3
        # #sel[ngc300] = False

        # ngc55 = (self.fields['HEX'] >= 100300) & (self.fields['HEX'] < 100400)
        # sel[ngc55] = False

        # # Set infinite weight to all disallowed fields
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
