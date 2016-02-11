"""
Constants.
"""
import ephem
from collections import OrderedDict as odict

RA_LMC = 80.8939
DEC_LMC = -69.7561
RA_SMC = 13.1867
DEC_SMC = -72.8286

# http://www.ctio.noao.edu/noao/content/coordinates-observatories-cerro-tololo-and-cerro-pachon

LON_CTIO = '-70:48:23.49'
LAT_CTIO = '-30:10:10.78'
ELEVATION_CTIO = 2206.8 # m

# Pole of the SMASH fields (RA,DEC)
SMASH_POLE = (10., -30.)

# Characteristics of the survey
# 90 sec exposures with 30 sec between exposures
EXPTIME   = 90*ephem.second # Exposure time
DOWNTIME  = 30*ephem.second # Time between exposures from readout/slew
NEXP      = 2 # Number of exposures taken in a row
FIELDTIME = EXPTIME+DOWNTIME
BANDS     = ('g','r')

# Time for taking standards
STANDARDS = 10*ephem.minute

# Characteristics of DECam
ARCSEC_TO_DEGREE = 1. / (60. * 60.)
PIXEL_SCALE = 0.2626 * ARCSEC_TO_DEGREE
NPIX_X = 4096
NPIX_Y = 2048
CCD_X = NPIX_X * PIXEL_SCALE # degree
CCD_Y = NPIX_Y * PIXEL_SCALE # degree

# Blanco characteritics
SOUTHERN_REACH = -89.

# SISPI json template formatting
PROPID = '2016A-0366'
OBJECT_FMT = "MAGLITES field - %(ID)d.%(TILING)d.%(PRIORITY)d"
SEQID_FMT = "MAGLITES scheduled - %(DATE)s"
FLOAT_FMT = '%.4f'
SISPI_DICT = odict([
    ("object",  None),
    ("seqnum",  None), # 1-indexed
    ("seqtot",  2),
    ("seqid",   None),
    ("expTime", 90),
    ("RA",      None),
    ("dec",     None),
    ("filter",  None),
    ("count",   1),
    ("expType", "object"),
    ("program", "maglites"),
    ("wait",    "False"),
    ("propid",  PROPID),
    ("comment", ""),
])

def FIELD2OBJECT(field):
    return OBJECT_FMT%(field)

def OBJECT2FIELD(object_str):
    ID,TILING,PRIORITY = map(int,object_str.split(' - ')[-1].split('.'))
    return dict(ID=ID,TILING=TILING,PRIORITY=PRIORITY)

def FIELD2SEQID(field):
    return SEQID_FMT%(field)

def SEQID2FIELD(seqid_str):
    DATE = str(seqid_str.split(' - ')[-1].strip())
    return dict(DATE=DATE)

