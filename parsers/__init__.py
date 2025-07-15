"""
TITANUS Data Parsers Package
Parsers for SWIS, SOLERIOX, and Magnetometer data
"""

from . import swis_parser
from . import soleriox_parser
from . import magnetometer_parser

__all__ = ['swis_parser', 'soleriox_parser', 'magnetometer_parser']
