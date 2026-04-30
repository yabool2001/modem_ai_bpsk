import  numpy as np , tomllib
from    numpy.typing import NDArray

with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

SPS  = int ( toml_settings[ "bpsk" ][ "SPS" ] )

def bpsk_symbols_2_bits ( symbols : NDArray[ np.float64 ] ) -> NDArray[ np.uint8 ] :
    """
    Odwrotna funkcja do create_bpsk_symbols_v0_1_6_fastest_short.
    Mapuje symbole BPSK (-1+0j -> 0, 1+0j -> 1) na bity.
    """
    return ( symbols > 0 ).astype ( np.uint8 )

def bits_2_bpsk_symbols ( bits : NDArray[ np.uint8 ] , sps : int | None = None ) -> NDArray[ np.complex128 ] :
    """
    Mapuje bity (0 -> -1+0j, 1 -> 1+0j) na symbole BPSK.
    Jeśli sps jest podane, każdy symbol jest powielany sps razy.
    """
    bits = np.asarray ( bits , dtype = np.uint8 )
    if np.any ( ( bits != 0 ) & ( bits != 1 ) ) :
        raise ValueError ( "bits must contain only 0 or 1" )
    symbols = ( bits.astype ( np.float64 ) * 2.0 - 1.0 ).astype ( np.complex128 )
    if sps is None :
        return symbols
    sps = int ( sps )
    if sps < 1 :
        raise ValueError ( "sps must be >= 1" )
    return np.repeat ( symbols , sps )