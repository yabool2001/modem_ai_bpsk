import  numpy as np , zlib
from numpy.typing import NDArray

BITS_PER_BYTE = 8

def pad_bits2bytes ( bits : NDArray[ np.uint8 ] ) -> NDArray[ np.uint8 ] :
    # dopełnij do pełnych bajtów (z prawej zerami) i spakuj
    pad = ( -len ( bits ) ) % BITS_PER_BYTE
    if pad :
        bits = np.concatenate ( [ bits , np.zeros ( pad , dtype = np.uint8 ) ] )
    return np.packbits ( bits )

def create_crc32_bytes ( bytes : NDArray[ np.uint8 ] ) -> NDArray[ np.uint8 ] :
    crc32 = zlib.crc32 ( bytes )
    return np.frombuffer ( crc32.to_bytes ( 4 , 'big' ) , dtype = np.uint8 )

def bits_2_int ( bits : np.ndarray ) -> int:
    """
    Zamienia tablicę bitów (najstarszy bit pierwszy) na wartość dziesiętną,
    używając operacji bitowych.

    Parametry:
    -----------
    bits : np.ndarray
        Tablica bitów (0/1) typu np.int64 lub podobnego, maks. 16 bitów.

    Zwraca:
    --------
    int
        Wartość dziesiętna odpowiadająca zakodowanym bitom.
    """

    if not isinstance ( bits , np.ndarray ) :
        raise TypeError ( "Argument musi być typu numpy.ndarray." )
    if not np.all ( (bits == 0 ) | ( bits == 1 ) ) :
        raise ValueError ( "Tablica może zawierać tylko wartości 0 i 1." )
    result = 0
    for bit in bits :
        result = ( result << 1 ) | int ( bit )
    return result