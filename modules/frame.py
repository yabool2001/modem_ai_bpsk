import numpy as np , os , tomllib
from dataclasses import dataclass , field
from modules import filters , modulation, ops_data , plot
from numpy.typing import NDArray

script_filename = os.path.basename ( __file__ )

# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    settings = tomllib.load ( settings_file )

BARKER13_BITS = np.array ( settings[ "BARKER13_BITS" ] , dtype = np.uint8 )
SPS = modulation.SPS
BETA = filters.BETA
SPAN = filters.SPAN
SAMPLES_PER_BYTE = SYMBOLS_PER_BYTE = ops_data.BITS_PER_BYTE * SPS
PACKET_LEN_BITS_LEN = 11
PACKET_LEN_SAMPLES_LEN = PACKET_LEN_SYMBOLS_LEN = PACKET_LEN_BITS_LEN * SPS
CRC32_BITS_LEN = 32
CRC32_SAMPLES_LEN = CRC32_SYMBOLS_LEN = CRC32_BITS_LEN * SPS
SYNC_SEQUENCE_BITS_LEN = len ( BARKER13_BITS )
SYNC_SEQUENCE_SAMPLES_LEN = SYNC_SEQUENCE_SYMBOLS_LEN = SYNC_SEQUENCE_BITS_LEN * SPS
FRAME_BITS_LEN = SYNC_SEQUENCE_BITS_LEN + PACKET_LEN_BITS_LEN + CRC32_BITS_LEN
FRAME_SYMBOLS_LEN = SYNC_SEQUENCE_SYMBOLS_LEN + PACKET_LEN_SYMBOLS_LEN + CRC32_SYMBOLS_LEN
FRAME_SAMPLES_LEN = SYNC_SEQUENCE_SAMPLES_LEN + PACKET_LEN_SAMPLES_LEN + CRC32_SAMPLES_LEN


@dataclass ( slots = True , eq = False )
class RxPayload :

    symbols : NDArray[ np.complex64 ]
    has_payload : bool = False
    bits : NDArray[ np.uint8 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint8 ) , init = False )
    bytes : NDArray[ np.uint8 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint8 ) , init = False )
    data_bytes : NDArray[ np.uint8 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint8 ) , init = False )
    # Pola uzupełnianie w __post_init__

    def __post_init__ ( self ) -> None :
        self.process_symbols ()
        return
    
    def process_symbols ( self ) -> None :
        data_end_idx = len ( self.symbols ) - ( CRC32_BITS_LEN * SPS )
        data_symbols = self.symbols [ : data_end_idx : SPS ]
        crc32_symbols = self.symbols [ data_end_idx : : SPS ]
        data_bits = modulation.bpsk_symbols_2_bits ( data_symbols )
        data_bytes = ops_data.pad_bits2bytes ( data_bits )
        crc32_bits = modulation.bpsk_symbols_2_bits ( crc32_symbols )
        crc32_bytes_read = ops_data.pad_bits2bytes ( crc32_bits )
        crc32_bytes_calculated = ops_data.create_crc32_bytes ( data_bytes )
        if ( crc32_bytes_read == crc32_bytes_calculated ).all () :
            self.has_payload = True
            self.bits = np.concatenate ( [ data_bits , crc32_bits ] )
            self.bytes = np.concatenate ( [ data_bytes , crc32_bytes_read ] )
            self.data_bytes = data_bytes
        return

    def plot_bpsk_complex_symbols ( self , title = "Payload BPSK complex symbols" , markers : bool = False , idxs : NDArray[ np.uint32 ] = None ) -> None :
        plot.bpsk_complex_symbols ( self.symbols , f"{title} {self.symbols.size=}" )
        plot.bpsk_complex_symbols_test ( self.symbols , f"TEST {title} {self.symbols.size=}" )

    def __repr__ ( self ) -> str :
        return ( f"{self.symbols.size=} , {self.has_payload=} , {self.bytes.size=}" )

@dataclass ( slots = True , eq = False )
class RxFrame :
    
    symbols : NDArray[ np.complex64 ]
    frame_start_abs_idx : np.uint32

    # Pola uzupełnianie w __post_init__
    header_bpsk_symbols : NDArray[ np.complex64 ] = field ( init = False )
    frame_end_abs_idx : np.uint32 = field ( init = False )
    payload_start_abs_idx : NDArray[ np.uint32 ] = field ( init = False )
    leftovers_start_abs_idx : np.uint32 = field ( init = False )
    has_frame : bool = False # ustawiany dopiero po walidacji pakietu, wcześniej używamy tylko lokalnego has_frame_header
    has_leftovers : bool = False
    # do zapamiętania jako tip przed skasowaniem payload_bytes : NDArray[ np.uint8 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint8 ) , init = False )
    payload : RxPayload = field ( init = False )
    
    def __post_init__ ( self ) -> None :
        if not self.frame_len_validation () :
            return self.frame_start_abs_idx
        self.process_packet ()
    
    def process_packet ( self ) -> None :
        sync_sequence_start_idx = 0
        sync_sequence_end_idx = sync_sequence_start_idx + SYNC_SEQUENCE_SAMPLES_LEN
        packet_len_start_idx = sync_sequence_end_idx
        packet_len_end_idx = packet_len_start_idx + PACKET_LEN_SAMPLES_LEN
        crc32_start_idx = packet_len_end_idx
        crc32_end_idx : np.uint32 = np.uint32 ( crc32_start_idx + CRC32_SAMPLES_LEN )

        samples_components = [ ( self.samples_filtered.real , "sync sequence real" ) , ( self.samples_filtered.imag , "sync sequence imag" ) , ( -self.samples_filtered.real , "sync sequence -real" ) , ( -self.samples_filtered.imag , "sync sequence -imag" ) ]
        for samples_component , samples_name in samples_components :
            sync_sequence_symbols = samples_component [ sync_sequence_start_idx : sync_sequence_end_idx : SPS ]
            sync_sequence_bits = modulation.bpsk_symbols_2_bits ( sync_sequence_symbols )
            if np.array_equal ( sync_sequence_bits , BARKER13_BITS ) :
                has_sync_sequence = True
                packet_len_symbols = samples_component [ packet_len_start_idx : packet_len_end_idx : SPS ]
                packet_len_bits = modulation.bpsk_symbols_2_bits ( packet_len_symbols )
                packet_len_uint16 = self.bits2uint16 ( packet_len_bits )
                check_components = [ ( self.samples_filtered.real , " frame real" ) , ( self.samples_filtered.imag , " frame imag" ) , ( -self.samples_filtered.real , " frame -real" ) , ( -self.samples_filtered.imag , " frame -imag" ) ]
                for samples_comp , frame_name in check_components :
                    crc32_symbols = samples_comp [ crc32_start_idx : crc32_end_idx : SPS ]
                    crc32_bits = modulation.bpsk_symbols_2_bits ( crc32_symbols )
                    crc32_bytes_read = ops_data.pad_bits2bytes ( crc32_bits )
                    crc32_bytes_calculated = ops_data.create_crc32_bytes ( ops_data.pad_bits2bytes ( np.concatenate ( [ sync_sequence_bits , packet_len_bits ] ) ) )
                    if ( crc32_bytes_read == crc32_bytes_calculated ).all () :
                        payload_end_idx = crc32_end_idx + ( packet_len_uint16 * SAMPLES_PER_BYTE )
                        has_frame_header = True
                        if not self.packet_len_validation ( payload_end_idx ) :
                            if settings["log"]["verbose_2"] : print ( f"{self.sync_sequence_peak_abs_idx=} {samples_name} {frame_name=} {has_sync_sequence=}, {has_frame_header=}" )
                            return
                        payload = RxPayload ( symbols = self.symbols [ crc32_end_idx : payload_end_idx ] )
                        if payload.has_packet :
                            self.has_frame = True
                            self.header_bpsk_symbols = modulation.bits_2_bpsk_symbols ( np.concatenate ( [ sync_sequence_bits , packet_len_bits , crc32_bits ] ) , sps = SPS )
                            self.payload = payload
                            self.payload_start_abs_idx = self.frame_start_abs_idx + crc32_end_idx
                            self.frame_end_abs_idx = self.frame_start_abs_idx + payload_end_idx # to może być tylko wtedy kiedy mamy poprawny pakiet, bo inaczej nie wiemy, czy i gdzie się kończy ramka, a bez tego nie możemy poprawnie ustawić leftoversów
                            if settings["log"]["verbose_2"] : print ( f"{sync_sequence_start_idx=} {has_sync_sequence=}, {self.frame_start_abs_idx=} {self.has_frame=}, {payload.has_packet=}" )
                            return
        if settings["log"]["verbose_2"] : print ( f"{self.frame_sync_sequence_peak_abs_idx=} {has_sync_sequence=}, {self.has_frame=}" )
        return
    
    def samples2bits ( self , samples : NDArray[ np.complex128 ] ) -> NDArray[ np.uint8 ] :
        return modulation.bpsk_symbols_2_bits_v0_1_7 ( samples [ : : self.sps ] )

    def bits2uint16 ( self , bits : NDArray[ np.uint8 ] ) -> np.uint16 :
        return np.uint16 ( ops_data.bits_2_int ( bits ) )

    def samples2bytes ( self , samples : NDArray[ np.complex128 ] ) -> NDArray[ np.uint8 ] :
        bits = self.samples2bits ( samples )
        return ops_data.pad_bits2bytes ( bits )
    
    def set_leftovers_idx_for_incomplete_frame ( self ) -> None :
        if settings["log"]["verbose_2"] : print ( f"Samples at index { self.frame_start_abs_idx } is too close to the end of samples to contain a complete frame. Skipping." )
        self.leftovers_start_abs_idx = self.frame_start_abs_idx - SPAN * SPS // 2 # Bez cofniecia się do początku filtra RRC nie ma wykrycia ramnki i pakietu w następnym wywołaniu
        self.has_leftovers = True

    def frame_len_validation ( self ) -> bool :
        if np.uint32 ( self.symbols.size ) <= np.uint32 ( FRAME_SAMPLES_LEN ) :
            self.set_leftovers_idx_for_incomplete_frame ()
            return False
        return True

    def packet_len_validation ( self , packet_end_idx : np.uint32 ) -> bool :
        if packet_end_idx > np.uint32 ( self.samples_filtered.size ) :
            self.set_leftovers_idx_for_incomplete_frame ()
            return False
        return True

    def plot_complex_samples_filtered ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples_filtered , f"{title} {self.samples_filtered.size=}" , marker_squares = marker , marker_peaks = peaks )

    def __repr__ ( self ) -> str :
        return ( f"{self.packet=}, {self.has_frame=}, {self.has_leftovers=}" )

