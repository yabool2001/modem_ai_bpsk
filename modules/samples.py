@dataclass ( slots = True , eq = False )
class RxSamples_v0_1_18 :

    # Pola uzupełnianie w __post_init__
    samples : NDArray[ np.complex128 ] = field ( init = False )
    y_train_tensor : torch.Tensor = field ( init = False )
    #samples : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    samples_filtered : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    samples_corrected : NDArray[ np.complex128 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.complex128 ) , init = False )
    has_amp_greater_than_ths : bool = False
    SPS = modulation.SPS
    SPAN = filters.SPAN
    ths : float = 1000.0
    frames : list[ RxFrame_v0_1_18 ] = field ( init = False , default_factory = list )
    leftovers : NDArray[ np.complex128 ] | None = field ( default = None )

    samples_corrected_len : np.uint32 = field ( init = False )
    sync_sequence_peaks : NDArray[ np.uint32 ] = field ( default_factory = lambda : np.array ( [] , dtype = np.uint32 ) , init = False )
    has_leftovers : bool = False
    leftovers_start_idx : np.uint32 = field ( init = False )

    def __post_init__ ( self ) -> None :
            self.samples = np.array ( [] , dtype = np.complex128 )
            #self.samples_filtered = np.array ( [] , dtype = np.complex128 )
            #self.samples_corrected = np.array ( [] , dtype = np.complex128 )

    def rx ( self , sdr_ctx : Pluto  | None = None , previous_samples_leftovers : NDArray[ np.complex128 ] | None = None , samples_filename : str | None = None , concatenate : bool = False ) -> None :
        '''
        concatenated: powoduje nawarstwienie nowych sampli na stare.
        UWAGA! Nieostrożne stosowanie może spowodować zawieszenie komputera z powodu braku pamięci RAM,
        jeśli próbujemy nawarstwić zbyt dużo sampli.
        Używaj z rozwagą i monitoruj zużycie pamięci.
        '''
        if sdr_ctx is not None :
            samples = sdr_ctx.rx ()
        elif samples_filename is not None :
            if samples_filename.endswith('.npy'):
                samples = ops_file.open_samples_from_npf ( samples_filename )
            elif samples_filename.endswith('.csv'):
                samples = ops_file.open_csv_and_load_np_complex128 ( samples_filename )
            else:
                raise ValueError(f"Error: unsupported file format for {samples_filename}! Supported formats: .npy, .csv")
        else :
            raise ValueError ( "Either sdr_ctx or samples_filename must be provided." )
        if concatenate :
            self.samples = np.concatenate ( [ self.samples , samples ] )
        else :
            self.samples = samples
        if previous_samples_leftovers is not None :
            self.samples = np.concatenate ( [ previous_samples_leftovers , self.samples ] )
        self.sample_initial_assesment ()

    def sample_initial_assesment (self) -> None :
        self.has_amp_greater_than_ths = np.any ( np.abs ( self.samples ) > self.ths )

    def detect_frames ( self , deep : bool = False , filter : bool = False , correct : bool = False , add_peak_at_0 : bool = False ) -> None :
        if filter :
            self.filter_samples ()
        else :
            self.samples_filtered = self.samples
        if correct :
            self.correct_samples ()
        else :
            self.samples_corrected = self.samples_filtered
        self.has_leftovers = False
        self.samples_corrected_len = np.uint32 ( len ( self.samples_corrected ) )
        self.sync_sequence_peaks = detect_sync_sequence_peaks_v0_1_15 ( self.samples_corrected , modulation.generate_barker13_bpsk_samples_v0_1_7 ( True ) , deep = deep )
        if add_peak_at_0 : self.sync_sequence_peaks = np.insert ( self.sync_sequence_peaks , 0 , 0 )
        previous_processed_idx : np.uint32 = 0
        for idx in self.sync_sequence_peaks :
            if idx > previous_processed_idx :
                frame = RxFrame_v0_1_18 ( samples_filtered = self.samples_corrected [ idx : ] , sync_sequence_peak_abs_idx = idx )
                if frame.has_frame :
                    self.frames.append ( frame )
                    previous_processed_idx = frame.frame_end_abs_idx
                else :
                    previous_processed_idx = idx
                if frame.has_leftovers :
                    self.has_leftovers = True
                    self.leftovers_start_idx = frame.leftovers_start_abs_idx
                    break
        if not self.has_leftovers :
            self.leftovers_start_idx = self.samples_corrected_len - SYNC_SEQUENCE_LEN_SAMPLES - self.SPAN * self.SPS // 2
        self.clip_samples_leftovers ()
        self.y_train_tensor = self.y_train_tensor_from_frames ()

    def clip_samples_for_training ( self ) -> None :
        '''Przycinanie ramki aby stosunek symboli BPSK do 0+j0 był ok. 80 do 20, co pomaga w treningu modelu.
        Nie powinno to być nigdy idealny 80/20, bo w rzeczywistych danych zawsze będzie pewna losowość, ale powinno być blisko tego.
        Poza tym należy dabć o to aby liczba sampli po przycięciu była wielokrotnością SPS i ml.CHUNK_SAMPLES_LEN.'''
        i = ml.CHUNK_SAMPLES_LEN * 10 # mnożnik ma na celu niedopuszczenie do zbyt wysokiego ratio, stosunku symboli BPSK do 0+j0
        total_bpsk_symbols = 0
        first_bpsk_symbol_idx = self.frames[ 0 ].frame_start_abs_idx
        last_bpsk_symbol_idx = self.frames[ -1 ].frame_end_abs_idx
        leftovers_start_idx = self.leftovers_start_idx
        total_bpsk_symbols = sum ( frame.header_bpsk_symbols.size + frame.packet.bpsk_symbols.size for frame in self.frames )
        ratio : float = total_bpsk_symbols / self.samples_corrected_len
        clip1 = ( ( first_bpsk_symbol_idx - 1 ) // i ) * i
        clip2 = ( last_bpsk_symbol_idx // i + 1) * i
        # Clamping (zapewnienie że nie wyskoczymy poza zakres indeksowania arrayu)
        clip1 = np.maximum ( 0 , clip1 )
        clip2 = np.minimum ( self.samples_corrected_len , clip2 )
        ratio_clipped = total_bpsk_symbols / ( clip2 - clip1 )
        print ( f"{clip1=} , {clip2=} , {ratio=:.2f} , {ratio_clipped=:.2f}" )
        clipped_samples = self.clip_samples_corrected ( self.samples_corrected , clip1 , clip2 )
        self.reset_frame_detection ()
        self.samples = clipped_samples
        self.sample_initial_assesment ()
        # Poniższa konfiguracja argumentów ma zapewnić, że funkcja detect_frames () będzie działać poprawnie na przyciętych, filtrowanych i po korekcji samplach,
        # bez ponownego filtrowania i korygowania.
        self.detect_frames ( deep = False , filter = False , correct = False )

    def reset_frame_detection ( self ) -> None :
        self.samples = self.samples_filtered = self.samples_corrected = np.array ( [] , dtype = np.complex128 )
        self.frames = []
        self.leftovers = None
        self.has_leftovers = False
        self.leftovers_start_idx = np.uint32 ( 0 )
        self.y_train_tensor = torch.tensor ( [] )
        self.has_amp_greater_than_ths = False
        self.samples_corrected_len = np.uint32 ( 0 )
        self.sync_sequence_peaks = np.array ( [] , dtype = np.uint32 )
        self.has_leftovers = False
        self.leftovers_start_idx = np.uint32 ( 0 )

    def filter_samples ( self ) -> None :
        self.samples_filtered = filters.apply_rrc_rx_convolve_v0_1_18 ( self.samples )

    def correct_samples ( self ) -> None :
        self.samples_corrected = modulation.zero_quadrature ( corrections.full_compensation_v0_1_5 ( self.samples_filtered , modulation.generate_barker13_bpsk_samples_v0_1_7 ( True ) ) )

    def plot_complex_samples ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples , f"RxSamples {title} {self.samples.size=}" , marker_squares = marker , marker_peaks = peaks )

    def plot_complex_samples_filtered ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples_filtered , f"RxSamples filtered {title} {self.samples_filtered.size=}" , marker_squares = marker , marker_peaks = peaks )

    def plot_tensor ( self , title : str = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None , frame_idx : int | None = None ) -> None :
        plot.tensor_waveform_v0_1_16 ( self.y_train_tensor , title = f"y_train_tensor {title}" , marker_squares = marker , marker_peaks = peaks , frame_idx = frame_idx )

    def plot_flat_tensor ( self , title : str = "" ) -> None :
        frames_start_idx = np.array ( [ frame.frame_start_abs_idx for frame in self.frames ] , dtype = np.uint32 )
        flat_tensor = self.flat_tensor_from_y_train ()
        plot.tensor_waveform_v0_1_16 ( flat_tensor , title = f"RxSamples y_train_tensor {title}" , marker_peaks = frames_start_idx )

    def plot_complex_samples_corrected_v0_1_20 ( self , title = "" , marker : bool = False ) -> None :
        frames_start_idx = np.array ( [ frame.frame_start_abs_idx for frame in self.frames ] , dtype = np.uint32 )
        plot.complex_waveform_v0_1_6 ( self.samples_corrected , f"RxSamples corrected {title} {self.samples_corrected.size=}" , marker_squares = marker , marker_peaks = frames_start_idx )

    def plot_complex_samples_corrected ( self , title = "" , marker : bool = False , peaks : NDArray[ np.uint32 ] = None ) -> None :
        plot.complex_waveform_v0_1_6 ( self.samples_corrected , f"RxSamples corrected {title} {self.samples_corrected.size=}" , marker_squares = marker , marker_peaks = peaks )

    def save_complex_samples2npf_v0_1_18 ( self , file_name : str , dir_name : str , add_timestamp : bool = True ) -> None :
        filename = ops_file.add_timestamp_2_filename ( file_name ) if add_timestamp else file_name
        filename_and_dirname = f"{dir_name}/{filename}"
        ops_file.save_complex_samples_2_npf ( filename_and_dirname , self.samples )

    def save_complex_samples_2_npf ( self , file_name : str , dir_name : str ) -> None :
        filename_with_timestamp = ops_file.add_timestamp_2_filename ( file_name )
        filename_with_timestamp_and_dir = f"{dir_name}/{filename_with_timestamp}"
        ops_file.save_complex_samples_2_npf ( filename_with_timestamp_and_dir , self.samples )

    def save_complex_samples_2_csv ( self , file_name : str ) -> None :
        filename_with_timestamp = ops_file.add_timestamp_2_filename ( file_name )
        ops_file.save_complex_samples_2_csv ( filename_with_timestamp , self.samples )

    def analyze ( self ) -> None :
        sdr.analyze_rx_signal ( self.samples )

    def clip_samples ( self , start : np.uint32 , end : np.uint32 ) -> None :
        if start < 0 or end > ( self.samples.size - 1 ) :
            raise ValueError ( "Start must be >= 0 & end <= samples length" )
        if start >= end :
            raise ValueError ( "Start must be < end" )
        #self.samples_filtered = self.samples_filtered [ start : end + 1 ]
        self.samples = self.samples [ start : end ]

    def clip_samples_filtered ( self , start : np.uint32 , end : np.uint32 ) -> None :
        if start < 0 or end > ( self.samples_filtered.size - 1 ) :
            raise ValueError ( "Start must be >= 0 & end <= samples_filtered length" )
        if start >= end :
            raise ValueError ( "Start must be < end" )
        #self.samples_filtered = self.samples_filtered [ start : end + 1 ]
        self.samples_filtered = self.samples_filtered [ start : end ]

    def clip_samples_corrected ( self , samples : NDArray[ np.complex128 ] , start : np.uint32 , end : np.uint32 ) -> NDArray[ np.complex128 ] :
        if start < 0 or end > ( samples.size - 1 ) :
            raise ValueError ( "Start must be >= 0 & end <= samples length" )
        if start >= end :
            raise ValueError ( "Start must be < end" )
        return samples [ start : end ]

    def clip_samples_leftovers ( self ) -> None :
        self.leftovers = self.samples [ self.leftovers_start_idx : ]

    def symbols_2_flat_tensor ( self ) -> torch.Tensor :
        # Złożenie wszystkich bpsk_symbols z wszystkich ramek w jeden strumień w kolejności wysyłania we flat tensor w celu porównania z tensorami zapisany podczas tx
        # Tutaj są tylko tensory wszystkich ramek bez ich pozycji i bez tensorów 0+j0 dla sampli bez ramek.
        if self.frames :
            all_symbols = np.concatenate (
                [
                    symbols
                    for frame in self.frames
                    for symbols in ( frame.header_bpsk_symbols , frame.packet.bpsk_symbols )
                ]
            ).astype ( np.complex64 , copy = False )
        else :
            all_symbols = np.array ( [] , dtype = np.complex64 )
        return torch.from_numpy ( all_symbols )

    def y_train_tensor_from_frames ( self ) -> torch.Tensor :
        y_train_symbols = np.zeros ( self.samples.size , dtype = np.complex64 )
        for frame in self.frames :
            frame_symbols = np.concatenate ( [ frame.header_bpsk_symbols , frame.packet.bpsk_symbols ] ).astype ( np.complex64 , copy = False )
            frame_start_idx = int ( frame.frame_start_abs_idx )
            if frame_start_idx >= y_train_symbols.size or frame_symbols.size == 0 :
                continue
            frame_end_idx = min ( frame_start_idx + frame_symbols.size , y_train_symbols.size )
            y_train_symbols[ frame_start_idx : frame_end_idx ] = frame_symbols[ : frame_end_idx - frame_start_idx ]
        return torch.from_numpy ( y_train_symbols )

    def flat_tensor_from_y_train ( self ) -> torch.Tensor :
        if not isinstance ( self.y_train_tensor , torch.Tensor ) or not torch.is_complex ( self.y_train_tensor ) :
            raise TypeError ( "y_train_tensor must be a complex torch.Tensor" )
        return torch.stack ( [ self.y_train_tensor.real , self.y_train_tensor.imag ] )

    def save_frames2y_train_tensor ( self , file_name : str , dir_name : str ) -> None :
        if not isinstance ( self.y_train_tensor , torch.Tensor ) or self.y_train_tensor.dtype != torch.complex64 :
            raise TypeError ( "y_train_tensor must be torch.Tensor with dtype=torch.complex64" )
        tensor_filename = f"{dir_name}/{file_name}.pt"
        Path ( dir_name ).mkdir ( parents = True , exist_ok = True )
        torch.save ( self.y_train_tensor , tensor_filename )

    def __repr__ ( self ) -> str :
        for frame in self.frames :
            frame_bits = modulation.bpsk_symbols_2_bits_v0_1_7 ( np.concatenate ( [ frame.header_bpsk_symbols[ : : self.SPS ] , frame.packet.bpsk_symbols[ : : self.SPS ] ] ) )
            print ( f"{ frame_bits.size=}, {frame.frame_start_abs_idx=}, {frame_bits[ : 10 ]}" )
        return ( f"{self.samples.size=}, {self.samples.dtype=}")

