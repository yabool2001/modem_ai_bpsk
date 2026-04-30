import numpy as np , pandas as pd , plotly.express as px
from numpy.typing import NDArray

def bpsk_complex_symbols_test ( symbols: NDArray[ np.complex64 ] , title: str = "Symbole BPSK" ) -> None:
    """
    Rysuje wykres symboli BPSK (np. z ADALM-Pluto) w postaci punktów połączonych przerywaną linią.

    Parametry:
    ----------
    symbols : np.ndarray
        Tablica symboli BPSK, może być typu float, int, complex (np. complex64).
    title : str
        Tytuł wykresu.
    filename : str
        Nazwa pliku źródłowego (opcjonalna dekoracja w tytule).

    Zwraca:
    -------
    None
    """
    if not isinstance ( symbols , np.ndarray ):
        raise TypeError ( "Argument 'symbols' musi być typu numpy.ndarray." )

    # Obsługa symboli zespolonych – bierzemy część rzeczywistą
    symbols_real = symbols.real if np.iscomplexobj ( symbols ) else symbols

    # Przygotowanie danych do wykresu
    df = pd.DataFrame ( { "symbol_index" : np.arange ( len ( symbols_real ) ) , "symbol": symbols_real } )

    # Wykres punktowy
    fig = px.scatter ( df , x = "symbol_index" , y = "symbol" , title = f"{title}" , labels = { "symbol" : "Wartość symbolu" , "symbol_index" : "Indeks symbolu" } )

    # Dodanie przerywanej linii łączącej punkty
    fig.add_scatter ( x = df[ "symbol_index" ] , y = df[ "symbol" ] , mode = 'lines+markers' , name = 'Symbole BPSK' , line = dict ( color = 'gray' , width = 1 , dash = 'dot' ) )

    # Konfiguracja osi i wyglądu
    fig.update_layout ( height = 500 , xaxis = dict ( rangeslider_visible = True ) , legend = dict ( x = 0.01 , y = 0.99 ) )

    # Oś Y dopasowana dynamicznie, ale możesz wymusić np. range=[-1.5, 1.5] jeśli chcesz sztywną skalę
    fig.show ()

    return

def bpsk_complex_symbols ( symbols : NDArray[ np.complex64 ] , title : str = "" ) -> None :
    """
    Rysuje wykres symboli BPSK (np. z ADALM-Pluto) w postaci punktów połączonych przerywaną linią.

    Parametry:
    ----------
    symbols : NDArray[ np.complex128 ] Tablica symboli BPSK typu np.complex128
    title : str Tytuł wykresu.
    Zwraca: None
    """
    if not isinstance ( symbols , np.ndarray ):
        raise TypeError ( "Argument 'symbols' musi być typu NDArray[np.complex128]." )

    # Obsługa symboli zespolonych – bierzemy część rzeczywistą
    symbols_real = symbols.real if np.iscomplexobj ( symbols ) else symbols
    # Przygotowanie danych do wykresu
    df = pd.DataFrame ( { "symbol_index" : np.arange ( len ( symbols_real ) ) , "symbol" : symbols_real } )
    # Wykres punktowy
    fig = px.scatter ( df , x = "symbol_index" , y = "symbol" , title = f"{ title }" , labels = { "symbol" : "Wartość symbolu" , "symbol_index" : "Indeks symbolu" } )
    # Dodanie przerywanej linii łączącej punkty
    fig.add_scatter ( x = df[ "symbol_index" ] , y = df[ "symbol" ] , mode = 'lines+markers' , name = 'Symbole' , line = dict ( color = 'gray' , width = 1 , dash = 'dot' ) )
    # Konfiguracja osi i wyglądu
    fig.update_layout ( height = 500 , xaxis = dict ( rangeslider_visible = True ) , legend = dict ( x = 0.01 , y = 0.99 ) )
    # Oś Y dopasowana dynamicznie, ale możesz wymusić np. range=[-1.5, 1.5] jeśli chcesz sztywną skalę
    fig.show ()

    return