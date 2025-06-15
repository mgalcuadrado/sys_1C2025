import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft
import parte1.tpe_parte1 as tpe

# Lectura de archivos de audio WAV
picasso_hombre_lento = 'Picasso_hombre_5s.wav'
picasso_hombre_rapido = 'Picasso_hombre_3s.wav'

sample_rate_hl, audio_data_hl = wavfile.read(picasso_hombre_lento)
sample_rate_hr, audio_data_hr = wavfile.read(picasso_hombre_rapido)


fragmento_a_cuasi_periodicos_l = tpe.seleccionar_fragmento(audio_data_hl, 2.40, 2.42, sample_rate_hl)
tpe.graficar_fragmento(fragmento_a_cuasi_periodicos_l, sample_rate_hl, 0, 'Segmentos cuasi-periodicos letra "A" - Señal lenta')

'''
# Análisis para la señal rapida
fragmento_a_rapido_completo = seleccionar_fragmento(audio_data_r, 1.4, 2, sample_rate_r)
fragmento_s_rapido_completo = seleccionar_fragmento(audio_data_r, 2, 2.5, sample_rate_r)
graficar_fragmento(fragmento_a_rapido_completo, sample_rate_r, 0, 'Fragmento letra "A" - Señal rápida')
graficar_fragmento(fragmento_s_rapido_completo, sample_rate_r, 0, 'Fragmento letra "S" - Señal rápida')
'''
fragmento_a_cuasi_periodicos_r = tpe.seleccionar_fragmento(audio_data_hr, 1.60, 1.62, sample_rate_hr)
tpe.graficar_fragmento(fragmento_a_cuasi_periodicos_r, sample_rate_hr, 0, 'Segmentos cuasi-periodicos letra "A" - Señal rápida')