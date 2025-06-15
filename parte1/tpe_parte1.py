import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft



# =================================== FUNCIONES AUXILIARES PARTE 1 ======================================
def seleccionar_fragmento(audio_data, start_time, end_time, sample_rate):
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    return audio_data[start_sample:end_sample]


def graficar_fragmento(audio_fragment, sample_rate, start_time, title, png_name):
    fragment_time = np.arange(audio_fragment.size) / sample_rate + start_time
    plt.figure(figsize=(12, 6))
    #plt.rcParams['font.family'] = 'Times New Roman'
    plt.plot(fragment_time, audio_fragment)
    plt.title(title, fontsize=17)
    plt.grid()
    plt.xlabel('Tiempo [s]', fontsize=15)
    plt.ylabel('Amplitud', fontsize=15)
    plt.savefig(png_name)
    plt.show()
    plt.close()
    


def realizar_fft(audio_fragment, sample_rate):
    fft_result = fft(audio_fragment)
    return np.fft.fftfreq(audio_fragment.size, 1 / sample_rate), np.abs(fft_result)


def graficar_frecuencia(frequencies, fft_magnitude, n, title, freq_max, png_name):
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies[:n // 2], fft_magnitude[:n // 2])  # Solo graficamos la mitad positiva
    plt.title(title)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud')
    plt.grid()
    plt.xlim(0, freq_max)
    plt.savefig(png_name)
    plt.show()
    plt.close()

def procesamiento_vocales_fft(audio_data, tiempo_inicio, tiempo_fin, sample_rate, titulo_tiempo, titulo_freq, limite_freq, png_name_tiempo, png_name_freq):
    signal_tiempo = seleccionar_fragmento(audio_data, tiempo_inicio, tiempo_fin, sample_rate)
    graficar_fragmento(signal_tiempo, sample_rate, 0, titulo_tiempo, png_name_tiempo)
    signal_freq, signal_amplitud = realizar_fft(signal_tiempo, sample_rate)
    graficar_frecuencia(signal_freq, signal_amplitud, signal_tiempo.size, titulo_freq, limite_freq, png_name_freq)



# ===================================== PROCESAMIENTO DE ARCHIVOS =======================================
# Lectura de archivos de audio WAV
picasso_hombre_lento = 'Picasso_hombre_5s.wav'
picasso_hombre_rapido = 'Picasso_hombre_3s.wav'

sample_rate_hl, audio_data_hl = wavfile.read(picasso_hombre_lento)
sample_rate_hr, audio_data_hr = wavfile.read(picasso_hombre_rapido)

# ==================================== PARTE 1: GRAFICO DE SEÑALES COMPLETAS ============================
graficar_fragmento(audio_data_hl, sample_rate_hl, 0, 'Picasso (Hombre - 5s)', 'picasso_hl_completa.png')
graficar_fragmento(audio_data_hr, sample_rate_hr, 0, 'Picasso (Hombre - 3s)', 'picasso_hl_completa.png')


# ==================================== PARTE 2: GRAFICO DE FRAGMENTOS DE LETRAS "a" Y "s" ============================

#Fonema /a/
picasso_hl_a = seleccionar_fragmento(audio_data_hl, 2.4, 3.0, sample_rate_hl)
graficar_fragmento(picasso_hl_a, sample_rate_hl, 0, 'Picasso (Hombre - 5s): Señal correspondiente a la letra "a"', 'picasso_hl_a.png')

#Fonema /s/
picasso_hl_s = seleccionar_fragmento(audio_data_hl, 3.4, 3.9, sample_rate_hl)
graficar_fragmento(picasso_hl_s, sample_rate_hl, 0, 'Picasso (Hombre - 5s): Señal correspondiente a la letra "s"', 'picasso_hl_s.png')

# ==================================== PARTE 3: PROCESAMIENTO DE VOCALES  ========================

# ==================================== PROCESAMIENTO DE VOCAL 1 = "i" ========================

## Hombre - lento
### Considerando múltiples periodos
procesamiento_vocales_fft(audio_data_hl, 1.2, 1.6, sample_rate_hl, 'Picasso (Hombre - 5s): Vocal "i" - Múltiples periodos', 'Picasso (Hombre - 5s): Espectro de frecuencia de la vocal "i" considerando múltiples periodos', 3500, 'fragmento_i_hl_mT.png', 'fft_i_hl_mT.png')
### Considerando un periodo
procesamiento_vocales_fft(audio_data_hl, 1.31, 1.3202, sample_rate_hl, 'Picasso (Hombre - 5s): Vocal "i" - Un solo periodo', 'Picasso (Hombre - 5s): Espectro de frecuencia de la vocal "i" considerando un periodo', 3500, 'fragmento_i_hl_1T.png', 'fft_i_hl_1T.png')

## Hombre - rápido
### Considerando múltiples periodos
procesamiento_vocales_fft(audio_data_hr, 0.6, 0.8, sample_rate_hr, 'Picasso (Hombre - 3s): Vocal "i" - Múltiples periodos', 'Picasso (Hombre - 3s): Espectro de frecuencia de la vocal "i" considerando múltiples periodos', 3500, 'fragmento_i_hr_mT.png', 'fft_i_hr_mT.png')

### Considerando un periodo
procesamiento_vocales_fft(audio_data_hr, 0.6103, 0.6133, sample_rate_hr, 'Picasso (Hombre - 3s): Vocal "i" - Un solo periodo', 'Picasso (Hombre - 3s): Espectro de frecuencia de la vocal "i" considerando un periodo', 3500, 'fragmento_i_hr_1T.png', 'fft_i_hr_1T.png')


# ==================================== PROCESAMIENTO DE VOCAL 2 = "a" ========================
## Hombre - lento
### Considerando múltiples periodos
procesamiento_vocales_fft(audio_data_hl, 2.40, 2.42, sample_rate_hl, 'Picasso (Hombre - 5s): Vocal "a" - Múltiples periodos', 'Picasso (Hombre - 5s): Espectro de frecuencia de la vocal "a" considerando múltiples periodos', 3500, 'fragmento_a_hl_mT.png', 'fft_a_hl_mT.png')
### Considerando un periodo
procesamiento_vocales_fft(audio_data_hl, 2.4025, 2.4097, sample_rate_hl, 'Picasso (Hombre - 5s): Vocal "a" - Un solo periodo', 'Picasso (Hombre - 5s): Espectro de frecuencia de la vocal "a" considerando un periodo', 3500, 'fragmento_a_hl_1T.png', 'fft_a_hl_1T.png')

## Hombre - rápido
### Considerando múltiples periodos
procesamiento_vocales_fft(audio_data_hr, 1.60, 1.62, sample_rate_hr, 'Picasso (Hombre - 3s): Vocal "a" - Múltiples periodos', 'Picasso (Hombre - 3s): Espectro de frecuencia de la vocal "a" considerando múltiples periodos', 3500, 'fragmento_a_hr_mT.png', 'fft_a_hr_mT.png')

### Considerando un periodo
procesamiento_vocales_fft(audio_data_hr, 1.6012, 1.6091, sample_rate_hr, 'Picasso (Hombre - 3s): Vocal "a" - Un solo periodo', 'Picasso (Hombre - 3s): Espectro de frecuencia de la vocal "a" considerando un periodo', 3500, 'fragmento_a_hr_1T.png', 'fft_a_hr_1T.png')


# ==================================== PROCESAMIENTO DE VOCAL 3 = "o" ========================

## Hombre - lento
### Considerando múltiples periodos
procesamiento_vocales_fft(audio_data_hl, 4.305, 4.379, sample_rate_hl, 'Picasso (Hombre - 5s): Vocal "o" - Múltiples periodos', 'Picasso (Hombre - 5s): Espectro de frecuencia de la vocal "o" considerando múltiples periodos', 3500, 'fragmento_o_hl_mT.png', 'fft_o_hl_mT.png')
### Considerando un periodo
procesamiento_vocales_fft(audio_data_hl, 4.305, 4.3155, sample_rate_hl, 'Picasso (Hombre - 5s): Vocal "o" - Un solo periodo', 'Picasso (Hombre - 5s): Espectro de frecuencia de la vocal "o" considerando un periodo', 3500, 'fragmento_o_hl_1T.png', 'fft_o_hl_1T.png')

## Hombre - rápido
### Considerando múltiples periodos
procesamiento_vocales_fft(audio_data_hr, 0.26, 0.32, sample_rate_hr, 'Picasso (Hombre - 3s): Vocal "o" - Múltiples periodos', 'Picasso (Hombre - 3s): Espectro de frecuencia de la vocal "o" considerando múltiples periodos', 3500, 'fragmento_o_hr_mT.png', 'fft_o_hr_mT.png')

### Considerando un periodo
procesamiento_vocales_fft(audio_data_hr, 0.2626, 0.2709, sample_rate_hr, 'Picasso (Hombre - 3s): Vocal "o" - Un solo periodo', 'Picasso (Hombre - 3s): Espectro de frecuencia de la vocal "o" considerando un periodo', 3500, 'fragmento_o_hr_1T.png', 'fft_o_hr_1T.png')


