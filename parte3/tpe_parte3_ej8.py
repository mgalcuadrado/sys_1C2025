import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin, lfilter
import librosa
import soundfile as sf

import funciones_tpe as func

FACTOR = 2
RUTA_ARCHIVO_ORIGINAL_LENTO = "Picasso_hombre_5s.wav"
RUTA_ARCHIVO_ACELERADO_DECIMACION_Y_VENTANEO = "Picasso_hombre_x2_decimacion.wav"
RUTA_ARCHIVO_ACELERADO_LIBROSA = "Picasso_hombre_x2_librosa.wav"
RUTA_ARCHIVO_ORIGINAL_RAPIDO = "Picasso_hombre_3s.wav"

#### ACELERACIÓN DE LA SEÑAL ###
func.Acelerar_audio_con_librosa(RUTA_ARCHIVO_ORIGINAL_LENTO, RUTA_ARCHIVO_ACELERADO_LIBROSA, FACTOR)
func.Acelerar_audio_con_decimacion_y_ventaneo(RUTA_ARCHIVO_ORIGINAL_LENTO, RUTA_ARCHIVO_ACELERADO_DECIMACION_Y_VENTANEO, FACTOR)

## Obtención de los vectores de data de los audios
fs_original_lento, data_original_lento = wavfile.read(RUTA_ARCHIVO_ORIGINAL_LENTO)
fs_librosa, data_librosa = wavfile.read(RUTA_ARCHIVO_ACELERADO_LIBROSA)
fs_decimacion, data_decimacion = wavfile.read(RUTA_ARCHIVO_ACELERADO_DECIMACION_Y_VENTANEO)
fs_original_rapido, data_original_rapido = wavfile.read(RUTA_ARCHIVO_ORIGINAL_RAPIDO)

## Realización de espectrogramas completos
func.Graficar_espectrograma_banda_angosta(data_original_lento, fs_original_lento, None, None, None, None, "Espectrograma de la señal original", "espectrograma_original.png", None)
func.Graficar_espectrograma_banda_angosta(data_librosa, fs_librosa, None, None, None, None, "Espectrograma de la señal acelerada utilizando el módulo librosa", "espectrograma_aceleracion_librosa.png", None)
func.Graficar_espectrograma_banda_angosta(data_decimacion, fs_decimacion, None, None, None, None,  "Espectrograma de la señal acelerada utilizando decimación y ventaneo", "espectrograma_aceleracion_decimacion.png", None)
func.Graficar_espectrograma_banda_angosta(data_original_rapido, fs_original_rapido, None, None, None, None, "Espectrograma de la señal original rápida", "espectrograma_original_rapido.png", None)

## Realización de espectrogramas con el eje de frecuencias acotado entre 0 y 2kHz
func.Graficar_espectrograma_banda_angosta(data_original_lento, fs_original_lento, None, None, 0, 2000, "Espectrograma de la señal original (para frecuencias entre 0Hz y 2kHz)", "espectrograma_original_f0_2k.png", None)
func.Graficar_espectrograma_banda_angosta(data_librosa, fs_librosa, None, None, 0, 2000, "Espectrograma de la señal acelerada utilizando el módulo librosa (para frecuencias entre 0Hz y 2kHz)", "espectrograma_aceleracion_librosa_f0_2k.png", None)
func.Graficar_espectrograma_banda_angosta(data_decimacion, fs_decimacion, None, None, 0, 2000,  "Espectrograma de la señal acelerada utilizando decimación y ventaneo (para frecuencias entre 0Hz y 2kHz)", "espectrograma_aceleracion_decimacion_f0_2k.png", None)
func.Graficar_espectrograma_banda_angosta(data_original_rapido, fs_original_rapido, None, None, 0, 2000, "Espectrograma de la señal original rápida (para frecuencias entre 0Hz y 2kHz)", "espectrograma_original_rapido_f0_2k.png", None)
