from scipy.io import wavfile

import funciones_tpe as func

FACTOR_DESACELERACION = 0.5
RUTA_ARCHIVO_ORIGINAL_LENTO = "Picasso_lento.wav"
RUTA_ARCHIVO_ORIGINAL_RAPIDO = "Picasso_rapido.wav"

RUTA_ARCHIVO_DESACELERADO_EXPANSION = "Picasso_interpolacion.wav"
RUTA_ARCHIVO_DESACELERADO_LIBROSA = "Picasso_desacelerado_librosa.wav"


#### DESACELERACIÓN DE LA SEÑAL ###
func.Acelerar_audio_con_librosa(RUTA_ARCHIVO_ORIGINAL_RAPIDO, RUTA_ARCHIVO_DESACELERADO_LIBROSA, FACTOR_DESACELERACION)
func.Desacelerar_audio_con_expansion_y_pasabajos(RUTA_ARCHIVO_ORIGINAL_RAPIDO, RUTA_ARCHIVO_DESACELERADO_EXPANSION, FACTOR_DESACELERACION)

## Obtención de los vectores de data de los audios
fs_original_lento, data_original_lento = wavfile.read(RUTA_ARCHIVO_ORIGINAL_LENTO)
fs_librosa, data_librosa = wavfile.read(RUTA_ARCHIVO_DESACELERADO_LIBROSA)
fs_expansion, data_expansion = wavfile.read(RUTA_ARCHIVO_DESACELERADO_EXPANSION)
fs_original_rapido, data_original_rapido = wavfile.read(RUTA_ARCHIVO_ORIGINAL_RAPIDO)

## Realización de espectrogramas completos
func.Graficar_espectrograma_banda_angosta(data_original_lento, fs_original_lento, None, None, None, None, "Espectrograma de la señal original", "espectrograma_original.png", None)
func.Graficar_espectrograma_banda_angosta(data_librosa, fs_librosa, None, None, None, None, "Espectrograma de la señal ralentizada utilizando el módulo librosa", "espectrograma_ralentizacion_librosa.png", None)
func.Graficar_espectrograma_banda_angosta(data_expansion, fs_expansion, None, None, None, None,  "Espectrograma de la señal ralentizada utilizando expansión", "espectrograma_ralentizacion_expansion.png", None)
func.Graficar_espectrograma_banda_angosta(data_original_rapido, fs_original_rapido, None, None, None, None, "Espectrograma de la señal original rápida", "espectrograma_original_rapido.png", None)

## Realización de espectrogramas con el eje de frecuencias acotado entre 0 y 2kHz
func.Graficar_espectrograma_banda_angosta(data_original_lento, fs_original_lento, None, None, 0, 2000, "Espectrograma de la señal original (para frecuencias entre 0Hz y 2kHz)", "espectrograma_original_f0_2k.png", None)
func.Graficar_espectrograma_banda_angosta(data_librosa, fs_librosa, None, None, 0, 2000, "Espectrograma de la señal ralentizada utilizando el módulo librosa (para frecuencias entre 0Hz y 2kHz)", "espectrograma_ralentizada_librosa_f0_2k.png", None)
func.Graficar_espectrograma_banda_angosta(data_expansion, fs_expansion, None, None, 0, 2000,  "Espectrograma de la señal acelerada utilizando decimación y ventaneo (para frecuencias entre 0Hz y 2kHz)", "espectrograma_ralentizacion_expansion_f0_2k.png", None)
func.Graficar_espectrograma_banda_angosta(data_original_rapido, fs_original_rapido, None, None, 0, 2000, "Espectrograma de la señal original rápida (para frecuencias entre 0Hz y 2kHz)", "espectrograma_original_rapido_f0_2k.png", None)
