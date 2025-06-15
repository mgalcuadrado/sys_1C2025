import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

FIGSIZE_X = 10
FIGSIZE_Y = 6

# =================================== FUNCIONES AUXILIARES PARTE 2 ======================================
# graficar_espectrograma_banda_angosta grafica el espectrograma de banda angosta de una señal audio_data con una fs sample_rate, considerando la porción de la señal desde start_time_s [en segundos] y end_time_s[en segundos], y grafica desde las frecuencias start_freq a end_freq. Si start_time_s y end_time_s son None se grafica la señal audio_data completa. Lo mismo pasa con start_freq y end_freq. Además, se pide el título del gráfico y el nombre de la imagen. Adicionalmente se ofrece un arreglo func, donde func[0] debe ser el nombre de la función a llamar y en el resto de los elementos del vector se pueden incluir otros datos necesarios para realizar la operación deseada.
def graficar_espectrograma_banda_angosta(audio_data, sample_rate, start_time_s, end_time_s, start_freq, end_freq, title, image_name, func):
    samples_per_segment = 2048 #longitud de la ventana considerada para cada fragmento de la señal
    sample_overlap = samples_per_segment / 2 # la mitad de las muestras se superponen entre una ventana y la siguiente para evitar efectos de borde
    sample = audio_data
    if start_time_s is not None and end_time_s is not None:
        start_time = int(start_time_s * sample_rate)
        end_time = int(end_time_s * sample_rate)
        sample = audio_data[start_time:end_time]   
    frecuencias, tiempos, intensidades = signal.spectrogram(sample, sample_rate, window='hamming', nperseg=samples_per_segment, noverlap=sample_overlap)
    intensidades_dB = 10 * np.log10(intensidades + 1e-10)
    # Gráfico del espectrograma de banda angosta completo 
    plt.figure(figsize=(FIGSIZE_X, FIGSIZE_Y))
    plt.pcolormesh(tiempos, frecuencias, intensidades_dB, shading='gouraud', cmap='viridis')
    if func is not None:
        func(frecuencias, tiempos, intensidades) 
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    plt.title(title)
    if start_freq is not None and end_freq is not None:
        plt.ylim(start_freq, end_freq)
    plt.colorbar(label='Densidad espectral [dB]')
    plt.tight_layout()
    plt.savefig(image_name)
    plt.show()

def hallar_fundamentales_y_graficar_en_espectrograma(frecuencias, tiempos, intensidades):
    fundamentales = []
    armonicos = []

    for i in range(intensidades.shape[1]):
        espectro = intensidades[:, i]
        peaks, _ = signal.find_peaks(espectro, height=np.max(espectro)*0.1, distance=10)  
        if len(peaks) == 0:
            fundamentales.append(np.nan)
            armonicos.append([])
            continue
        frecs_pico = frecuencias[peaks]
        fundamental = frecs_pico[0]
        fundamentales.append(fundamental)
        
        # Buscar armónicos como múltiplos enteros del fundamental ± tolerancia
        tolerancia = 30  # Hz
        harmonics = [freq for freq in frecs_pico if abs(freq % fundamental) < tolerancia or abs(fundamental - freq % fundamental) < tolerancia]
        armonicos.append(harmonics)

    plt.plot(tiempos, fundamentales, color='r', label='Fundamental')

    
    
# ===================================== PROCESAMIENTO DE ARCHIVOS =======================================
# Lectura de archivos de audio WAV
picasso_hombre_lento = 'Picasso_hombre_5s.wav' #se decide usar solo la muestra lenta
sample_rate_hl, audio_data_hl = wavfile.read(picasso_hombre_lento)

# =================================== ÍTEM 4: ESPECTROGRAMA DE BANDA ANGOSTA ======================================
#Gráfico del espectrograma de banda angosta completo
graficar_espectrograma_banda_angosta(audio_data_hl, sample_rate_hl, None, None, None, None, 'Espectrograma de banda angosta de la muestra completa', "Espectrograma_banda_angosta_muestra_completa.png", None)
# Gráfico del espectrograma de banda angosta completo mostrando solo frecuencias de 0 a 2kHz
graficar_espectrograma_banda_angosta(audio_data_hl, sample_rate_hl, None, None, 0, 2000, 'Espectrograma de banda angosta de la muestra completa (con frecuencias de hasta 2kHz)', "Espectrograma_banda_angosta_muestra_completa_frecuencias_bajas.png", None)
# Detección de fundamental y armónicos, graficados sobre el espectrograma
graficar_espectrograma_banda_angosta(audio_data_hl, sample_rate_hl, None, None, None, None, 'Espectrograma de banda angosta de la muestra completa con sus fundamentales superpuestos', "Espectrograma_banda_angosta_muestra_completa_con_fundamentales.png", hallar_fundamentales_y_graficar_en_espectrograma)




# =================================== ÍTEM 7: MODIFICACIÓN DE LA SEÑAL CON TD-PSOLA ======================================



signal_normalizada = audio_data_hl / np.max(np.abs(audio_data_hl))


# === Detección de picos de pitch (en segmento sonoro manualmente elegido) ===

output = np.zeros_like(signal_normalizada, dtype=float)
segmentos = [["i", 0.32, 1.69], ["a", 1.89, 3.36], ["o", 3.95, 5.79]]

n_segmentos = 3

# === Detección de picos de pitch (en segmento sonoro manualmente elegido) ===
 
output = np.zeros_like(signal_normalizada, dtype=float)
acumulador_pesos = np.zeros_like(output, dtype=float) #para solucionar la sobresaturación de la señal
i = 0
while i < n_segmentos:
    segmento = segmentos[i]
    inicio = int(segmento[1]*sample_rate_hl)
    fin = int(segmento[2]*sample_rate_hl)
    i+=1
    signal_cortada = signal_normalizada[inicio: fin]  
    picos, _ = signal.find_peaks(signal_cortada,distance=sample_rate_hl / 500)  # distancia ~ periodo mínimo 
    if len(picos) == 0: ##caso borde: por si no se encontraron picos (No debería ocurrir para los segmentos seleccionados)
        continue
    # === TD-PSOLA para modificar pitch ===
    factor = 1.4 # multiplicar pitch por 1.4 

    periodos = np.diff(picos) #se busca la diferencia entre los picos consecutivos para calcular la distancia entre picos originales
    periodo_medio = np.mean(periodos) #se busca el promedio de estos periodos para delimitar un periodo medio entre picos originales
    window_size = int(periodo_medio * 2)  # tamaño de ventana de dos periodos para evitar solapamientos excesivos de los picos (y aliasing por extensión)
    half_window = window_size // 2 
    window = signal.hann(window_size) # se va a multiplicar lo guardado por una ventana de hann para suavizar la señal
    for pico in picos:
        #nearest = min(picos, key=lambda x: abs(x - p_out))  # encontrar pico original más cercano al nuevo pico
        start = inicio + pico - half_window  #se delimita la ventana a copiar para que quede centrada en el pico original: inicio del fragmento + posicion del pico original - la mitad de la ventana (para quedar a ventana/2 del pico)
        end = start + window_size               #final de la ventana a una ventana completa del inicio de la misma
        if start < 0 or end > len(signal_normalizada):      #se verifica no "pasarse" de la sección de señal válida
            continue
        frame = signal_normalizada[start:end] * window      #se crean los frames nuevos suavizándolos al multiplicarlos por la ventana
        out_start = inicio + int(pico/factor) - half_window            #se delimita la ventana en la que se van a copiar los valores centrándolos en pico nuevo
        out_end = out_start + window_size
        if out_start < 0 or out_end > len(output):          #se verifica no "pasarse" de la sección de señal válida
            continue
        output[out_start:out_end] += frame                  #se guardan los frames en el arreglo de salida, con los valores centrados en el pico nuevo
        acumulador_pesos[out_start:out_end] += window       #se registra en un acumulador cuántas veces una misma ventana solapó esos ítems del arreglo de salida para aplanar después la señal y evitar sobresaturación de la señal nueva


# === Guardar resultado ===
#primero se busca dessaturar un poco la señal
acumulador_pesos[acumulador_pesos == 0] = 1  
output /= acumulador_pesos
#después se agregan los segmentos faltantes de la señal que no son parte de los armónicos modificados
indice = 0
inicio_muestras_nulas = 0
muestras_nulas_seguidas = 0
for out in output:
    if out == 0:
        if muestras_nulas_seguidas == 0:
            inicio_muestras_nulas = indice
        muestras_nulas_seguidas+=1
        indice+=1
        continue
    #si muchas muestras son nulas es que ese espacio no era parte del intervalo de las vocales, así que se lo añade al arreglo de salida
    if muestras_nulas_seguidas > 100: 
        output[inicio_muestras_nulas:indice] += signal_normalizada[inicio_muestras_nulas:indice]
    muestras_nulas_seguidas = 0
    indice+=1
output[inicio_muestras_nulas:indice-1] += signal_normalizada[inicio_muestras_nulas:indice-1]

output = np.int16(output / np.max(np.abs(output)) * 32767)
#finalmente se guarda la señal modificada en el archivo .wav
wavfile.write("modificada.wav", sample_rate_hl, output)

#Gráfico del espectrograma de banda angosta completo
graficar_espectrograma_banda_angosta(output, sample_rate_hl, None, None, None, None, 'Espectrograma de banda angosta de la muestra modificada', "Espectrograma_banda_angosta_muestra_modificada.png", None)
# Gráfico del espectrograma de banda angosta completo mostrando solo frecuencias de 0 a 2kHz
graficar_espectrograma_banda_angosta(output, sample_rate_hl, None, None, 0, 2000, 'Espectrograma de banda angosta de la muestra modificada (con frecuencias de hasta 2kHz)', "Espectrograma_banda_angosta_muestra_modificada_frecuencias_bajas.png", None)
# Detección de fundamental y armónicos, graficados sobre el espectrograma
graficar_espectrograma_banda_angosta(output, sample_rate_hl, None, None, None, None, 'Espectrograma de banda angosta de la muestra modificada con sus fundamentales superpuestos', "Espectrograma_banda_angosta_muestra_modificada_con_fundamentales.png", hallar_fundamentales_y_graficar_en_espectrograma)


plt.figure(figsize=(FIGSIZE_X, FIGSIZE_Y))
tiempo = np.arange(audio_data_hl.size) / sample_rate_hl 
plt.plot(tiempo, audio_data_hl, label= "original")
tiempo = np.arange(output.size) / sample_rate_hl 
plt.plot(tiempo, output, label= "modificada")
plt.grid()
plt.xlabel('Tiempo [s]', fontsize=15)
plt.ylabel('Amplitud', fontsize=15)
plt.legend()
plt.savefig("Comparacion_de_muestras_en_el_tiempo.png")
plt.show()
plt.close()