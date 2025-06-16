def Desacelerar_audio_con_expansion_y_pasabajos(input_wav, output_wav, factor, numtaps: int = 101):
    """
    Desacelera un archivo WAV mono utilizando expansión por inserción de ceros
    y un filtro pasa-bajos para interpolación.
    """
    # Leer archivo
    fs, data = wavfile.read(input_wav)

    # Normalización para trabajar con float32
    data = data.astype(np.float32)

    # Expansión: insertar (factor - 1) ceros entre muestras
    expanded = np.zeros(int(len(data) * factor), dtype=np.float32)
    expanded[::factor] = data

    # Filtro pasa-bajos para interpolar (ventana Hamming)
    cutoff = 0.5 / factor  # Frecuencia normalizada (Nyquist = 1.0)
    fir_coeff = firwin(numtaps, cutoff, window='hamming')
    filtered = lfilter(fir_coeff, 1.0, expanded)

    # Compensar retardo del filtro
    delay = (numtaps - 1) // 2
    filtered = np.roll(filtered, -delay)
    filtered[-delay:] = 0.0

    # Re-normalizar a int16 para guardar
    filtered = np.clip(filtered, -32768, 32767).astype(np.int16)

    # Guardar con la misma frecuencia de muestreo
    wavfile.write(output_wav, fs, filtered)
