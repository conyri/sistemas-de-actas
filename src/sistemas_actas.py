import argparse
import os
import numpy as np
import librosa
import whisper
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import soundfile as sf  # Para temp files

     

# Constantes
SEGMENTO_DURACION = 3.0  # Segundos por segmento para análisis fino (ajusta para precisión)
PALABRAS_CLAVE = ["obra", "riesgo", "medida", "acta", "reunión", "proyecto", "civil"]
COMANDOS_INICIO = ["se inicia la sesión", "iniciamos", "empezamos"]  # Frases para activar modo general
COMANDO_ACTIVAR = ["en acta", "dentro de acta"]  # Activa transcripción
COMANDO_DESACTIVAR = ["fuera de acta", "off acta"]  # Desactiva
OUTPUT_DIR = "../tests/outputs"

def cargar_audio(ruta_archivo):
    """Carga audio (WAV/MP3 con librosa; Whisper maneja formatos)."""
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"Archivo no encontrado: {ruta_archivo}")
    
    try:
        y, sr = librosa.load(ruta_archivo, sr=16000, mono=True)
        print(f"Audio cargado: Duración {len(y)/sr:.2f}s, Sample rate {sr}Hz")
        return y, sr
    except Exception as e:
        print(f"Error cargando audio: {e}")
        raise

def segmentar_audio_fino(y, sr):
    """Divide audio en segmentos pequeños para análisis secuencial."""
    duracion_total = len(y) / sr
    segmentos = []
    for inicio in np.arange(0, duracion_total, SEGMENTO_DURACION):
        fin = min(inicio + SEGMENTO_DURACION, duracion_total)
        audio_seg = y[int(inicio * sr):int(fin * sr)]
        segmentos.append({
            'inicio': inicio,
            'fin': fin,
            'duracion': fin - inicio,
            'audio': audio_seg
        })
    print(f"Segmentos finos creados: {len(segmentos)} (cada ~{SEGMENTO_DURACION}s)")
    return segmentos

def transcribir_segmento(audio_seg, sr, model):
    """Transcribe segmento con Whisper."""
    temp_path = "temp_seg.wav"
    sf.write(temp_path, audio_seg, sr)
    result = model.transcribe(temp_path, language='es', fp16=False) # Español
    os.remove(temp_path)
    texto = result["text"].strip().lower()
    return texto if texto else ""

def detectar_comando(texto):
    """Detecta si el texto contiene comandos (case-insensitive)."""
    texto_lower = texto.lower()
    
    if any(cmd in texto_lower for cmd in COMANDOS_INICIO):
        return "inicio"
    elif any(cmd in texto_lower for cmd in COMANDO_ACTIVAR):
        return "activar"
    elif any(cmd in texto_lower for cmd in COMANDO_DESACTIVAR):
        return "desactivar"
    return None

def buscar_keywords(texto):
    """Busca palabras clave en texto (case-insensitive)."""
    texto_lower = texto.lower()
    matches = [kw for kw in PALABRAS_CLAVE if kw in texto_lower]
    return matches if matches else []

def procesar_audio_con_comandos(y, sr):
    """Procesa audio: Transcribe segmentos, detecta comandos y filtra 'en acta'."""
    model = whisper.load_model("base")  # Modelo ligero para español; usa "small" para mejor precisión
    segmentos_finos = segmentar_audio_fino(y, sr)
    
    acta_segmentos = []  # Solo partes relevantes
    estado_activo = False  # False hasta "inicio sesión"
    en_acta = False  # False hasta "en acta"
    
    print("Procesando segmentos con comandos...")
    for i, seg in enumerate(segmentos_finos):
        transcripcion = transcribir_segmento(seg['audio'], sr, model)
        if not transcripcion:
            continue
        
        comando = detectar_comando(transcripcion)
        
        if comando == "inicio":
            estado_activo = True
            print(f"[{seg['inicio']:.1f}s] Inicio de sesión detectado. Modo activo.")
            continue  # No incluir el comando en acta
        
        if not estado_activo:
            print(f"[{seg['inicio']:.1f}s] Esperando inicio de sesión... (Ignorado: {transcripcion[:50]}...)")
            continue
        
        if comando == "activar":
            en_acta = True
            print(f"[{seg['inicio']:.1f}s] 'En acta' detectado. Iniciando transcripción.")
            continue  # No incluir comando
        
        if comando == "desactivar":
            en_acta = False
            print(f"[{seg['inicio']:.1f}s] 'Fuera de acta' detectado. Pausando transcripción.")
            # Incluye el comando para registro
            acta_segmentos.append({
                'inicio': seg['inicio'],
                'fin': seg['fin'],
                'transcripcion': transcripcion,
                'matches': buscar_keywords(transcripcion),
                'tipo': 'comando'
            })
            continue
        
        # Si está 'en acta', incluye el segmento
        if en_acta and transcripcion:
            matches = buscar_keywords(transcripcion)
            acta_segmentos.append({
                'inicio': seg['inicio'],
                'fin': seg['fin'],
                'transcripcion': transcripcion.capitalize(),  # Formato para acta
                'matches': matches,
                'tipo': 'contenido'
            })
            print(f"[{seg['inicio']:.1f}s] En acta: {transcripcion[:50]}... Matches: {matches}")
        else:
            print(f"[{seg['inicio']:.1f}s] Fuera de acta (Ignorado: {transcripcion[:50]}...)")
    
    if not acta_segmentos:
        print("Advertencia: No se detectaron partes 'en acta'. Verifica comandos en audio.")
    
    print(f"Partes de acta extraídas: {len(acta_segmentos)}")
    return acta_segmentos

def generar_acta_txt(acta_segmentos, nombre_archivo):
    """Genera TXT solo con partes relevantes."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ruta_txt = os.path.join(OUTPUT_DIR, f"{nombre_archivo}_acta_v2.txt")
    
    with open(ruta_txt, 'w', encoding='utf-8') as f:
        f.write("ACTA DE REUNIÓN INTELIGENTE (Solo Partes 'En Acta')\n")
        f.write("=" * 50 + "\n\n")
        f.write("Fecha: Automática\n\n")
        
        for seg in acta_segmentos:
            tiempo = f"{seg['inicio']:.1f}s - {seg['fin']:.1f}s"
            f.write(f"Tiempo: {tiempo}\n")
            f.write(f"Transcripción: {seg['transcripcion']}\n")
            if seg['matches']:
                f.write(f"Keywords: {', '.join(seg['matches'])}\n")
            f.write("-" * 40 + "\n\n")
        
        f.write("Resumen: Solo contenido relevante procesado.\n")
    
    print(f"TXT generado: {ruta_txt}")
    return ruta_txt

def generar_acta_pdf(acta_segmentos, nombre_archivo):
    """Genera PDF con tabla de partes relevantes."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ruta_pdf = os.path.join(OUTPUT_DIR, f"{nombre_archivo}_acta_v2.pdf")
    
    doc = SimpleDocTemplate(ruta_pdf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=16, spaceAfter=20)
    story.append(Paragraph("ACTA DE REUNIÓN INTELIGENTE", title_style))
    story.append(Spacer(1, 12))
    
    data = [['Tiempo', 'Transcripción', 'Keywords']]
    for seg in acta_segmentos:
        tiempo = f"{seg['inicio']:.1f}s - {seg['fin']:.1f}s"
        trans = seg['transcripcion'][:60] + "..." if len(seg['transcripcion']) > 60 else seg['transcripcion']
        kws = ', '.join(seg['matches']) if seg['matches'] else 'Ninguno'
        data.append([tiempo, trans, kws])
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Resumen: Filtrado por comandos verbales.", styles['Normal']))
    
    doc.build(story)
    print(f"PDF generado: {ruta_pdf}")
    return ruta_pdf

def main():
    parser = argparse.ArgumentParser(description="Sistema de Actas con Comandos Verbales")
    parser.add_argument('--archivo', required=True, help='Ruta al archivo de audio (WAV/MP3)')
    args = parser.parse_args()
    
    print(f"Iniciando procesamiento inteligente: {args.archivo}")
    
    try:
        y, sr = cargar_audio(args.archivo)
        acta_segmentos = procesar_audio_con_comandos(y, sr)
        
        if not acta_segmentos:
            print("No hay contenido para acta. Prueba con audio que incluya comandos.")
            return
        
        nombre_base = os.path.splitext(os.path.basename(args.archivo))[0]
        generar_acta_txt(acta_segmentos, nombre_base)
        generar_acta_pdf(acta_segmentos, nombre_base)
        
        print("¡Proceso completado! Revisa tests/outputs para acta filtrada.")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
