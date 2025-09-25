# IMPORTS: Librerías necesarias para el procesamiento básico
import argparse  # Para leer argumentos de línea de comandos (ej. --archivo)
import os  # Para manejar archivos y carpetas (ej. crear directorios, verificar existencia)
import librosa  # Para cargar y procesar audio (convierte WAV/MP3 a array numérico)
import whisper  # Librería principal de OpenAI para transcripción de audio a texto con IA
from datetime import datetime  # Para obtener la fecha actual (nuevo: para título dinámico)
from reportlab.lib.pagesizes import letter  # Tamaño estándar de página para PDF (carta)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer  # Elementos para construir PDF: documento, párrafos y espaciadores
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # Estilos predefinidos y personalizados para texto en PDF
from reportlab.lib import colors  # Colores para estilizar PDF (ej. azul para títulos)

# CONFIGURACIÓN: Constantes globales para fácil ajuste
OUTPUT_DIR = "../tests/outputs"  # Carpeta donde se guardan TXT y PDF (relativa a src/)
MODEL_SIZE = "base"  # Tamaño del modelo Whisper: "tiny" (rápido), "base" (balanceado), "small" (mejor en español)

def obtener_fecha_actual():
    """
    PROPÓSITO: Obtiene la fecha actual en formato YYYY-MM-DD para usar en títulos.
    - Usa datetime.now() para fecha/hora local.
    - Formato simple: "2024-10-15" (fácil de leer y ordenar).
    INPUTS: Ninguno
    OUTPUTS: String con la fecha (ej. "2024-10-15")
    LÓGICA: Automático; no depende de zona horaria (usa local). Útil para auditabilidad en tesis.
    """
    fecha = datetime.now().strftime("%Y-%m-%d")  # Formato: Año-Mes-Día (estándar ISO)
    return fecha

def cargar_audio(ruta_archivo):
    """
    PROPÓSITO: Carga el archivo de audio en memoria para procesamiento.
    - Verifica si el archivo existe.
    - Usa librosa para cargar en formato estandarizado (mono, 16kHz para Whisper).
    INPUTS: ruta_archivo (string, ej. "../tests/prueba.wav")
    OUTPUTS: Tupla (y, sr) donde y es array NumPy de muestras de audio, sr es sample rate (16000 Hz)
    LÓGICA: Prepara audio crudo; lanza error si no existe o formato inválido. Imprime info de duración.
    """
    # Verifica existencia del archivo para evitar errores downstream
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"Archivo no encontrado: {ruta_archivo}")
    
    # Carga audio: sr=16000 (estándar para voz en Whisper), mono=True (un canal para simplicidad)
    y, sr = librosa.load(ruta_archivo, sr=16000, mono=True)
    # Calcula duración en segundos para info al usuario
    duracion = len(y) / sr
    print(f"Audio cargado: Duración {duracion:.2f}s, Sample rate {sr}Hz")
    return y, sr

def transcribir_audio_completo(y, sr):
    """
    PROPÓSITO: Transcribe el audio entero usando Whisper y retorna solo el texto unificado.
    - Carga el modelo una vez (eficiente).
    - Procesa el array completo; Whisper unifica el texto automáticamente (sin divisiones manuales).
    INPUTS: y (array de audio), sr (sample rate)
    OUTPUTS: String con el texto transcrito completo (un párrafo continuo)
    LÓGICA: Usa fp16=False para CPU (evita warning FP16); language='es' para español preciso.
    NOTA: Whisper maneja puntuación y mayúsculas auto; strip() limpia espacios extra.
    """
    print("Cargando modelo Whisper...")
    # Carga modelo: "base" es ~142MB, bueno para español; descarga auto si no existe
    model = whisper.load_model(MODEL_SIZE)
    
    print("Transcribiendo audio completo...")
    # Transcribe todo el array y: fp16=False (usa FP32 en CPU, más estable pero lento)
    result = model.transcribe(y, language='es', fp16=False)
    
    # Extrae texto completo unificado: Whisper lo proporciona directamente en result["text"]
    texto_completo = result["text"].strip()  # Limpia espacios al inicio/fin
    
    print(f"Transcripción completada: {len(texto_completo)} caracteres")
    return texto_completo

def generar_transcripcion_txt(texto_completo, nombre_archivo):
    """
    PROPÓSITO: Crea archivo TXT con solo el texto transcrito completo (sin timestamps).
    - Incluye un header con fecha actual y el texto unificado.
    - Usa UTF-8 para acentos en español.
    INPUTS: texto_completo (string), nombre_archivo (string base, ej. "prueba")
    OUTPUTS: Ruta al TXT generado
    LÓGICA: Escribe directo; crea carpeta si no existe. Título dinámico con fecha para trazabilidad.
    """
    # Crea directorio de outputs si no existe (evita errores)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Ruta completa: ej. "../tests/outputs/prueba_transcripcion.txt"
    ruta_txt = os.path.join(OUTPUT_DIR, f"{nombre_archivo}_transcripcion.txt")
    
    # Obtiene fecha actual para título
    fecha = obtener_fecha_actual()
    
    with open(ruta_txt, 'w', encoding='utf-8') as f:  # 'w' para escribir, UTF-8 para español
        # Header con fecha dinámica: ej. "TRANSCRIPCIÓN DEL 2024-10-15"
        f.write(f"TRANSCRIPCIÓN DEL {fecha.upper()}\n")
        f.write("=" * 50 + "\n\n")
        # Escribe el texto completo directamente (un bloque)
        f.write(texto_completo.capitalize() + "\n")  # Capitaliza primera letra para formalidad
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Longitud: {len(texto_completo)} caracteres\n")  # Info extra
        f.write(f"Fecha de creación: {fecha}\n")  # Repite fecha al final para referencia
    
    print(f"TXT generado: {ruta_txt}")
    return ruta_txt

def generar_transcripcion_pdf(texto_completo, nombre_archivo):
    """
    PROPÓSITO: Crea PDF simple con solo el texto transcrito completo (un párrafo unificado).
    - Usa ReportLab para layout: Título con fecha + párrafo grande con todo el texto.
    - Estilos personalizados para visual (azul título, texto justificado).
    INPUTS: texto_completo (string), nombre_archivo (string base)
    OUTPUTS: Ruta al PDF generado
    LÓGICA: Construye "story" con un solo Paragraph principal → Build genera PDF.
    NOTA: Título incluye fecha; ideal para tesis (imprimible, profesional); texto fluye en páginas auto.
    """
    # Crea directorio
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Ruta completa: ej. "../tests/outputs/prueba_transcripcion.pdf"
    ruta_pdf = os.path.join(OUTPUT_DIR, f"{nombre_archivo}_transcripcion.pdf")
    
    # Obtiene fecha actual para título
    fecha = obtener_fecha_actual()
    
    # Documento base: pagesize=letter (8.5x11 pulgadas, estándar)
    doc = SimpleDocTemplate(ruta_pdf, pagesize=letter)
    styles = getSampleStyleSheet()  # Estilos predefinidos de ReportLab
    
    # Estilo custom para título: Grande, azul, espacio después (incluye fecha)
    titulo_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        textColor=colors.darkblue
    )
    
    # Estilo custom para texto completo: Normal, justificado, espacio entre líneas
    texto_style = ParagraphStyle(
        'CustomText',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        alignment=1,  # 1 = justificado (texto alineado a ambos lados)
        leftIndent=20  # Indentado para legibilidad
    )
    
    story = []  # Lista de elementos a agregar al PDF (flowables)
    # Título dinámico con fecha: ej. "TRANSCRIPCIÓN DEL 2024-10-15"
    story.append(Paragraph(f"TRANSCRIPCIÓN DEL {fecha.upper()}", titulo_style))
    story.append(Spacer(1, 12))  # Espacio vertical (12 puntos)
    
    # Agrega el texto completo como un solo párrafo grande
    story.append(Paragraph(texto_completo.capitalize(), texto_style))  # Capitaliza y usa estilo
    
    # Construye y guarda PDF (auto-pagina si texto largo)
    doc.build(story)
    print(f"PDF generado: {ruta_pdf}")
    return ruta_pdf

def main():
    """
    PROPÓSITO: Función principal —coordina el flujo completo del script.
    - Lee argumentos (archivo de audio).
    - Maneja errores con try/except.
    - Llama funciones en secuencia: Carga → Transcribe → Genera outputs con título de fecha.
    INPUTS: Argumentos de CLI (ej. --archivo)
    OUTPUTS: Archivos TXT/PDF con texto completo y fecha; prints de progreso
    LÓGICA: Usa argparse para flexibilidad; nombre_base de archivo para outputs únicos.
    EJECUCIÓN: Se llama al final con if __name__ == "__main__".
    """
    # Configura parser para argumentos: --archivo es requerido
    parser = argparse.ArgumentParser(description="Transcribe audio completo con Whisper (solo texto unificado, título con fecha).")
    parser.add_argument('--archivo', required=True, help='Ruta al archivo de audio (WAV/MP3)')
    args = parser.parse_args()
    
    # Extrae nombre base sin extensión (ej. "prueba_comandos" de "prueba_comandos.wav")
    nombre_base = os.path.splitext(os.path.basename(args.archivo))[0]
    
    # Obtiene fecha actual para info inicial
    fecha = obtener_fecha_actual()
    print(f"Iniciando transcripción simple ({fecha}): {args.archivo}")
    
    try:
        # Paso 1: Carga el audio
        y, sr = cargar_audio(args.archivo)
        
        # Paso 2: Transcribe y obtiene texto completo
        texto_completo = transcribir_audio_completo(y, sr)
        
        # Verifica si hay texto (evita outputs vacíos)
        if not texto_completo:
            print("No se detectó audio o transcripción vacía.")
            return
        
        # Paso 3: Genera archivos con título de fecha
        generar_transcripcion_txt(texto_completo, nombre_base)
        generar_transcripcion_pdf(texto_completo, nombre_base)
        
        print("¡Transcripción completada! Revisa outputs para TXT/PDF con fecha.")
        
    except Exception as e:
        # Manejo de errores: Imprime mensaje y traceback completo para debug
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# PUNTO DE ENTRADA: Ejecuta main() si se corre directamente (no si se importa)
if __name__ == "__main__":
    main()
