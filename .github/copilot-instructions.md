# Copilot Instructions for `sistemas-de-actas`

## Project Overview
This project automates the generation of meeting minutes (actas) from audio recordings using voice commands and keyword filtering. The main logic is in `src/sistemas_actas.py`, which processes audio files, detects commands, transcribes relevant segments, and generates filtered outputs in TXT and PDF formats.

## Architecture & Data Flow
- **Input:** Audio files (WAV/MP3) placed in `tests/`.
- **Processing:**
  - Audio is segmented and transcribed using OpenAI Whisper.
  - Voice commands (e.g., "en acta", "fuera de acta") control which segments are included in the final acta.
  - Keyword filtering highlights important content.
- **Output:** Filtered TXT and PDF files are saved in `tests/outputs/`.

## Key Files & Directories
- `src/sistemas_actas.py`: Main script. Contains all core logic for audio segmentation, command detection, transcription, and output generation.
- `tests/`: Place test audio files here. Outputs are written to `tests/outputs/`.
- `docs/diagramas/`: For architecture diagrams (if any).

## Developer Workflows
- **Run the main script:**
  ```powershell
  python src/sistemas_actas.py --archivo tests/prueba1.wav
  ```
  Outputs will be generated in `tests/outputs/`.
- **Dependencies:**
  - Requires Python 3.8+ and the following packages: `numpy`, `librosa`, `whisper`, `reportlab`, `soundfile`.
  - Install with:
    ```powershell
    pip install numpy librosa openai-whisper reportlab soundfile
    ```
- **Debugging:**
  - Print statements are used for step-by-step tracing.
  - Errors are printed with stack traces for easier diagnosis.

## Project-Specific Patterns
- **Command-driven segmentation:** Only segments between "en acta" and "fuera de acta" are included in outputs.
- **Keyword filtering:** The list of keywords is defined in `PALABRAS_CLAVE` in `src/sistemas_actas.py`.
- **Output conventions:** Output filenames follow the pattern `<audio_base_name>_acta_v2.txt` and `.pdf`.
- **No test framework:** Manual testing via CLI and inspection of output files.

## Integration Points
- **OpenAI Whisper:** Used for Spanish-language transcription. Model is loaded as `whisper.load_model("base")`.
- **ReportLab:** Used for PDF generation.

## Example Usage
```powershell
python src/sistemas_actas.py --archivo tests/prueba1.wav
```

## Tips for AI Agents
- Always filter segments using the command logic in `procesar_audio_con_comandos`.
- Outputs must be written to `tests/outputs/`.
- Use the provided keyword list for filtering and highlighting.
- Maintain the output filename conventions for consistency.

---
If any section is unclear or missing, please provide feedback to improve these instructions.
