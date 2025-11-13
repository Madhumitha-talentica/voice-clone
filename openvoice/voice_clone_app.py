import os
import argparse
import time
import torch
import gradio as gr
import langid

from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter


# -----------------------------
# Configuration
# -----------------------------
BASE_EN = "checkpoints/base_speakers/EN"
BASE_ZH = "checkpoints/base_speakers/ZH"
CONVERTER = "checkpoints/converter"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Supported styles for English (Chinese only 'default')
EN_STYLES = [
    "default",
    "whispering",
    "cheerful",
    "terrified",
    "angry",
    "sad",
    "friendly",
]
ZH_STYLES = ["default"]


# -----------------------------
# Lazy Global Model Holder
# -----------------------------
class VoiceModels:
    _loaded = False

    def __init__(self):
        self.en_tts = None
        self.zh_tts = None
        self.converter = None
        self.en_default_se = None
        self.en_style_se = None
        self.zh_default_se = None

    def load(self):
        if self._loaded:
            return
        # Base speakers
        self.en_tts = BaseSpeakerTTS(f"{BASE_EN}/config.json", device=DEVICE)
        self.en_tts.load_ckpt(f"{BASE_EN}/checkpoint.pth")
        self.zh_tts = BaseSpeakerTTS(f"{BASE_ZH}/config.json", device=DEVICE)
        self.zh_tts.load_ckpt(f"{BASE_ZH}/checkpoint.pth")
        # Converter
        self.converter = ToneColorConverter(f"{CONVERTER}/config.json", device=DEVICE)
        self.converter.load_ckpt(f"{CONVERTER}/checkpoint.pth")
        # Speaker embeddings
        self.en_default_se = torch.load(f"{BASE_EN}/en_default_se.pth").to(DEVICE)
        self.en_style_se = torch.load(f"{BASE_EN}/en_style_se.pth").to(DEVICE)
        self.zh_default_se = torch.load(f"{BASE_ZH}/zh_default_se.pth").to(DEVICE)
        self._loaded = True


MODELS = VoiceModels()


# -----------------------------
# Utility Functions
# -----------------------------
def detect_language(text: str) -> str:
    if not text.strip():
        return "en"
    return langid.classify(text)[0]


def validate_inputs(text: str, ref_audio: str, style: str):
    problems = []
    if len(text.strip()) < 2:
        problems.append("Prompt too short (>=2 characters).")
    if len(text) > 400:
        problems.append("Prompt too long (<=400 characters).")
    if not ref_audio:
        problems.append("Reference audio not provided.")
    return problems


# -----------------------------
# Inference Core
# -----------------------------
def clone_voice(prompt: str, reference_audio: str, style: str, sample_choice: str):
    t0 = time.time()
    MODELS.load()

    # Validate
    # Prefer uploaded reference; fallback to sample_choice
    effective_ref = reference_audio if reference_audio else sample_choice
    issues = validate_inputs(prompt, effective_ref, style)
    if issues:
        return "\n".join([f"[ERROR] {m}" for m in issues]), None, None

    lang = detect_language(prompt)
    lang = "zh" if lang.startswith("zh") else ("en" if lang.startswith("en") else lang)

    unsupported_original = None
    if lang not in ("en", "zh"):
        unsupported_original = lang
        # Fallback strategy: treat all other languages as English (romanization not handled)
        lang = "en"

    if lang == "zh":
        if style not in ZH_STYLES:
            return f"[ERROR] Style '{style}' not valid for Chinese (use 'default').", None, None
        tts_model = MODELS.zh_tts
        source_se = MODELS.zh_default_se
        tts_style = "default"
        language_tag = "Chinese"
    else:
        if style not in EN_STYLES:
            return f"[ERROR] Style '{style}' not valid for English.", None, None
        tts_model = MODELS.en_tts
        source_se = MODELS.en_default_se if style == "default" else MODELS.en_style_se
        tts_style = style
        language_tag = "English"

    # Extract target speaker embedding
    try:
        target_se, _ = se_extractor.get_se(
            effective_ref,
            MODELS.converter,
            target_dir="processed",
            vad=True,
        )
    except Exception as e:  # pragma: no cover
        return f"[ERROR] Failed to extract speaker embedding: {e}", None, None

    # Generate base TTS to temporary wav
    base_path = os.path.join(OUTPUT_DIR, "base_tmp.wav")
    tts_model.tts(prompt, base_path, speaker=tts_style, language=language_tag)

    # Convert tone color
    out_path = os.path.join(OUTPUT_DIR, "cloned.wav")
    MODELS.converter.convert(
        audio_src_path=base_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=out_path,
        message="@MyShell",
    )

    elapsed = time.time() - t0
    info = (
        f"Success. Language={language_tag} | Style={style} | "
        f"Duration={elapsed:.2f}s | Device={DEVICE}"
    )
    if unsupported_original:
        info += f" | Note: Original detected language '{unsupported_original}' not natively supported; fallback to English synthesis."
    return info, out_path, effective_ref


# -----------------------------
# Speech-to-Speech Helper
# -----------------------------
def speech_to_speech(source_audio: str, target_ref_audio: str, target_sample_choice: str, source_sample_choice: str):
    """Convert the timbre of source_audio to target speaker using ToneColorConverter.
    Preserves content/prosody/emotion from source.
    """
    t0 = time.time()
    MODELS.load()

    # Resolve inputs with fallback to samples
    effective_src = source_audio if source_audio else source_sample_choice
    effective_tgt = target_ref_audio if target_ref_audio else target_sample_choice

    problems = []
    if not effective_src:
        problems.append("Source audio not provided.")
    if not effective_tgt:
        problems.append("Target reference audio not provided.")
    if problems:
        return "\n".join([f"[ERROR] {m}" for m in problems]), None, None, None

    # Extract embeddings
    try:
        src_se, _ = se_extractor.get_se(
            effective_src,
            MODELS.converter,
            target_dir="processed",
            vad=True,
        )
        tgt_se, _ = se_extractor.get_se(
            effective_tgt,
            MODELS.converter,
            target_dir="processed",
            vad=True,
        )
    except Exception as e:
        return f"[ERROR] Failed to extract embeddings: {e}", None, None, None

    # Convert timbre using source content
    out_path = os.path.join(OUTPUT_DIR, "s2s_cloned.wav")
    MODELS.converter.convert(
        audio_src_path=effective_src,
        src_se=src_se,
        tgt_se=tgt_se,
        output_path=out_path,
        message="@MyShell",
    )

    elapsed = time.time() - t0
    info = f"Success. Speech→Speech | Duration={elapsed:.2f}s | Device={DEVICE}"
    return info, out_path, effective_src, effective_tgt


# -----------------------------
# Gradio UI
# -----------------------------
def build_ui():
    with gr.Blocks(title="Voice Clone", analytics_enabled=False, css="""
    #app-title {text-align:center; font-weight:600; font-size:2.0rem; margin: 0.5em 0;}
    .small-hint {font-size:0.75rem; color:#888;}
    /* Hide default gradio footer / branding */
    footer, .logo-and-links, .built-with-gradio, .api-info {display:none !important;}
    """) as demo:
        gr.Markdown("""<div id='app-title'>🔊 Voice Clone Studio</div>""")
        gr.Markdown(
            """
            Enter text, upload (or record) a short clean voice sample (≥3s), select a style, and generate cloned speech.
            """
        )

        # Discover sample audios from resources
        sample_files = [
            os.path.join("resources", f)
            for f in sorted(os.listdir("resources"))
            if f.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg"))
        ]

        with gr.Tabs():
            with gr.Tab("Text → Voice"):
                with gr.Row():
                    with gr.Column(scale=5):
                        text_in = gr.Textbox(
                            label="Text Prompt",
                            placeholder="Type the text to speak...",
                            lines=4,
                            info="Max 400 characters. Language auto-detected.",
                        )
                        ref_audio = gr.Audio(
                            label="Reference Speaker Audio",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        sample_dropdown = gr.Dropdown(
                            label="Or Pick Sample Audio",
                            choices=sample_files,
                            value=sample_files[0] if sample_files else None,
                            allow_none=True,
                            info="If no upload/mic audio provided, selected sample is used.",
                        )
                        style = gr.Dropdown(
                            label="Style",
                            choices=EN_STYLES,
                            value="default",
                        )
                        run_btn = gr.Button("🎤 Clone Voice", variant="primary")

                    with gr.Column(scale=5):
                        info_box = gr.Textbox(label="Status / Log", interactive=False)
                        out_audio = gr.Audio(label="Cloned Audio", autoplay=True)
                        ref_audio_echo = gr.Audio(label="Reference Audio (Echo)")

                run_btn.click(
                    fn=clone_voice,
                    inputs=[text_in, ref_audio, style, sample_dropdown],
                    outputs=[info_box, out_audio, ref_audio_echo],
                    api_name="clone",
                )

            with gr.Tab("Voice → Voice"):
                with gr.Row():
                    with gr.Column(scale=5):
                        s2s_source = gr.Audio(
                            label="Source Speech (content/prosody)",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        s2s_source_sample = gr.Dropdown(
                            label="Or Pick Source Sample",
                            choices=sample_files,
                            value=sample_files[0] if sample_files else None,
                            allow_none=True,
                        )
                        s2s_target_ref = gr.Audio(
                            label="Target Speaker Reference",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        s2s_target_sample = gr.Dropdown(
                            label="Or Pick Target Sample",
                            choices=sample_files,
                            value=sample_files[1] if len(sample_files) > 1 else (sample_files[0] if sample_files else None),
                            allow_none=True,
                        )
                        s2s_btn = gr.Button("🔁 Convert Voice", variant="primary")

                    with gr.Column(scale=5):
                        s2s_info = gr.Textbox(label="Status / Log", interactive=False)
                        s2s_out = gr.Audio(label="Converted Audio", autoplay=True)
                        s2s_src_echo = gr.Audio(label="Source Audio (Echo)")
                        s2s_tgt_echo = gr.Audio(label="Target Reference (Echo)")

                s2s_btn.click(
                    fn=speech_to_speech,
                    inputs=[s2s_source, s2s_target_ref, s2s_target_sample, s2s_source_sample],
                    outputs=[s2s_info, s2s_out, s2s_src_echo, s2s_tgt_echo],
                    api_name="speech_to_speech",
                )

        gr.Markdown(
            """<div class='small-hint'>If the cloned voice doesn't match well, ensure: (1) Clean audio (2) Single speaker (3) ≥3 seconds (4) Limited noise.</div>"""
        )
        # Footer removed per user request

        return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Voice Clone App")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio server")
    return parser.parse_args()


def main():
    args = parse_args()
    demo = build_ui()
    demo.queue()
    demo.launch(share=args.share, server_port=args.port, show_api=False)


if __name__ == "__main__":
    main()
