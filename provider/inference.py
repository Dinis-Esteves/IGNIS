import sys
import json
import torch

DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def make_pipeline(task, model_name, **kwargs):
    from transformers import pipeline
    return pipeline(task, model=model_name, device=DEVICE, **kwargs)


# ── Audio ────────────────────────────────────────────────────────────────────

def run_speech_recognition(model_name, input_data, config):
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, dtype=DTYPE, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(DEVICE if DEVICE >= 0 else "cpu")
        processor = AutoProcessor.from_pretrained(model_name)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=DTYPE,
            device=DEVICE,
        )
        result = pipe(input_data, **config)
        print(result["text"])
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_audio_classification(model_name, input_data, config):
    try:
        pipe = make_pipeline("audio-classification", model_name)
        result = pipe(input_data, **config)
        for r in result:
            print(f"{r['label']}: {r['score']:.4f}")
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_keyword_spotting(model_name, input_data, config):
    try:
        pipe = make_pipeline("audio-classification", model_name)
        result = pipe(input_data, **config)
        for r in result:
            print(f"{r['label']}: {r['score']:.4f}")
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_text_to_speech(model_name, input_data, config):
    try:
        from transformers import pipeline
        import soundfile as sf
        pipe = pipeline("text-to-speech", model=model_name)
        result = pipe(input_data, **config)
        output_path = "/tmp/ignis_tts_output.wav"
        sf.write(output_path, result["audio"][0], result["sampling_rate"])
        print(output_path)
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_text_to_audio(model_name, input_data, config):
    try:
        from transformers import pipeline
        import soundfile as sf
        pipe = pipeline("text-to-audio", model=model_name)
        result = pipe(input_data, **config)
        output_path = "/tmp/ignis_audio_output.wav"
        sf.write(output_path, result["audio"][0], result["sampling_rate"])
        print(output_path)
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


# ── NLP ──────────────────────────────────────────────────────────────────────

def run_text_generation(model_name, input_data, config):
    try:
        pipe = make_pipeline("text-generation", model_name)
        result = pipe(input_data, **config)
        print(result[0]["generated_text"])
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_summarization(model_name, input_data, config):
    try:
        pipe = make_pipeline("summarization", model_name)
        result = pipe(input_data, **config)
        print(result[0]["summary_text"])
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_translation(model_name, input_data, config):
    try:
        pipe = make_pipeline("translation", model_name)
        result = pipe(input_data, **config)
        print(result[0]["translation_text"])
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_question_answering(model_name, input_data, config):
    try:
        # input_data expected as "question|||context"
        parts = input_data.split("|||", 1)
        if len(parts) != 2:
            print("model error: input must be 'question|||context'", file=sys.stderr)
            sys.exit(1)
        pipe = make_pipeline("question-answering", model_name)
        result = pipe(question=parts[0].strip(), context=parts[1].strip(), **config)
        print(result["answer"])
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_text_classification(model_name, input_data, config):
    try:
        pipe = make_pipeline("text-classification", model_name)
        result = pipe(input_data, **config)
        print(f"{result[0]['label']}: {result[0]['score']:.4f}")
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_fill_mask(model_name, input_data, config):
    try:
        pipe = make_pipeline("fill-mask", model_name)
        result = pipe(input_data, **config)
        for r in result:
            print(f"{r['token_str']}: {r['score']:.4f}")
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


# ── Computer Vision ──────────────────────────────────────────────────────────

def run_image_classification(model_name, input_data, config):
    try:
        pipe = make_pipeline("image-classification", model_name)
        result = pipe(input_data, **config)
        for r in result:
            print(f"{r['label']}: {r['score']:.4f}")
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_object_detection(model_name, input_data, config):
    try:
        pipe = make_pipeline("object-detection", model_name)
        result = pipe(input_data, **config)
        for r in result:
            print(f"{r['label']}: {r['score']:.4f} @ {r['box']}")
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_image_captioning(model_name, input_data, config):
    try:
        pipe = make_pipeline("image-to-text", model_name)
        result = pipe(input_data, **config)
        print(result[0]["generated_text"])
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_depth_estimation(model_name, input_data, config):
    try:
        pipe = make_pipeline("depth-estimation", model_name)
        result = pipe(input_data, **config)
        print(f"depth map shape: {result['depth'].size}")
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_video_classification(model_name, input_data, config):
    try:
        pipe = make_pipeline("video-classification", model_name)
        result = pipe(input_data, **config)
        for r in result:
            print(f"{r['label']}: {r['score']:.4f}")
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_image_segmentation(model_name, input_data, config):
    try:
        pipe = make_pipeline("image-segmentation", model_name)
        result = pipe(input_data, **config)
        for r in result:
            print(f"{r['label']}: {r['score']:.4f}")
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


# ── Multimodal ───────────────────────────────────────────────────────────────

def run_visual_qa(model_name, input_data, config):
    try:
        # input_data expected as "question|||image_path"
        parts = input_data.split("|||", 1)
        if len(parts) != 2:
            print("model error: input must be 'question|||image_path'", file=sys.stderr)
            sys.exit(1)
        pipe = make_pipeline("visual-question-answering", model_name)
        result = pipe(image=parts[1].strip(), question=parts[0].strip(), **config)
        print(result[0]["answer"])
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_document_qa(model_name, input_data, config):
    try:
        # input_data expected as "question|||image_path"
        parts = input_data.split("|||", 1)
        if len(parts) != 2:
            print("model error: input must be 'question|||image_path'", file=sys.stderr)
            sys.exit(1)
        pipe = make_pipeline("document-question-answering", model_name)
        result = pipe(image=parts[1].strip(), question=parts[0].strip(), **config)
        print(result[0]["answer"])
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


# ── Dispatch ─────────────────────────────────────────────────────────────────

TASKS = {
    "SPEECH_RECOGNITION":  run_speech_recognition,
    "AUDIO_CLASSIFICATION": run_audio_classification,
    "KEYWORD_SPOTTING":    run_keyword_spotting,
    "TEXT_TO_SPEECH":      run_text_to_speech,
    "TEXT_TO_AUDIO":       run_text_to_audio,
    "TEXT_INFERENCE":      run_text_generation,
    "SUMMARIZATION":       run_summarization,
    "TRANSLATION":         run_translation,
    "QUESTION_ANSWERING":  run_question_answering,
    "TEXT_CLASSIFICATION": run_text_classification,
    "FILL_MASK":           run_fill_mask,
    "IMAGE_CLASSIFICATION": run_image_classification,
    "OBJECT_DETECTION":    run_object_detection,
    "IMAGE_CAPTIONING":    run_image_captioning,
    "DEPTH_ESTIMATION":    run_depth_estimation,
    "VIDEO_CLASSIFICATION": run_video_classification,
    "IMAGE_SEGMENTATION":  run_image_segmentation,
    "VISUAL_QA":           run_visual_qa,
    "DOCUMENT_QA":         run_document_qa,
}


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: inference.py <task_type> <model_name> <input_data> [json_config]", file=sys.stderr)
        sys.exit(1)

    task       = sys.argv[1]
    model_name = sys.argv[2]
    input_data = sys.argv[3]
    config     = {}

    if len(sys.argv) >= 5:
        try:
            config = json.loads(sys.argv[4])
        except json.JSONDecodeError as e:
            print(f"invalid config JSON: {e}", file=sys.stderr)
            sys.exit(1)

    fn = TASKS.get(task)
    if not fn:
        print(f"unknown task: {task}", file=sys.stderr)
        sys.exit(1)

    fn(model_name, input_data, config)
