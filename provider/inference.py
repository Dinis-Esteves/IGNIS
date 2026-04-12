import sys
import json
import torch

DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def make_pipeline(task, model_name, **kwargs):
    from transformers import pipeline
    return pipeline(task, model=model_name, device=DEVICE, **kwargs)


import json as _json  # alias to avoid shadowing the param name


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


def run_zero_shot_audio_classification(model_name, input_data, config):
    # input_data: "audio_path|||label1,label2,label3"
    try:
        parts = input_data.split("|||", 1)
        if len(parts) != 2:
            print("model error: input must be 'audio_path|||label1,label2,label3'", file=sys.stderr)
            sys.exit(1)
        audio_path = parts[0].strip()
        candidate_labels = [l.strip() for l in parts[1].split(",")]
        pipe = make_pipeline("zero-shot-audio-classification", model_name)
        result = pipe(audio_path, candidate_labels=candidate_labels, **config)
        for r in result:
            print(f"{r['label']}: {r['score']:.4f}")
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


def run_token_classification(model_name, input_data, config):
    try:
        pipe = make_pipeline("token-classification", model_name)
        result = pipe(input_data, **config)
        for r in result:
            print(f"{r['word']} [{r['entity']}]: {r['score']:.4f}")
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_zero_shot_classification(model_name, input_data, config):
    # input_data: "text|||label1,label2,label3"
    try:
        parts = input_data.split("|||", 1)
        if len(parts) != 2:
            print("model error: input must be 'text|||label1,label2,label3'", file=sys.stderr)
            sys.exit(1)
        text = parts[0].strip()
        candidate_labels = [l.strip() for l in parts[1].split(",")]
        pipe = make_pipeline("zero-shot-classification", model_name)
        result = pipe(text, candidate_labels=candidate_labels, **config)
        for label, score in zip(result["labels"], result["scores"]):
            print(f"{label}: {score:.4f}")
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_table_question_answering(model_name, input_data, config):
    # input_data: "question|||{\"col\":[val,...],...}"  (JSON table)
    try:
        parts = input_data.split("|||", 1)
        if len(parts) != 2:
            print("model error: input must be 'question|||{json_table}'", file=sys.stderr)
            sys.exit(1)
        question = parts[0].strip()
        table = _json.loads(parts[1].strip())
        pipe = make_pipeline("table-question-answering", model_name)
        result = pipe(table=table, query=question, **config)
        print(result["answer"])
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_feature_extraction(model_name, input_data, config):
    try:
        pipe = make_pipeline("feature-extraction", model_name)
        result = pipe(input_data, **config)
        # result is a nested list; print as JSON
        import numpy as np
        arr = np.array(result)
        print(_json.dumps(arr.tolist()))
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


def run_zero_shot_image_classification(model_name, input_data, config):
    # input_data: "image_path|||label1,label2,label3"
    try:
        parts = input_data.split("|||", 1)
        if len(parts) != 2:
            print("model error: input must be 'image_path|||label1,label2,label3'", file=sys.stderr)
            sys.exit(1)
        image_path = parts[0].strip()
        candidate_labels = [l.strip() for l in parts[1].split(",")]
        pipe = make_pipeline("zero-shot-image-classification", model_name)
        result = pipe(image_path, candidate_labels=candidate_labels, **config)
        for r in result:
            print(f"{r['label']}: {r['score']:.4f}")
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_zero_shot_object_detection(model_name, input_data, config):
    # input_data: "image_path|||label1,label2,label3"
    try:
        parts = input_data.split("|||", 1)
        if len(parts) != 2:
            print("model error: input must be 'image_path|||label1,label2,label3'", file=sys.stderr)
            sys.exit(1)
        image_path = parts[0].strip()
        candidate_labels = [l.strip() for l in parts[1].split(",")]
        pipe = make_pipeline("zero-shot-object-detection", model_name)
        result = pipe(image_path, candidate_labels=candidate_labels, **config)
        for r in result:
            print(f"{r['label']}: {r['score']:.4f} @ {r['box']}")
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_mask_generation(model_name, input_data, config):
    try:
        pipe = make_pipeline("mask-generation", model_name)
        result = pipe(input_data, **config)
        masks = result["masks"]
        print(f"{len(masks)} masks generated")
        for i, m in enumerate(masks):
            print(f"  mask[{i}]: shape={m.shape}")
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_image_feature_extraction(model_name, input_data, config):
    try:
        pipe = make_pipeline("image-feature-extraction", model_name)
        result = pipe(input_data, **config)
        import numpy as np
        arr = np.array(result)
        print(_json.dumps(arr.tolist()))
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_keypoint_matching(model_name, input_data, config):
    # input_data: "image_path1|||image_path2"
    try:
        parts = input_data.split("|||", 1)
        if len(parts) != 2:
            print("model error: input must be 'image_path1|||image_path2'", file=sys.stderr)
            sys.exit(1)
        from PIL import Image
        img1 = Image.open(parts[0].strip())
        img2 = Image.open(parts[1].strip())
        pipe = make_pipeline("keypoint-matching", model_name)
        result = pipe({"image": img1, "image2": img2}, **config)
        kp = result.get("keypoints", [])
        matches = result.get("matches", [])
        print(f"keypoints: {len(kp)}, matches: {len(matches)}")
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


def run_image_text_to_text(model_name, input_data, config):
    # input_data: "prompt" OR "prompt|||image_path"
    try:
        parts = input_data.split("|||", 1)
        pipe = make_pipeline("image-text-to-text", model_name)
        if len(parts) == 2:
            prompt = parts[0].strip()
            image_path = parts[1].strip()
            content = [
                {"type": "image", "url": image_path},
                {"type": "text", "text": prompt},
            ]
        else:
            content = [{"type": "text", "text": input_data.strip()}]
        messages = [{"role": "user", "content": content}]
        result = pipe(messages, **config)
        last = result[0]["generated_text"][-1]
        # content may be a string or a list depending on model
        text = last["content"] if isinstance(last["content"], str) else last["content"][-1]["text"]
        print(text)
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


def run_any_to_any(model_name, input_data, config):
    try:
        pipe = make_pipeline("any-to-any", model_name)
        result = pipe(input_data, **config)
        print(result)
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


# ── Dispatch ─────────────────────────────────────────────────────────────────

TASKS = {
    # Audio
    "SPEECH_RECOGNITION":             run_speech_recognition,
    "AUDIO_CLASSIFICATION":           run_audio_classification,
    "KEYWORD_SPOTTING":               run_keyword_spotting,
    "TEXT_TO_SPEECH":                 run_text_to_speech,
    "TEXT_TO_AUDIO":                  run_text_to_audio,
    "ZERO_SHOT_AUDIO_CLASSIFICATION": run_zero_shot_audio_classification,
    # NLP
    "TEXT_INFERENCE":          run_text_generation,
    "SUMMARIZATION":           run_summarization,
    "TRANSLATION":             run_translation,
    "QUESTION_ANSWERING":      run_question_answering,
    "TEXT_CLASSIFICATION":     run_text_classification,
    "FILL_MASK":               run_fill_mask,
    "TOKEN_CLASSIFICATION":    run_token_classification,
    "ZERO_SHOT_CLASSIFICATION": run_zero_shot_classification,
    "TABLE_QUESTION_ANSWERING": run_table_question_answering,
    "FEATURE_EXTRACTION":      run_feature_extraction,
    # Computer Vision
    "IMAGE_CLASSIFICATION":           run_image_classification,
    "OBJECT_DETECTION":               run_object_detection,
    "IMAGE_CAPTIONING":               run_image_captioning,
    "DEPTH_ESTIMATION":               run_depth_estimation,
    "VIDEO_CLASSIFICATION":           run_video_classification,
    "IMAGE_SEGMENTATION":             run_image_segmentation,
    "ZERO_SHOT_IMAGE_CLASSIFICATION": run_zero_shot_image_classification,
    "ZERO_SHOT_OBJECT_DETECTION":     run_zero_shot_object_detection,
    "MASK_GENERATION":                run_mask_generation,
    "IMAGE_FEATURE_EXTRACTION":       run_image_feature_extraction,
    "KEYPOINT_MATCHING":              run_keypoint_matching,
    # Multimodal
    "VISUAL_QA":          run_visual_qa,
    "DOCUMENT_QA":        run_document_qa,
    "IMAGE_TEXT_TO_TEXT": run_image_text_to_text,
    "ANY_TO_ANY":         run_any_to_any,
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
