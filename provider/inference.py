import sys
import json
import torch

DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def run_text_generation(model_name, input_data, config):
    try:
        from transformers import pipeline
        pipe = pipeline("text-generation", model=model_name, device=DEVICE)
        result = pipe(input_data, **config)
        print(result[0]["generated_text"])
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        sys.exit(1)


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


TASKS = {
    "TEXT_INFERENCE":    run_text_generation,
    "SPEECH_RECOGNITION": run_speech_recognition,
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
