import sys
from transformers import pipeline

def main():
    if len(sys.argv) < 3:
        print("usage: inference.py <model_name> <input_text>", file=sys.stderr)
        sys.exit(1)

    model_name = sys.argv[1]
    input_text = sys.argv[2]

    pipe = pipeline("text-generation", model=model_name)
    result = pipe(input_text, max_new_tokens=200)
    print(result[0]["generated_text"])

if __name__ == "__main__":
    main()
