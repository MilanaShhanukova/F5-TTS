import argparse
import json
import subprocess
from pathlib import Path
from ruaccent import RUAccent

def parse_args():
    parser = argparse.ArgumentParser(description='Run TTS inference with various examples using JSON.')
    parser.add_argument('--model_path', required=True, 
                      help='Path to the model checkpoint')
    parser.add_argument('--output_dir', required=True,
                      help='Directory to save the output files')
    parser.add_argument('--examples_file', required=True,
                      help='Path to the JSON file containing examples')
    parser.add_argument('--gpu_id', default=1, type=int,
                      help='GPU ID to use for inference')
    return parser.parse_args()

def load_examples(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['examples']

def run_inference(source_lang, target_lang, ref_text, gen_text, file_name, 
                 audio_path, model_path, output_dir, gpu_id=1, fix_duration=None):
    accentizer = RUAccent()
    accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False)

    if source_lang == "rus":
        ref_text = accentizer.process_all(ref_text)
    if target_lang == "rus":
        gen_text = accentizer.process_all(gen_text)
    
    # Add quotes around texts
    ref_text = f'"{ref_text}"'
    gen_text = f'"{gen_text}"'

    print(f"Processing file: {file_name}")
    print(f"Reference text: {ref_text}")
    print(f"Generation text: {gen_text}")

    command = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "python",
        "f5_tts/infer/infer_cli.py",
        "-r", audio_path,
        "--ref_text", ref_text,
        "--gen_text", gen_text,
        "--file_name", file_name,
        "-p", model_path,
        "--output_dir", output_dir
    ]
    
    if fix_duration is not None:
        command.extend(["--fix_duration", str(fix_duration)])

    try:
        subprocess.run(" ".join(command), shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {file_name}: {e}")

def main():
    args = parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    examples = load_examples(args.examples_file)
    
    for example in examples:
        run_inference(
            **example,
            model_path=args.model_path,
            output_dir=args.output_dir,
            gpu_id=args.gpu_id,
        )

if __name__ == "__main__":
    main()