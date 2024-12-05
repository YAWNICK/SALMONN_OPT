import argparse
import json
import os
import csv

import torch
from transformers import WhisperFeatureExtractor

from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample


def main(args):
    # Load configuration
    cfg = Config(args)
    
    # Initialize model
    model = SALMONN.from_config(cfg.config.model)
    model.to(args.device)
    model.eval()

    # Initialize processor
    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

    # Load JSON input file
    with open(args.input_json, 'r') as f:
        data = json.load(f)

    # Ensure 'annotation' key exists in the input JSON
    if 'annotation' not in data:
        raise ValueError("Input JSON must contain an 'annotation' field with an array of samples.")
    
    annotations = data['annotation']
    results = []

    for sample in annotations:
        try:
            wav_path = sample['path']
            prompt = "Listen to the speech and translate it into German."  # TODO rather get prompt from config or as argument?
            text = sample.get('text', '')

            # Prepare input for the model
            samples = prepare_one_sample(wav_path, wav_processor)
            formatted_prompt = [
                cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())
            ]

            # Generate prediction
            with torch.cuda.amp.autocast(dtype=torch.float16):
                prediction = model.generate(samples, cfg.config.generate, prompts=formatted_prompt)[0]

            # Append result
            results.append({
                "path": wav_path,
                "reference_text": text,
                "predicted_translation": prediction,
                "task": sample.get("task", "unknown")  # Keep original task if available
            })

        except Exception as e:
            print(f"Error processing sample {wav_path}: {e}")
            results.append({
                "path": wav_path,
                "error": str(e)
            })

    print(f"Processed {len(results)} samples")

    # Save results to output TSV file
    with open(args.output_tsv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "reference_text", "predicted_translation", "task", "error"],
            delimiter='\t'
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    print(f"Predictions saved to {args.output_tsv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override some settings in the config; key-value pairs in xxx=yyy format"
    )
    parser.add_argument("--input-json", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output-tsv", type=str, required=True, help="Path to save output JSON file")
    
    args = parser.parse_args()
    main(args)