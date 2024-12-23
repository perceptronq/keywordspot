import argparse
from src.train import train_model  # For CNN training
from src.train_dscnn import train_dscnn  # For DSCNN training
from src.inference import run_inference as cnn_inference  # For CNN inference
from src.inference_dscnn import run_inference as dscnn_inference  # For DSCNN inference
from src.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Keyword Spotting")
    parser.add_argument(
        "mode", choices=["train_cnn", "train_dscnn", "infer_cnn", "infer_dscnn"], 
        help="Select mode: train_cnn, train_dscnn, infer_cnn, infer_dscnn"
    )
    parser.add_argument("--audio_path", type=str, help="Path to the audio file for inference")
    args = parser.parse_args()

    config = load_config()

    if args.mode == "train_cnn":
        print("Starting CNN training...")
        train_model(config)
    elif args.mode == "train_dscnn":
        print("Starting DSCNN training...")
        train_dscnn(config)
    elif args.mode == "infer_cnn":
        if not args.audio_path:
            print("Error: --audio_path is required for inference.")
            return
        print("Running inference with CNN...")
        cnn_inference(config, args.audio_path)
    elif args.mode == "infer_dscnn":
        if not args.audio_path:
            print("Error: --audio_path is required for inference.")
            return
        print("Running inference with DSCNN...")
        dscnn_inference(config, args.audio_path)
    else:
        print("Invalid mode. Use --help for available options.")

if __name__ == "__main__":
    main()
