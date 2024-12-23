import torch
from src.model import KeywordSpottingModel  # CNN model
from src.dscnn_model import DSCNN           # DSCNN model
from src.config import load_config
import time

def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_memory_usage(model, data_type=torch.float32):
    """
    Estimate memory usage of the model based on its parameters and data type.
    """
    total_params = count_parameters(model)
    bytes_per_param = torch.tensor([], dtype=data_type).element_size()
    memory_usage_mb = (total_params * bytes_per_param) / (1024 ** 2)
    return memory_usage_mb

def measure_training_time(model, dummy_loader, device):
    """
    Measure training time for a few iterations using dummy data.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    start_time = time.time()
    for i, (inputs, labels) in enumerate(dummy_loader):
        if i == 10:  # Simulate 10 batches
            break
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    end_time = time.time()

    return end_time - start_time

def analyze_model(model_name):
    """
    Analyze memory usage and training time for the given model.
    """
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "cnn":
        print("Analyzing CNN model...")
        model = KeywordSpottingModel(num_classes=config["num_classes"]).to(device)
        model.load_state_dict(torch.load(config["model_save_path"], map_location=device))
    elif model_name == "dscnn":
        print("Analyzing DSCNN model...")
        model = DSCNN(num_classes=config["num_classes"]).to(device)
        model.load_state_dict(torch.load(config["dscnn_model_save_path"], map_location=device))
    else:
        print("Error: Unknown model name. Use 'cnn' or 'dscnn'.")
        return

    # Memory usage
    memory_usage = estimate_memory_usage(model)
    print(f"Model Memory Usage: {memory_usage:.2f} MB")

    # Dummy data loader for time measurement
    dummy_inputs = torch.randn(config["batch_size"], 1, 64, 64).to(device)
    dummy_labels = torch.randint(0, config["num_classes"], (config["batch_size"],)).to(device)
    dummy_loader = [(dummy_inputs, dummy_labels)] * 10  # Simulate 10 batches

    # Training time
    training_time = measure_training_time(model, dummy_loader, device)
    print(f"Estimated Training Time for 10 batches: {training_time:.2f} seconds")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze CNN and DSCNN models")
    parser.add_argument("model", choices=["cnn", "dscnn"], help="Model to analyze: cnn or dscnn")
    args = parser.parse_args()

    analyze_model(args.model)
