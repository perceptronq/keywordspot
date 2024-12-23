config = {
    # Dataset paths
    "dataset_path": r"C:\Users\shiva\Desktop\KwsSixWords\data\custom_dataset",

    
    # Hyperparameters
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 30,

    # Save paths for models and artifacts
    "model_save_path": "./saved_models/kws_model.pth",             # CNN model
    "dscnn_model_save_path": "./saved_models/dscnn_model.pth",     # DSCNN model (new)
    "label_encoder_save_path": "./saved_models/label_encoder.pkl", # Label encoder

    # Output path for results
    "output_path": "./outputs",

    # Number of classes (update based on your dataset)
    "num_classes": 6
}

def load_config():
    return config
