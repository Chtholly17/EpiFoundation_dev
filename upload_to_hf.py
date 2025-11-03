#!/usr/bin/env python3
"""
Script to upload a PyTorch model to Hugging Face Hub
"""
import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path

def upload_model_to_hf(model_path: str, repo_name: str, description: str = "Fine-tuned model"):
    """
    Upload a PyTorch model to Hugging Face Hub
    
    Args:
        model_path: Path to the .pth file
        repo_name: Repository name (without username)
        description: Description for the model
    """
    api = HfApi()
    username = "Chtholly17"
    repo_id = f"{username}/{repo_name}"
    
    print(f"Creating repository: {repo_id}")
    
    try:
        # Try to create the repository (will succeed if it doesn't exist)
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            exist_ok=True
        )
        print(f"✓ Repository created/verified: {repo_id}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"✓ Repository already exists: {repo_id}")
        else:
            raise
    
    # Upload the file
    print(f"\nUploading model file: {model_path}")
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="pytorch_model.bin",  # Standard HF name
        repo_id=repo_id,
        repo_type="model",
    )
    
    # Also upload with the original name
    original_filename = Path(model_path).name
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=original_filename,
        repo_id=repo_id,
        repo_type="model",
    )
    
    print(f"\n✓ Successfully uploaded model to: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "experiment/kidney_atlas_zero_inflated_finetune/ckpts/finetuned.pth"
    REPO_NAME = "kidney-atlas-zero-inflated"
    DESCRIPTION = "Fine-tuned scTransformer model for BM+MC atlas with zero-inflated loss"
    
    # Check if file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: File not found at {MODEL_PATH}")
        exit(1)
    
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # Size in MB
    print(f"Model file size: {file_size:.2f} MB")
    
    # Upload
    upload_model_to_hf(MODEL_PATH, REPO_NAME, DESCRIPTION)



