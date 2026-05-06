from huggingface_hub import HfApi
import os

def deploy_model(repo_id, model_path='distilbert-reviews-genres'):
    api = HfApi()
    
    print(f"Uploading model from {model_path} to HF Hub: {repo_id}")
    
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
        )
        print("Successfully deployed model to Hugging Face Hub!")
    except Exception as e:
        print(f"Error during deployment: {e}")

if __name__ == '__main__':
    # Replace with your actual HF username
    USER = "AzDevops143" 
    deploy_model(repo_id=f"{USER}/distilbert-reviews-genres")
