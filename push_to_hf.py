import os
import sys
import shutil
import tempfile
import subprocess
from huggingface_hub import login, HfApi, HfFolder
from dotenv import load_dotenv

def push_to_huggingface(token=None, space_name="haepada/nps_test"):
    """
    Login to Hugging Face and push the current directory to the Space.
    
    Args:
        token (str, optional): Hugging Face token. If None, it will try to read from .env file.
        space_name (str, optional): Name of the Hugging Face Space. Default is "haepada/nps_test".
    """
    # Try to load token from .env file if not provided
    if token is None:
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        
    # If still no token, ask for it
    if token is None:
        token = input("Please enter your Hugging Face token (from https://huggingface.co/settings/tokens): ")
    
    # Login to Hugging Face
    print("Logging in to Hugging Face...")
    login(token=token, add_to_git_credential=True)
    
    # Create API object
    api = HfApi()
    
    # Check login status
    try:
        user_info = api.whoami()
        print(f"Successfully logged in as: {user_info['name']}")
    except Exception as e:
        print(f"Login failed: {str(e)}")
        return
    
    # Create a temporary directory for the filtered files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy all files except those in .gitignore
        print("Preparing files for upload...")
        
        # Copy all files
        for item in os.listdir('.'):
            if item != '.git' and item != '__pycache__' and not item.endswith('.pyc') and item != '.env':
                src_path = os.path.join('.', item)
                dst_path = os.path.join(temp_dir, item)
                
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
        
        # Create empty directories that might be in .gitignore
        os.makedirs(os.path.join(temp_dir, 'data', 'personas'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'data', 'conversations'), exist_ok=True)
        
        # Create .gitkeep files
        open(os.path.join(temp_dir, 'data', 'personas', '.gitkeep'), 'w').close()
        open(os.path.join(temp_dir, 'data', 'conversations', '.gitkeep'), 'w').close()
        
        # Upload to Hugging Face
        print(f"Uploading to Hugging Face Space: {space_name}")
        try:
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=space_name,
                repo_type="space",
                commit_message="Update nompang_test app with 127 personality variables",
                ignore_patterns=["*.pyc", "__pycache__", ".DS_Store", ".env"]
            )
            print("Successfully uploaded to Hugging Face Space!")
            print(f"Visit: https://huggingface.co/spaces/{space_name}")
        except Exception as e:
            print(f"Upload failed: {str(e)}")

if __name__ == "__main__":
    # If token is passed as an argument, use it
    token = sys.argv[1] if len(sys.argv) > 1 else None
    push_to_huggingface(token=token) 