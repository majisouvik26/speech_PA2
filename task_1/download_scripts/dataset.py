import gdown
import os
import zipfile

## download script for vox

def download_and_extract_folder(folder_url_or_id, output_dir):
    """
    Download all files from a Google Drive folder (using gdown.download_folder)
    and extract any .zip files found.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gdown.download_folder(url=folder_url_or_id, output=output_dir, quiet=False, use_cookies=False)
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                print(f"Extracting {zip_path}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(root)
                os.remove(zip_path)
                print(f"Removed {zip_path}")

voxceleb_url = "https://drive.google.com/drive/folders/1qypIUgCoPfp5mCqPCbBobnw9hJKlW1Xm"
download_and_extract_folder(voxceleb_url, "./datasets")

