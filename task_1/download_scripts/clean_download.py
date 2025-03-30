import os
import requests

## download script for list of trial pairs - VoxCeleb1(cleaned)

def download_file(url, output_path):
    """
    Downloads a file from a given URL and writes it to output_path.
    """
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded file to {output_path}")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    veri_test2_url = "https://mm.kaist.ac.kr/datasets/voxceleb/meta/veri_test2.txt"
    output_path = "./datasets/vox1/veri_test2.txt" 
    download_file(veri_test2_url, output_path)
