import os

def load_audio_files(base_path, languages):
    """Load audio files with verified directory structure"""
    file_paths = []
    labels = []
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset path {base_path} not found")
    
    lang_detection_dir = os.path.join(base_path, "LanguageDetectionDataset")
    if not os.path.exists(lang_detection_dir):
        raise FileNotFoundError(f"LanguageDetectionDataset folder not found in {base_path}")
    
    existing_lang_dirs = os.listdir(lang_detection_dir)
    
    for lang in languages:
        matched_dir = next((d for d in existing_lang_dirs if d.lower() == lang.lower()), None)
        
        if not matched_dir:
            print(f"Warning: Directory for language '{lang}' not found. Available: {existing_lang_dirs}")
            continue
            
        lang_dir = os.path.join(lang_detection_dir, matched_dir)
        
        if not os.listdir(lang_dir):
            print(f"Warning: No files found in {lang_dir}")
            continue
            
        for file in os.listdir(lang_dir):
            if file.lower().endswith(".mp3"):
                file_paths.append(os.path.join(lang_dir, file))
                labels.append(matched_dir)
                
    print(f"Successfully loaded {len(file_paths)} audio files")
    return file_paths, labels