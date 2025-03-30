from config import *
from dataloader import load_audio_files
from stats import create_feature_dataframe
from model import prepare_data, train_and_evaluate

audio_files, labels = load_audio_files(DATASET_PATH, LANGUAGES)
df = create_feature_dataframe(audio_files, labels, N_MFCC)

X, y = prepare_data(df)
accuracy, report = train_and_evaluate(X, y, TEST_SIZE, RANDOM_STATE)

with open("classification_results.txt", "w") as f:
    f.write(f"Classification Accuracy: {accuracy:.2f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    

print(f"Classification Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)
