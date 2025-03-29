from config import *
from dataloader import load_audio_files
from stat import create_feature_dataframe
from model import prepare_data, train_and_evaluate

# load and extract features
audio_files, labels = load_audio_files(DATASET_PATH, LANGUAGES)
df = create_feature_dataframe(audio_files, labels, N_MFCC)

# prepare data
X, y = prepare_data(df)
accuracy, report = train_and_evaluate(X, y, TEST_SIZE, RANDOM_STATE)

# classification report
print(f"Classification Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)
