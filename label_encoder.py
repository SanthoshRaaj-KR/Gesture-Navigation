import pickle
from sklearn.preprocessing import LabelEncoder

GESTURES = ["restore_browser", "minimize_browser", "tab_left", "tab_right"]

label_encoder = LabelEncoder()
label_encoder.fit(GESTURES) 

# Paste your existing encoder object if it's in memory
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Label encoder saved!")
