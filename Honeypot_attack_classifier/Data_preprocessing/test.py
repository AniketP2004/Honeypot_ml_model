import joblib

BASE = r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\Models_metrics'

y_multi = joblib.load(rf'{BASE}\y_multiclass.pkl')
le_multi = joblib.load(rf'{BASE}\label_encoder_multi.pkl')

labels = le_multi.inverse_transform(y_multi)

import pandas as pd
counts = pd.Series(labels).value_counts()

print("Classes found:", le_multi.classes_)
print("\nCounts per class:")
print(counts)