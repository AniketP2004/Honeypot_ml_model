
# # import joblib
# # from sklearn.metrics import classification_report

# # X_test, ym_test, yb_test = joblib.load(r'Models_metrics\metrics\test_split.pkl')
# # preprocessor = joblib.load(r'Models_metrics\metrics\preprocessor.pkl')
# # X_test_t = preprocessor.transform(X_test)

# # gb_multi = joblib.load(r'Models_metrics\metrics\gradient_boosting_model.pkl')
# # gb_binary = joblib.load(r'Models_metrics\metrics\binary_models.pkl')

# # print("=== MULTICLASS ===")
# # print(classification_report(ym_test, gb_multi.predict(X_test_t)))

# # print("=== BINARY ===")
# # print(classification_report(yb_test, gb_binary.predict(X_test_t)))
# # from sklearn.metrics import confusion_matrix
# # print(confusion_matrix(ym_test, gb_multi.predict(X_test_t)))
# # # import joblib, pandas as pd

# # # X = joblib.load(r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\X_cleaned.pkl')
# # # y_multi = joblib.load(r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\y_multiclass.pkl')
# # # le_multi = joblib.load(r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\Models_metrics\metrics\label_encoder_multi.pkl')

# # # print("classes_:", repr(le_multi.classes_))
# # # labels = le_multi.inverse_transform(y_multi)
# # # print("unique labels:", pd.Series(labels).value_counts())

# import joblib, pandas as pd

# BASE = r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\Models_metrics\metrics'
# X = joblib.load(rf'{BASE}\X_cleaned.pkl')
# y_multi = joblib.load(rf'{BASE}\y_multiclass.pkl')
# le_multi = joblib.load(rf'{BASE}\label_encoder_multi.pkl')

# labels = le_multi.inverse_transform(y_multi)
# recon = X[labels == 'web_recon']

# print(recon['protocol'].value_counts())
# print(recon[['ttl','window_size','payload_size','dst_port','is_exploit_path','is_known_scanner','ip_request_rate']].describe())

import joblib, pandas as pd

BASE = r'C:\Users\project\Desktop\Honeypot_new_repo\Honeypot_attack_classifier\Models_metrics\metrics'
X = joblib.load(rf'{BASE}\X_cleaned.pkl')
y_multi = joblib.load(rf'{BASE}\y_multiclass.pkl')
le_multi = joblib.load(rf'{BASE}\label_encoder_multi.pkl')
preprocessor = joblib.load(rf'{BASE}\preprocessor.pkl')
gb = joblib.load(rf'{BASE}\gradient_boosting_model.pkl')
binary = joblib.load(rf'{BASE}\binary_models.pkl')
le_binary = joblib.load(rf'{BASE}\label_encoder_binary.pkl')

labels = le_multi.inverse_transform(y_multi)
brute_rows = X[labels == 'brute_force']
row = brute_rows.sample(1, random_state=7)
print("RAW ROW:")
print(row.to_dict(orient='records'))

X_t = preprocessor.transform(row)
pred = gb.predict(X_t)[0]
proba = gb.predict_proba(X_t)[0].max()
print("Multiclass predicted:", le_multi.inverse_transform([pred])[0], "conf:", proba)

bpred = binary.predict(X_t)[0]
bproba = binary.predict_proba(X_t)[0].max()
print("Binary predicted:", le_binary.inverse_transform([bpred])[0], "conf:", bproba)