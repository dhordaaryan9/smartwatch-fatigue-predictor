import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib  

df = pd.read_csv('model_ready_telemetry.csv')

df.dropna()

x = df.drop(columns=['Fatigue_State'])
y = df['Fatigue_State']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=7,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train,y_train)

predictions = model.predict(X_test)
print(classification_report(y_test,predictions))

joblib.dump(model, 'fatigue_classifier.pkl')