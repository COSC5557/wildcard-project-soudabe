import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import shap




data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

df['target'] = data.target

df.head()

y=df['target'].to_frame()
X=df[df.columns.difference(['target'])]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


xgb_mod=xgb.XGBClassifier(random_state=42,gpu_id=0)
xgb_mod=xgb_mod.fit(X_train,y_train.values.ravel())


y_pred = xgb_mod.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# Generate the Tree explainer and SHAP values
explainer = shap.TreeExplainer(xgb_mod)
shap_values = explainer.shap_values(X)
expected_value = explainer.expected_value


shap.summary_plot(shap_values, X,title="SHAP summary plot")

shap.summary_plot(shap_values, X,plot_type="bar")


shap.plots._waterfall.waterfall_legacy(expected_value, shap_values[79], features=X.loc[79,:], feature_names=X.columns, max_display=15, show=True)


shap.dependence_plot("worst concave points", shap_values, X, interaction_index="mean concave points")


for name in X_train.columns:
     shap.dependence_plot(name, shap_values, X)
shap.dependence_plot("worst concave points", shap_values, X, interaction_index="mean concave points")


shap.force_plot(explainer.expected_value, shap_values[:100,:], X.iloc[:100,:])


shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])


shap.decision_plot(expected_value, shap_values[79],link='logit' ,features=X.loc[79,:], feature_names=(X.columns.tolist()),show=True,title="Decision Plot")



df.head()




# Resources:
# https://towardsdatascience.com/explainable-ai-xai-a-guide-to-7-packages-in-python-to-explain-your-models-932967f0634b

