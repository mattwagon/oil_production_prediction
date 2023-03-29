from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

def basic_mvp():

    baseline_model = LinearRegression()

    baseline_model.fit(X_train_scaled, y_train)

    baseline_pred_val = baseline_model.predict(X_val_scaled)
    baseline_pred_test = baseline_model.predict(X_test_scaled)
    baseline_pred_test2 = baseline_model.predict(X_test2_scaled)
    baseline_mae_val = np.mean(np.abs(baseline_pred_val - y_val), axis=0)
    baseline_mae_test = np.mean(np.abs(baseline_pred_test - y_test), axis=0)
    baseline_mae_test2 = np.mean(np.abs(baseline_pred_test2 - y_test2), axis=0)
