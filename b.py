# 必要なライブラリをインポートする
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

#画面のタイトルをつける
st.markdown("<h1 style='text-align: center;'>回帰分析アプリ</h1>", unsafe_allow_html=True)

# ファイルアップロード機能を実装する
st.set_option('deprecation.showfileUploaderEncoding', False)

# 1. CSVまたはExcelファイルのインポート
uploaded_file = st.file_uploader("ファイルをアップロードしてください。", type=["csv", "xlsx"])

if uploaded_file is not None:
    # アップロードされたファイルを読み込む
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")

    # 読み込んだデータフレームを表示する
    st.write(df)

    # 説明変数と目的変数の選択
    st.sidebar.title("Regression Analysis")
    X_cols = st.sidebar.multiselect("Select explanatory variables", df.columns)
    y_col = st.sidebar.selectbox("Select a target variable", df.columns)

    # 選択した説明変数と目的変数のデータを取得
    X = df[X_cols].values
    y = df[y_col].values

    # 回帰分析
    reg_model = LinearRegression().fit(X, y)

    # 回帰式の係数と切片の表示
    st.write("### Regression equation")
    st.write(f"{y_col} = {reg_model.intercept_:.2f}", end="")
    for i, col in enumerate(X_cols):
        st.write(f" + {reg_model.coef_[i]:.2f}{col}", end="")
    st.write("")

    # 決定係数の表示
    r2 = reg_model.score(X, y)
    st.write("### Coefficient of determination")
    st.write(f"R^2 = {r2:.2f}")


    # 予測値と実測値のプロット
    y_pred = reg_model.predict(X)
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_pred, y=y, ax=ax)
    ax.set(xlabel="Predicted values", ylabel="Observed values")
    st.pyplot(fig)
    
    # x軸に使用する説明変数を選択する
    x_col = st.selectbox('X軸に使用するカラム', X_cols)
    
    ax.plot(y_pred, y)
    ax.set_xlabel('y_pred')
    ax.set_ylabel('y')
    ax.autoscale()
    plt.show()

    # グラフを描画する
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='True values')
    ax.plot(X, y_pred, color='red', linewidth=3, label='Predicted values')
    ax.set_xlabel(x_col)
    ax.set_ylabel('Value')
    ax.legend()
    st.pyplot(fig)

    # 評価指標の表示
    # 評価指標の表示
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    intercept = reg_model.intercept_
    coefficients = reg_model.coef_
    t_values = coefficients / rmse
    p_values = 2 * (1 - stats.t.cdf(abs(coefficients) / rmse, len(X) - len(X_cols) - 1))
    corr = np.corrcoef(y_pred, y)[0, 1]
    print(f"Correlation coefficient: {corr:.2f}")
    st.write("### Evaluation metrics")
    st.write(pd.DataFrame({
        "Metric": ["MSE", "RMSE", "Intercept"] + [f"Coefficient for {col}" for col in X_cols] + [f"t-value for {col}" for col in X_cols] + ["corr"],
        "Value": [mse, rmse, intercept] + list(coefficients) + list(t_values) + [corr]
    }))


