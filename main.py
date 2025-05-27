import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv("Delivery.csv")

st.title("🚚 배달 데이터 군집 분석 웹앱")
df = load_data()
st.write("🔍 데이터 미리보기", df.head())

# 사용자 선택: 군집에 사용할 컬럼
features = st.multiselect("군집 분석에 사용할 컬럼을 선택하세요", df.select_dtypes(include=['float64', 'int64']).columns)

if len(features) >= 2:
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 군집 수 선택
    n_clusters = st.slider("군집 수 (KMeans)", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # PCA로 2D 변환
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 시각화
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Cluster"] = labels

    st.subheader("🎯 2D 군집 결과")
    fig, ax = plt.subplots()
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="Set2", ax=ax)
    st.pyplot(fig)

    # 군집별 평균 정보
    df["Cluster"] = labels
    st.subheader("📈 군집별 평균 값")
    st.write(df.groupby("Cluster")[features].mean())
else:
    st.warning("2개 이상의 수치형 컬럼을 선택해주세요.")
