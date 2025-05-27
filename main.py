import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk

# 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv("Delivery.csv")

st.title("🚚 배달 데이터 군집 분석 + 지도 시각화 웹앱")

df = load_data()

# 'Num' 컬럼 제거
df = df.drop(columns=['Num'], errors='ignore')

# 컬럼명 표준화
df = df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})

st.write("🔍 데이터 미리보기", df.head())

# 사용자 선택: 군집에 사용할 컬럼
features = st.multiselect("군집 분석에 사용할 컬럼을 선택하세요", df.select_dtypes(include=['float64', 'int64']).columns.difference(['latitude', 'longitude']))

if len(features) >= 2:
    # 결측값 제거 후 군집 분석
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 군집 수 선택
    n_clusters = st.slider("군집 수 (KMeans)", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # PCA로 2D 변환 및 시각화
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Cluster"] = labels

    st.subheader("🎯 2D 군집 결과")
    fig, ax = plt.subplots()
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="Set2", ax=ax)
    st.pyplot(fig)

    # 군집 결과 저장
    df["Cluster"] = labels

    st.subheader("📈 군집별 평균 값")
    st.write(df.groupby("Cluster")[features].mean())

    # 지도 시각화
    st.subheader("🗺️ 군집 결과 지도 시각화")

    if 'latitude' in df.columns and 'longitude' in df.columns:
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=df['latitude'].mean(),
                longitude=df['longitude'].mean(),
                zoom=10,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df,
                    get_position='[longitude, latitude]',
                    get_color='[Cluster * 50, 100, 150, 160]',
                    get_radius=100,
                    pickable=True,
                )
            ]
        ))
    else:
        st.warning("위도/경도 정보가 없어 지도를 그릴 수 없습니다.")
else:
    st.warning("2개 이상의 수치형 컬럼을 선택해주세요.")
