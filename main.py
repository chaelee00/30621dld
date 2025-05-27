import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 로드
@st.cache_data
def load_data():
    df = pd.read_csv("Delivery.csv")
    return df

st.title("📦 배달 데이터 시각화 대시보드")
df = load_data()

# 데이터프레임 미리보기
st.subheader("데이터 미리보기")
st.write(df.head())

# 시각화 예시: 배송 시간 분포
st.subheader("배송 시간 분포")
fig, ax = plt.subplots()
sns.histplot(df['Delivery Time'], kde=True, ax=ax)
st.pyplot(fig)

