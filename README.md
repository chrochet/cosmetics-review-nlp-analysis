# cosmetics-review-text-mining
웹 리뷰 기반 화장품 만족·불만족 요인 탐색 

---

## 🖥️ 프로젝트 소개
본 프로젝트는 웹 페이지(화장품 리뷰)의 HTML 데이터를 수집한 뒤,  
텍스트 전처리 과정을 거쳐 키워드 분석(TF-IDF)과 토픽 모델링(LDA)을 수행하여  
리뷰에서 나타나는 주요 만족/불만족 요인을 탐색하는 텍스트 마이닝 프로젝트이다.

---

## 🕰️ 개발 기간
2024.11.23- 2025.02.04

---

## 🧑‍🤝‍🧑 멤버 구성
- 개인 프로젝트

---

## 🧩 연구 / 시스템 구조 (Workflow)
### 01. 웹 데이터 수집 (HTML Scraping)
- Selenium 기반으로 웹 페이지 스크롤 로딩 처리
- BeautifulSoup로 HTML 파싱 후 리뷰 텍스트 추출
- 수집 결과를 DataFrame으로 저장(CSV)

### 02. 텍스트 전처리 및 정제
- 결측/중복 처리 및 텍스트 정규화
- 특수문자/이모지 제거
- KoNLPy(Okt) 기반 형태소 분석 및 토큰화
- 불용어 제거 및 분석용 토큰 데이터 생성

### 03. 키워드 분석 (TF-IDF)
- 결측값 제거 후 데이터 정리
- TF-IDF 기반 키워드 중요도 산출
- 상위 키워드 도출로 핵심 언급 요소 탐색

### 04. 토픽 모델링 (LDA)
- 전처리된 토큰 데이터 기반 LDA 학습
- 토픽별 핵심 단어 확인
- 토픽 응집도(Coherence) 평가를 통해 토픽 품질 확인

---

## ⚙️ 개발 환경
- **Language**: Python
- **Scraping**: Selenium, BeautifulSoup4
- **Preprocessing/NLP**: pandas, numpy, re, KoNLPy(Okt)
- **Keyword Analysis**: scikit-learn (TF-IDF)
- **Topic Modeling**: gensim (LDA, CoherenceModel)
- **Visualization**: matplotlib

---

## 📌 주요 기능
### 01 웹페이지 HTML 수집 및 리뷰 추출
- Selenium 기반 페이지 스크롤 처리
- BeautifulSoup HTML 파싱으로 리뷰 텍스트 수집
- CSV 저장

: `./리들샷 300.ipynb`

---

### 02 데이터 전처리 및 텍스트 정제  
- 결측/중복 제거
- 텍스트 정규화
- 형태소 분석 기반 토큰화(Okt)
- 불용어 제거 및 Token 데이터 생성

:`./전처리 12.21 .ipynb`

---

### 03 TF-IDF 기반 키워드 분석 
- 리뷰 텍스트 결측값 제거 및 정리
- TF-IDF로 핵심 키워드 중요도 산출
- 상위 키워드 추출

: `./tf-idf.ipynb`

---

### 04 LDA 기반 토픽 모델링
- 토큰 기반 corpus/dictionary 생성
- LDA 토픽 모델 학습
- Coherence Score 평가로 토픽 품질 확인

: `./LDA .ipynb`







# 💻 Code (Toggle)

## 01. HTML Scraping (리들샷 300.ipynb)
<details>
<summary>💻 code | Selenium 스크롤 로딩 + BeautifulSoup 파싱 + CSV 저장</summary>
<div markdown="1">

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
import pandas as pd

driver = webdriver.Chrome()
url = "YOUR_URL_HERE"
driver.get(url)
time.sleep(3)

# ✅ 스크롤 로딩 (리뷰 더 불러오기)
last_height = driver.execute_script("return document.body.scrollHeight")
for _ in range(15):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# ✅ HTML 파싱
soup = BeautifulSoup(driver.page_source, "html.parser")

# ✅ 리뷰 텍스트 추출 (selector는 페이지 구조에 맞게 수정)
reviews = []
for r in soup.select("YOUR_REVIEW_SELECTOR"):
    text = r.get_text(strip=True)
    if text:
        reviews.append(text)

driver.quit()

# ✅ DataFrame 저장
df = pd.DataFrame({"review": reviews})
df.to_csv("raw_reviews.csv", index=False, encoding="utf-8-sig")
print(f"Saved: {len(df)} reviews")
````

</div>
</details>

---

## 02. Text Preprocessing (전처리 12.21.ipynb)

<details>
<summary>💻 code | 정규식 정제 + 결측/중복 제거 + 형태소 분석(Okt) + 불용어 처리</summary>
<div markdown="1">

```python
import pandas as pd
import re
from konlpy.tag import Okt

okt = Okt()

# ✅ 데이터 로드
df = pd.read_csv("raw_reviews.csv")

# ✅ 결측/중복 제거
df = df.dropna(subset=["review"])
df = df.drop_duplicates(subset=["review"])

# ✅ 텍스트 정규화 함수
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)                 # URL 제거
    text = re.sub(r"[^가-힣0-9a-zA-Z\s]", " ", text)            # 특수문자/이모지 제거
    text = re.sub(r"\s+", " ", text).strip()                   # 공백 정리
    return text

df["clean_review"] = df["review"].apply(clean_text)

# ✅ 불용어(예시)
stopwords = set(["진짜", "너무", "완전", "그냥", "정말", "약간", "것", "수", "때"])

# ✅ 형태소 분석 + 토큰화
def tokenize(text: str):
    tokens = okt.morphs(text, stem=True)
    tokens = [t for t in tokens if len(t) > 1 and t not in stopwords]
    return tokens

df["tokens"] = df["clean_review"].apply(tokenize)

# ✅ 전처리 결과 저장
df.to_csv("preprocessed_reviews.csv", index=False, encoding="utf-8-sig")
print("Saved: preprocessed_reviews.csv")
```

</div>
</details>

---

## 03. TF-IDF Keyword Extraction (tf-idf.ipynb)

<details>
<summary>💻 code | TF-IDF 학습 + 상위 키워드 추출</summary>
<div markdown="1">
<img width="641" height="411" alt="image" src="https://github.com/user-attachments/assets/8a3c5dc5-0c76-4298-a732-4f24d2f79f26" />

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("preprocessed_reviews.csv")

# ✅ TF-IDF는 문자열 입력 필요 → 토큰 join
# tokens 컬럼이 문자열(list 형태)로 저장된 경우 eval로 변환

def safe_join(x):
    if pd.isna(x):
        return ""
    if isinstance(x, str):
        try:
            return " ".join(eval(x))
        except:
            return x
    return " ".join(x)

df["text_for_tfidf"] = df["tokens"].apply(safe_join)

vectorizer = TfidfVectorizer(max_features=2000)
tfidf = vectorizer.fit_transform(df["text_for_tfidf"])

# ✅ 전체 리뷰 기준 상위 키워드
scores = tfidf.sum(axis=0).A1
keywords = vectorizer.get_feature_names_out()

top_n = 20
top_idx = scores.argsort()[::-1][:top_n]
top_keywords = [(keywords[i], round(scores[i], 3)) for i in top_idx]

print("Top TF-IDF Keywords")
for k, s in top_keywords:
    print(k, s)
```

</div>
</details>

---

## 04. LDA Topic Modeling (LDA.ipynb)

<details>
<summary>💻 code | Dictionary/Corpus 생성 → LDA 학습 → Coherence Score 평가</summary>
<div markdown="1">


<img width="499" height="162" alt="image" src="https://github.com/user-attachments/assets/54c6b795-e92a-4218-b6d2-db62beeef8b2" />

<img width="1500" height="219" alt="image" src="https://github.com/user-attachments/assets/5df73bd5-ef53-4b13-a7e5-89f3f618d1d5" />

```python
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel

# ✅ 데이터 로드
df = pd.read_csv("preprocessed_reviews.csv")

# ✅ tokens 컬럼 복원

def parse_tokens(x):
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            return eval(x)
        except:
            return x.split()
    return x

tokens_list = df["tokens"].apply(parse_tokens).tolist()

# ✅ Dictionary / Corpus
dictionary = Dictionary(tokens_list)
corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]

# ✅ LDA 학습
num_topics = 5
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=42,
    passes=10
)

# ✅ 토픽 출력
for i, topic in lda_model.print_topics(num_words=8):
    print(f"Topic {i}: {topic}")

# ✅ Coherence 평가
coherence_model = CoherenceModel(
    model=lda_model,
    texts=tokens_list,
    dictionary=dictionary,
    coherence="c_v"
)
coherence_score = coherence_model.get_coherence()
print("Coherence Score:", round(coherence_score, 4))
```

</div>
</details>
```



