# cosmetics-review-text-mining
웹 리뷰 기반 화장품 만족·불만족 요인 탐색 

---

## 🖥️ 프로젝트 소개
본 프로젝트는 웹 페이지(화장품 리뷰)의 HTML 데이터를 수집한 뒤,  
텍스트 전처리 과정을 거쳐 키워드 분석(TF-IDF)과 토픽 모델링(LDA)을 수행하여  
리뷰에서 나타나는 주요 만족/불만족 요인을 탐색하는 텍스트 마이닝 프로젝트입니다.

특히, 실무에서 요구되는 **데이터 정제 및 전처리 역량**을 중심으로  
웹 데이터 수집 → 정제 → 분석까지 End-to-End 파이프라인 구현에 초점을 두었습니다.

---

## 🕰️ 개발 기간
- 2025.XX.XX - 2025.XX.XX

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
[상세보기 · WIKI]()  
- Selenium 기반 페이지 스크롤 처리
- BeautifulSoup HTML 파싱으로 리뷰 텍스트 수집
- CSV 저장

: `./리들샷 300.ipynb`

---

### 02 데이터 전처리 및 텍스트 정제
[상세보기 · WIKI]()   
- 결측/중복 제거
- 텍스트 정규화
- 형태소 분석 기반 토큰화(Okt)
- 불용어 제거 및 Token 데이터 생성

:`./전처리 12.21 .ipynb`

---

### 03 TF-IDF 기반 키워드 분석
[상세보기 · WIKI]()  
- 리뷰 텍스트 결측값 제거 및 정리
- TF-IDF로 핵심 키워드 중요도 산출
- 상위 키워드 추출

: `./tf-idf.ipynb`

---

### 04 LDA 기반 토픽 모델링
[상세보기 · WIKI]()  
- 토큰 기반 corpus/dictionary 생성
- LDA 토픽 모델 학습
- Coherence Score 평가로 토픽 품질 확인

: `./LDA .ipynb`
