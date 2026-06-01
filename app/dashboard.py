import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re, string
from nltk.corpus import stopwords
import nltk

nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# —- Page config ————————————————————————————————————
st.set_page_config(
    page_title='Brand Sentiment Analyzer',
    page_icon='📊',
    layout='wide'
)

# —- Load model ————————————————————————————————————
@st.cache_resource
def load_model():
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('models/sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return tfidf, model

tfidf, model = load_model()

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# —- Clean text function ——————————————————————————
# —- Clean text function ———————————————————————————
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [t for t in text.split() if t not in stop_words]
    return ' '.join(tokens)

# —- Header ————————————————————————————————————————
st.title('📊 Brand Sentiment Analyzer')
st.markdown('**Analyze public sentiment for any brand using ML**')
st.divider()

# —- Tab layout ————————————————————————————————————
tab1, tab2, tab3 = st.tabs(['🔍 Analyze Text', '📁 Analyze CSV', '📝 Brand Report'])

# —- TAB 1: Single text analysis ———————————————————
with tab1:
    st.subheader('Analyze a Single Review or Tweet')
    user_input = st.text_area('Enter text here:', height=120, placeholder='e.g. The new iPhone is absolutely amazing!')
    
    if st.button('Analyze', type='primary'):
        if user_input.strip():
            cleaned = clean_text(user_input)
            vec = tfidf.transform([cleaned])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]
            
            col1, col2, col3 = st.columns(3)
            label = 'Positive  ' if pred == 1 else 'Negative  '
            color = 'green' if pred == 1 else 'red'
            col1.metric('Sentiment', label)
            col2.metric('Confidence', f'{max(prob):.1%}')
            vader_score = sia.polarity_scores(user_input)['compound']
            col3.metric('VADER Score', f'{vader_score:.3f}')
        else:
            st.warning('Please enter some text.')

  # —- TAB 2: CSV batch analysis —————————————————————
with tab2:
    st.subheader('Upload a CSV and Analyze in Bulk')
    uploaded = st.file_uploader('Upload CSV with a text column', type='csv')
    
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.write('Preview:', df_up.head())
        col = st.selectbox('Which column has the text?', df_up.columns)
        
        if st.button('Run Batch Analysis'):
            df_up['clean'] = df_up[col].astype(str).apply(clean_text)
            vec = tfidf.transform(df_up['clean'])
            df_up['sentiment'] = model.predict(vec)
            df_up['confidence'] = model.predict_proba(vec).max(axis=1)
            df_up['label'] = df_up['sentiment'].map({1: 'Positive', 0: 'Negative'})
            
            counts = df_up['label'].value_counts()
            fig = px.pie(values=counts.values, names=counts.index,
                         color_discrete_map={'Positive': '#22C55E', 'Negative': '#EF4444'})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_up[[col, 'label', 'confidence']])
            
            csv = df_up.to_csv(index=False)
            st.download_button('Download Results', csv,
                               'sentiment_results.csv', 'text/csv')

# —- TAB 3: Pre-analyzed brand report —————————————————
with tab3:
    st.subheader('Brand Insight Reports')
    brand = st.selectbox('Choose a brand:', ['apple', 'google'])
    
    try:
        bdf = pd.read_csv(f'data/processed/{brand}_analyzed.csv')
        total = len(bdf)
        pos_pct = (bdf['predicted_sentiment'] == 1).mean() * 100
        neg_pct = 100 - pos_pct
        
        col1, col2, col3 = st.columns(3)
        col1.metric('Total Mentions', f'{total:,}')
        col2.metric('Positive', f'{pos_pct:.1f}%')
        col3.metric('Negative', f'{neg_pct:.1f}%')
        
        fig = px.histogram(bdf, x='confidence', color='predicted_sentiment',
                           color_discrete_map={1: '#22C55E', 0: '#EF4444'},
                           title='Confidence Score Distribution',
                           labels={'predicted_sentiment': 'Sentiment'})
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.info(f'Run Week 3 analysis to generate {brand} report.')
