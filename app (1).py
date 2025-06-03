import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import pyLDAvis.gensim_models
import streamlit.components.v1 as components

st.title("📊 Review Dashboard")

uploaded_file = st.file_uploader("Upload a CSV file with reviews", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview:", df.head())

    # Word count
    df['word_count'] = df['review'].astype(str).apply(lambda x: len(x.split()))
    st.subheader("📝 Review Length Distribution")
    fig, ax = plt.subplots()
    df['word_count'].hist(ax=ax, bins=20)
    st.pyplot(fig)

    # Wordcloud
    st.subheader("☁️ WordCloud")
    text = " ".join(df['review'].astype(str))
    wordcloud = WordCloud(width=800, height=400).generate(text)
    fig_wc, ax_wc = plt.subplots()
    ax_wc.imshow(wordcloud)
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    # Topic Modeling
    st.subheader("🧠 Topic Modeling")
    texts = df['review'].astype(str).tolist()
    processed = [[w for w in simple_preprocess(t) if w not in STOPWORDS] for t in texts]
    dictionary = Dictionary(processed)
    corpus = [dictionary.doc2bow(text) for text in processed]
    lda_model = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

    for idx, topic in lda_model.print_topics():
        st.write(f"**Topic {idx}**: {topic}")

    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    html_string = pyLDAvis.prepared_data_to_html(vis)
    components.html(html_string, height=800, scrolling=True)
