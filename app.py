import streamlit as st

st.title("📊 Opinion Dashboard")
st.write("Upload your dataset and explore topics!")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    import pandas as pd
    from gensim.utils import simple_preprocess
    from gensim.parsing.preprocessing import STOPWORDS
    from gensim.corpora.dictionary import Dictionary
    from gensim.models import LdaModel
    import pyLDAvis.gensim_models
    import streamlit.components.v1 as components

    df = pd.read_csv(uploaded_file)
    texts = df['review'].astype(str).tolist()

    def preprocess(text):
        return [w for w in simple_preprocess(text) if w not in STOPWORDS]

    processed = [preprocess(t) for t in texts]
    dictionary = Dictionary(processed)
    corpus = [dictionary.doc2bow(text) for text in processed]
    lda_model = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

    st.write("### Topics Identified")
    for idx, topic in lda_model.print_topics():
        st.write(f"**Topic {idx}**: {topic}")

    # Visualization with pyLDAvis
    st.write("### Topic Visualization")
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    html_string = pyLDAvis.prepared_data_to_html(vis)
    components.html(html_string, height=800, scrolling=True)
