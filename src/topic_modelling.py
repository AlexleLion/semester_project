import os
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import pyLDAvis.gensim
import pickle 
import pyLDAvis
import nltk
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('french') + [
    "alors", "au", "aucuns", "aussi", "autre", "avant", "avec", "avoir", "bon", "car", "ce", "cela", "ces", "ceux", 
    "chaque", "ci", "comme", "comment", "dans", "des", "du", "dedans", "dehors", "depuis", "devrait", "doit", "donc", 
    "dos", "début", "elle", "elles", "en", "encore", "essai", "est", "et", "eu", "fait", "faites", "fois", "font", 
    "hors", "ici", "il", "ils", "je", "juste", "la", "le", "les", "leur", "là", "ma", "maintenant", "mais", "mes", 
    "mine", "moins", "mon", "mot", "même", "ni", "nommés", "notre", "nous", "nouveaux", "ou", "où", "par", "parce", 
    "pas", "peut", "peu", "plupart", "pour", "pourquoi", "quand", "que", "quel", "quelle", "quelles", "quels", "qui", 
    "sa", "sans", "ses", "seulement", "si", "sien", "son", "sont", "sous", "soyez", "sujet", "sur", "ta", "tandis", 
    "tellement", "tels", "tes", "ton", "tous", "tout", "trop", "très", "tu", "valeur", "voie", "voient", "vont", 
    "votre", "vous", "vu", "ça", "étaient", "état", "étions", "été", "être", "de", "a", "4", "-", "une", "plus","7", 
    "un", "se", "'", "_",'‘', 'ne', "cette", "bien", "toujours", "si", "aussi", "peu", "deux", "trois", "quatre", 
    "cinq", "nature", "faire", "faut", "peut", "doit", "doivent", "peuvent", "chez", "méme", "soit", "dont", "non", 
    "tous", "toutes", "toute", "c’est", "dune", "d’une", "tant", "ainsi", "cest", "surtout", "étre", "pourtant", 
    "souvent", "trés", "leurs", "quelques", " ", "pendant", "aprés", "autres", "ment", "celle", "beaucoup", "ee", "etre","meme",
    "gr", "oe", "eee", "be", "oo", "ii", "ct", "we", "ie", "at", "qf", "pe", "he", "an", "eae", "ey", "fe", "eo", "ae", "re"
    "er", "ea", "ge", "see", "ay", "ees", "pee", "so", "oa", "of","ar", "ete", "ja", "ye", "pa" "ve", "aes", "ree", "cest"])


data_dir = {
    "masculin": "../data/txt/txt_merged/masculins",
    "feminin": "../data/txt/txt_merged/feminins",
    "mixte" : "../data/txt/txt_merged/mixtes"
}

def load_data(data_dir):
    texts, labels = [], []
    for label, path in data_dir.items():
        for file_path in glob.glob(os.path.join(path, "*.txt")):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels

def preprocess_text(text):
    # Nettoyage de base
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def first_lda(texts, labels):
    processed_texts = [preprocess_text(text) for text in texts]

    # Convertir l'ensemble des stopwords en liste
    stop_words_list = list(stop_words)

    # Utilisation dans CountVectorizer
    vectorizer = CountVectorizer( input = 'content', max_df=0.95, min_df=2, stop_words=stop_words_list)
    X = vectorizer.fit_transform(processed_texts)
    
    lda = LatentDirichletAllocation(n_components=10, random_state=42)
    lda.fit(X)
    
    doc_topic_distribution = lda.transform(X)
    df_results = pd.DataFrame(doc_topic_distribution, columns=[f"Topic {i+1}" for i in range(lda.n_components)])
    df_results['Genre'] = labels

    # Moyenne des proportions de chaque topic par genre
    mean_topics_by_genre = df_results.groupby('Genre').mean()
    return mean_topics_by_genre

texts, labels = load_data(data_dir)

first_lda(texts, labels)



def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

def complex_lda(texts):
    data_words = list(sent_to_words(texts))
    # Enlève les stop words
    data_words = remove_stopwords(data_words)
    
    id2word = corpora.Dictionary(data_words)
    texts = data_words

    # Fréquence des mots
    corpus = [id2word.doc2bow(text) for text in texts]
    num_topics = 5
    
    # LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics)
    pyLDAvis.enable_notebook()
    LDAvis_data_filepath = os.path.join('./results/lda'+str(num_topics))
    os.makedirs(os.path.dirname(LDAvis_data_filepath), exist_ok=True)
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)

    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, './results/lda'+ str(num_topics) +'.html')


texts, _ = load_data(data_dir)
complex_lda(texts)
