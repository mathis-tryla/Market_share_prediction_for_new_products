# How to run this script :
# 1 : create your open ai api key on their website and put it in a file in this folder.
# 2 : download classification_model_weights.pth from Gdrive and put it in this folder. link : https://drive.google.com/file/d/1FHfWZwTjwmj6Os0vZGhiiTfq60TjkgsD/view?usp=sharing
# 2 : install all dependencies.
# 3 : python3 extract_features.py comments.txt(file) classification_model_weights.pth(file) api_key.txt(file)
# Ex : python3 extract_features.py comments.txt classification_model_weights.pth api_key.txt
import sys
import time
import string
import re
import nltk
import numpy as np
import pandas as pd
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import torch
import stanza
import openai
import yake
from fairseq.models.roberta import CamembertModel
from sklearn import cluster
from sklearn import metrics
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt



# Initialize all tools
def init_all():
    sw = init_nltk_stop_words()
    c = init_camembert_for_embeding()
    return sw, c


# Import webscrapped reviews
def import_reviews_and_tokenize_to_sentence(comments_file):
    nltk.download('punkt')
    french_tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')
    # Load file
    file = open(comments_file, encoding="utf8")
    # Tokenize file
    tokens = french_tokenizer.tokenize(file.read())
    file.close()

    # Cut reviews into sentences
    sentences = []
    for sentence_not_splited in tokens:
        for sentence in sentence_not_splited.split("\n"):
            sentences.append(sentence)

    return sentences


# Classify the reviews in 4 categories (1=NoMeaningfullCritic, 2=NegativeCriticOfFeature, 3=PositiveCriticOfFeature, 4=AmeliorationDemand)
def classify_reviews(reviews, classification_model_weights):
    max_len = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer and model
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)
    model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=4)
    model.load_state_dict(torch.load(classification_model_weights))

    # Encode the reviews
    tokenized_reviews_ids = [tokenizer.encode(review, add_special_tokens=True, max_length=max_len) for review in
                             reviews]
    # Pad the resulted encoded reviews
    tokenized_reviews_ids = pad_sequences(tokenized_reviews_ids, maxlen=max_len, dtype="long", truncating="post",
                                          padding="post")

    # Create attention masks
    attention_masks = []
    for seq in tokenized_reviews_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    prediction_inputs = torch.tensor(tokenized_reviews_ids)
    prediction_masks = torch.tensor(attention_masks)

    # Apply the finetuned model (Camembert)
    flat_pred = []
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(prediction_inputs.to(device), token_type_ids=None, attention_mask=prediction_masks.to(device))
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        flat_pred.extend(np.argmax(logits, axis=1).flatten())

    return flat_pred


# Append reviews commenting a feature or characteristic to a dataframe
def usefull_reviews_to_df(reviews, preds):
    temporary_list = []
    for i in range(len(preds)):
        if preds[i] != 0 :
            temporary_list.append([reviews[i],preds[i]])
    return pd.DataFrame(temporary_list, columns=['review', 'category'])


# Remove non meaningfull words from sentences and lemmatize them
def postaging_and_lemmatization(df):
    stanza.download('fr')  # download French model
    nlp = stanza.Pipeline('fr')  # initialize French neural pipeline

    simplified_sentences = []
    simplified_lemmatized_sentences = []
    for sentence in df['review']:
        doc = nlp(sentence)
        sentence_to_add = ''
        lemmatized_sentence_to_add = ''
        for token in doc.iter_words():
            if token.upos in ['NOUN', 'VERB', 'CCOJN', 'ADV', 'AUX', 'PROPN', 'ADJ', 'ADP']:
                # print(token.text, token.lemma, token.upos)
                sentence_to_add = sentence_to_add + token.text + ' '
                lemmatized_sentence_to_add = lemmatized_sentence_to_add + token.lemma + ' '
        simplified_sentences.append(sentence_to_add)
        simplified_lemmatized_sentences.append(lemmatized_sentence_to_add)

    df['simplified_sentences'] = simplified_sentences
    df['simplified_lemmatized_sentences'] = simplified_lemmatized_sentences

    return df


# Initialize tools for stop words trimming
def init_nltk_stop_words():
    nltk.download('stopwords')
    sws = stopwords.words('french')
    for sw in sws:
        if sw in ['mais', 'pas', 'ne', 'n']:
            sws.remove(sw)

    return sws


# Remove stop words from text
def __remove_stop_words__(text):
    try:
        # Remove punctuation, unnecessary whitespaces, and convert to lower case
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        text = re.sub(r'\s+', ' ', text.strip()
                      .lower()
                      .translate(str.maketrans(string.punctuation,
                                               ' ' * len(string.punctuation))))

        # Remove stop words and unnecessary words
        words = text.split(' ')
        text = ' '.join(w for w in words if (not w in stop_words) and (len(w) > 0))

    except Exception as e:
        print(f"Error: {e}")

    return text


# Remove stop words from simplified_lemmatized_sentences
def remove_stop_words(df):
    df["processed_sentences"] = df["simplified_lemmatized_sentences"].apply(lambda x: __remove_stop_words__(x))
    return df


# Initialize tools for embeding
def init_camembert_for_embeding():
    return CamembertModel.from_pretrained('camembert-base')


# Embed the reviews
def camembert_embed(text):
    tokens = camembert.encode(text)
    with torch.no_grad():
        encoded_layers = camembert.extract_features(tokens, return_all_hiddens=True)
    token_embeddings = torch.stack(encoded_layers, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    sum_vec = torch.sum(token_embeddings[-4:], dim=0)
    sentence_embedding = torch.mean(sum_vec, dim=0)
    array = sentence_embedding.detach().numpy()
    return array


# Embed the reviews
def apply_camembert_embed(df):
    df['tokens'] = df['processed_sentences'].apply(lambda x: camembert_embed(x))
    return df


# Clusterize the reviews with kmeans
def cluster_reviews(df, show = False):
    # Change the number of clusters here
    num_clusters = 12

    X = df['tokens'].tolist()
    kmeans = cluster.KMeans(n_clusters=num_clusters)
    kmeans.fit(X)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    if show:
        print("Cluster id labels for input data")
        print(labels)
        print("Centroids data")
        print(centroids)

        print(
            "Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
        print(kmeans.score(X))

        silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

        print("Silhouette_score: ")
        print(silhouette_score)

        model_tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)

        Y = model_tsne.fit_transform(X)

        plt.scatter(Y[:, 0], Y[:, 1], c=labels, s=290, alpha=.5)

        plt.show()

    df['cluster_id'] = df['index'].apply(lambda x: labels[x])
    df['cluster_size'] = df.groupby('cluster_id')['cluster_id'].transform('count')
    df_sorted = df.sort_values(by=['cluster_size', 'cluster_id'], ascending=False)

    return df_sorted


# return review clusters
def get_clusters(df):
    clusters = {}

    for index, row in df.iterrows():
        # print(f"{row['title']} - {row['body']}\n")
        cluster_id = row['cluster_id']
        # print(cluster_id)

        item = {'processed_sentences': row['processed_sentences']}

        try:
            clusters[cluster_id].append(item)
        except:
            clusters[cluster_id] = [item]

    return clusters


# Initialize keyword extracion tools
def init_keywords_extraction():
    # Setup de l'extracteur de mots clés
    language = "fr"  # langue
    max_ngram_size = 2 # nombre de ngrams
    deduplication_threshold = 0.2 # taux de duplicats acceptés (0.1 = aucun | 0.9 = tous)
    num_of_keywords = 20
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=num_of_keywords, features=None)
    return custom_kw_extractor


# Extract meaningfull keywords from a cluster
def extract_keywords_of_cluster(a, custom_kw_extractor):
  kw = custom_kw_extractor.extract_keywords(a)
  return kw


# Extract meaningfull keywords from clusters
def extract_keywords(clusters):
    custom_kw_extractor = init_keywords_extraction()
    ngrams_df = pd.DataFrame(data={'cluster': [], 'ngram': [], 'score': []})
    for key in clusters.keys():
        print(f"\n##{key}##")
        cluster_sentences = ''
        for item in clusters[key]:
            cluster_sentences = cluster_sentences + item['processed_sentences'] + '. '
        keywords = custom_kw_extractor.extract_keywords(cluster_sentences)
        for kw in keywords:
            print(kw)
            ngrams_df = ngrams_df.append({'cluster': key, 'ngram': kw[0], 'score': kw[1]}, ignore_index=True)
    return ngrams_df


# initialize request to GPT api
def init_request_text_gpt():
    product_category = "shampooing"
    prompt = "Écris une très courte demande d'amélioration d'un produit. Cette demande ne doit faire qu'une seule phrase. Cette phrase doit être en français. Ce produit est un "
    prompt = prompt + product_category
    prompt = prompt + ". Utilise les mots clés suivants : "

    return prompt


# Request GPT3 to write a generic feature or characteristic request from clusters keywords
# See https://platform.openai.com/docs/models/gpt-3 for all the models available
def request_gpt(df, ngrams_df, api_file, gpt_model = "text-davinci-003"):
    api_key_file = open(api_file, encoding="utf8")
    openai.api_key = api_key_file.read()
    api_key_file.close()
    temperature = 0.3
    request_text = init_request_text_gpt()

    reponses = []
    # For all clusters
    # for cluster_index in df.sort_values(by=['cluster_id'],ascending=True)['cluster_id'].unique():
    # For test purpose
    for cluster_index in df.sort_values(by=['cluster_id'], ascending=True)['cluster_id'].unique()[:3]:
        kw_gpt = ''
        df_cluster = df[df['cluster_id'] == cluster_index]
        print(f"\n\n\t\t##CLUSTER {cluster_index}##")
        ngrams_df_cluster = ngrams_df[ngrams_df['cluster'] == cluster_index]
        for ngram in ngrams_df_cluster.iterrows():
            kw_gpt = kw_gpt + f" \"{ngram[1].ngram}\""
        print(request_text + kw_gpt)
        # Envoi de la requête
        response = openai.Completion.create(
            model=gpt_model,
            prompt=request_text + kw_gpt,
            temperature=temperature,
            max_tokens=75,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.6
        )
        reponses.append(response)
        print(response["choices"][0]["text"])


def main():
    if len(sys.argv) == 4:
        comments_file = sys.argv[1]
        classification_model_weights = sys.argv[2]
        api_key_file = sys.argv[3]

        # Load the webscrapped reviews
        reviews = import_reviews_and_tokenize_to_sentence(comments_file)

        # Sproofreading of the reviews

        # Classify the reviews
        preds = classify_reviews(reviews, classification_model_weights)
        df_reviews = usefull_reviews_to_df(reviews, preds)

        # POS TAG and lemmatization
        df_reviews = postaging_and_lemmatization(df_reviews)
        df_reviews = remove_stop_words(df_reviews)

        # Remove stop words
        df_reviews = remove_stop_words(df_reviews)

        # Embeding
        apply_camembert_embed(df_reviews)
        # Clustering
        df_sorted = cluster_reviews(df_reviews)
        clusters = get_clusters(df_sorted)
        # Keyword extraction
        ngrams_df = extract_keywords(clusters)

        # Request GPT
        request_gpt(df_sorted, ngrams_df, api_key_file)

        return "done"

    else:
        return "wrong number of arguments"


start_time = time.time()
print(f"-- Starting the extraction of features")
stop_words, camembert = init_all()
output = main()
end_time = time.time()
final_time = end_time - start_time
print(f"-- Extraction of features DONE")
print(f"-- Output = {output} DONE")
print(f"-- {final_time} seconds--")
