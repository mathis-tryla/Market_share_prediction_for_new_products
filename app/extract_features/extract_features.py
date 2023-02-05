import string, re, nltk
import torch, stanza
import numpy as np
import pandas as pd
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from fairseq.models.roberta import CamembertModel
from sklearn import cluster, metrics
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


"""
Classify the reviews in 4 categories 
    (1=NoMeaningfullCritic, 
    2=NegativeCriticOfFeature, 
    3=PositiveCriticOfFeature, 
    4=AmeliorationDemand)
"""
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
def __remove_stop_words__(text, stop_words):
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
def remove_stop_words(df, stop_words):
    df["processed_sentences"] = df["simplified_lemmatized_sentences"].apply(lambda x: __remove_stop_words__(x, stop_words))
    return df


# Initialize tools for embeding
def init_camembert_for_embeding():
    return CamembertModel.from_pretrained('camembert-base')


# Embed the reviews
def camembert_embed(camembert, text):
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
def apply_camembert_embed(df, camembert):
    df['tokens'] = df['processed_sentences'].apply(lambda x: camembert_embed(camembert, x))
    return df


# Clusterize the reviews with kmeans
def cluster_reviews(df, nb_clusters_reviews, show = False):
    X = df['tokens'].tolist()
    kmeans = cluster.KMeans(n_clusters=nb_clusters_reviews)
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
        X = np.array(X)
        Y = model_tsne.fit_transform(X)

        plt.scatter(Y[:, 0], Y[:, 1], c=labels, s=290, alpha=.5)

        #plt.show()

    df['cluster_id'] = df['index'].apply(lambda x: labels[x])
    df['cluster_size'] = df.groupby('cluster_id')['cluster_id'].transform('count')
    df_sorted = df.sort_values(by=['cluster_size', 'cluster_id'], ascending=False)

    return df_sorted

# Get nb of reviews asking for new features in order to calculate the innovation score
def calculate_innovation_score(df):
    nb_reviews_asking_new_features = len(df[df['category'] == 4])
    nb_reviews_total = len(df)
    score = (nb_reviews_asking_new_features / nb_reviews_total)
    return round(score, 10)

def get_innovation_score(comments_file, classification_model_weights, nb_clusters_reviews):
    # Initialize camembert and get stop words
    stop_words, camembert = init_all()

    # Load the webscrapped reviews
    reviews = import_reviews_and_tokenize_to_sentence(comments_file)

    # Classify the reviews
    preds = classify_reviews(reviews, classification_model_weights)
    df_reviews = usefull_reviews_to_df(reviews, preds)

    # POS TAG and lemmatization
    df_reviews = postaging_and_lemmatization(df_reviews)
    df_reviews = remove_stop_words(df_reviews, stop_words)

    # Remove stop words
    df_reviews = remove_stop_words(df_reviews, stop_words)

    # Embedding
    df_reviews['index'] = df_reviews.index
    apply_camembert_embed(df_reviews, camembert)

    # Clustering
    df_sorted = cluster_reviews(df_reviews, nb_clusters_reviews)

    print(f"-- Calculate innovation score DONE")
    return calculate_innovation_score(df_sorted)
