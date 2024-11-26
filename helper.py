from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import re
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
from model import load_sematic_analysis_pipeline
from model import load_sentiment_analysis_pipeline
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st


extract = URLExtract()

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = df['message'].str.split().apply(len).sum()
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
    num_links = df['message'].apply(lambda x: len(extract.find_urls(x))).sum()

    return num_messages, words, num_media_messages, num_links


def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x, df


def create_wordcloud(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc


def most_common_words(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    stop_words = ["the", "a", "an", "in", "is", "you", "I", "and", "it", "to", "for", "of", "that", "on", "my", "at", "with", "by"]

    words = []
    for message in df['message']:
        words.extend(re.findall(r'\b\w+\b', message.lower()))

    filtered_words = [word for word in words if word not in stop_words]

    most_common_df = pd.DataFrame(Counter(filtered_words).most_common(20), columns=['Word', 'Count'])

    return most_common_df




def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend(re.findall(r'\s(👍|❤️|😂|😊|🙏|👏|🔥|🙌|😍|😎|🤣|✌️|👀|👉|🙄|😅|🤔|😁|🤗|😇)', message))

    emoji_count = Counter(emojis).most_common()
    emoji_df = pd.DataFrame(emoji_count, columns=['Emoji', 'Count'])

    return emoji_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

def map_sentiment_label_stars(label):
    if label == "5 stars":
        return 'Positive'
    elif label == "4 stars":
        return 'Positive'
    elif label == '3 stars':
        return 'Netural'
    elif label == '2 stars':
        return 'Negative'
    elif label == '1 star':
        return 'Negative'
    else:
        return "uncertain"

def plot_sematic_analysis(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    bert_model = load_sematic_analysis_pipeline()
    sentiment_model = load_sentiment_analysis_pipeline()

    with st.spinner("Computing BERT predictions..."):
        embeddings = []
        max_length = 0
        sentiments = []

  
        for message in df['message']:
            embedding = bert_model(message)[0]  
            embeddings.append(embedding)
            max_length = max(max_length, len(embedding))

            sentiment = sentiment_model(message)[0]
            sentiments.append(map_sentiment_label_stars(sentiment['label']))

        df['sentiment'] = sentiments

    padded_embeddings = np.array([np.pad(embedding, ((0, max_length - len(embedding)), (0, 0)), mode='constant') for embedding in embeddings])

    flattened_embeddings = padded_embeddings.reshape(padded_embeddings.shape[0], -1)

    with st.spinner("Computing Cluster"):
        num_clusters = 3  
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(flattened_embeddings)

    st.title('Semantic Analysis with Streamlit')

    st.subheader('Cluster Distribution:')
    cluster_counts = df['cluster'].value_counts()
    fig, ax = plt.subplots()
    cluster_counts.sort_index().plot(kind='bar', ax=ax)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    st.subheader('Scatter Plot of Embeddings with Cluster Coloring:')
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(flattened_embeddings[:, 0], flattened_embeddings[:, 1], c=df['cluster'], cmap='viridis')
    ax.set_xlabel('Embedding Dimension 1')
    ax.set_ylabel('Embedding Dimension 2')
    st.pyplot(fig)

    st.subheader('Word Clouds for Each Cluster:')
    for cluster_id in range(num_clusters):
        cluster_messages = df[df['cluster'] == cluster_id]['message']
        text = ' '.join(cluster_messages)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        st.subheader(f'Cluster {cluster_id} Word Cloud:')
        st.image(wordcloud.to_image())

    st.subheader('Histogram of Message Lengths for Each Cluster:')
    for cluster_id in range(num_clusters):
        cluster_lengths = df[df['cluster'] == cluster_id]['message'].apply(len)
        plt.hist(cluster_lengths, bins=20, label=f'Cluster {cluster_id}', alpha=0.5, density=True)

    plt.title('Histogram of Message Lengths for Each Cluster')
    plt.xlabel('Message Length')
    plt.ylabel('Density')
    plt.legend()
    st.pyplot(plt)

    st.subheader('Sentiment Distribution:')
    sentiment_counts = df['sentiment'].value_counts()
    fig, ax = plt.subplots()
    sentiment_counts.sort_index().plot(kind='bar', ax=ax)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    st.subheader('Final Results of Sentiments:')
    st.table(df[['message', 'sentiment']])

    st.subheader('Overall Sentiment:')
    overall_score = df['sentiment'].value_counts()
    st.write(f'The overall sentiment is: {overall_score.idxmax()}')

# if __name__ == "__main__":
#     main()
