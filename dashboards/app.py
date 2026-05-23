import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from collections import Counter


# page config

st.set_page_config(
    page_title="Google Photos Dashboard",
    layout="wide"
)


# graph style

sns.set_style("darkgrid")


# load data

df = pd.read_csv("clean.csv")


# sidebar filter

st.sidebar.title("Filters")

selected_sentiment = st.sidebar.selectbox(
    "Select Sentiment",
    ["All", "Positive", "Negative", "Neutral"]
)

if selected_sentiment != "All":

    df = df[
        df["Sentiment"] == selected_sentiment
    ]


# title

st.title(
    "Google Photos Review Intelligence Dashboard"
)

st.info(
    "This dashboard analyzes Google Photos reviews using NLP and sentiment analysis to identify customer satisfaction, recurring complaints, and product insights."
)


# KPI metrics

avg_polarity = df[
    "Sentiment_Polarity"
].mean()

negative_reviews = df[
    df["Sentiment"] == "Negative"
]

negative_percentage = (
    len(negative_reviews) / len(df)
) * 100


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Reviews",
        len(df)
    )

with col2:

    st.metric(
        "Average Polarity",
        round(avg_polarity, 2)
    )

with col3:

    st.metric(
        "Negative %",
        f"{negative_percentage:.2f}%"
    )


# dataset preview

st.header("Dataset Preview")

st.dataframe(df.head())


# sentiment distribution

st.header("Sentiment Distribution")

a = df["Sentiment"].value_counts()


# pie chart

fig1, ax1 = plt.subplots(figsize=(6,6))

ax1.pie(
    a,
    labels=a.index,
    autopct='%1.1f%%'
)

plt.title("Sentiment Distribution Pie Chart")

st.pyplot(fig1)


# bar chart

fig2, ax2 = plt.subplots(figsize=(6,4))

a.plot(
    kind="bar",
    ax=ax2
)

plt.title("Sentiment Distribution Bar Chart")

plt.xlabel("Sentiment")

plt.ylabel("Number of Reviews")

st.pyplot(fig2)


# auto insights

st.header("Auto Generated Insights")

top_sentiment = a.idxmax()

st.write(
    f"Most reviews are {top_sentiment}."
)

st.write(
    f"Average sentiment polarity is {avg_polarity:.2f}."
)

if avg_polarity > 0:

    st.success(
        "Users generally have a positive perception of the app."
    )

elif avg_polarity < 0:

    st.error(
        "Users generally have a negative perception of the app."
    )

else:

    st.warning(
        "Users have neutral perception of the app."
    )


# negative review analysis

st.header("Negative Review Analysis")

st.subheader("Negative Review Percentage")

st.write(f"{negative_percentage:.2f}%")

if negative_percentage > 30:

    st.error(
        "High number of negative reviews detected."
    )

else:

    st.success(
        "Negative reviews are under control."
    )


# top negative reviews

st.subheader("Top Negative Reviews")

st.dataframe(
    negative_reviews[
        [
            "Translated_Review",
            "Sentiment_Polarity"
        ]
    ].head(10)
)


# negative wordcloud

st.subheader("Negative Review WordCloud")

negative_text = " ".join(
    negative_reviews[
        "Translated_Review"
    ].dropna()
)

if len(negative_text.strip()) > 0:

    negative_wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=STOPWORDS
    ).generate(negative_text)

    fig3, ax3 = plt.subplots(figsize=(10,5))

    ax3.imshow(negative_wordcloud)

    ax3.axis("off")

    st.pyplot(fig3)

else:

    st.warning(
        "No negative reviews available for wordcloud."
    )


# positive review analysis

positive_reviews = df[
    df["Sentiment"] == "Positive"
]

st.header("Positive Review Analysis")


# top positive reviews

st.subheader("Top Positive Reviews")

st.dataframe(
    positive_reviews[
        [
            "Translated_Review",
            "Sentiment_Polarity"
        ]
    ].head(10)
)


# positive wordcloud

st.subheader("Positive Review WordCloud")

positive_text = " ".join(
    positive_reviews[
        "Translated_Review"
    ].dropna()
)

if len(positive_text.strip()) > 0:

    positive_wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=STOPWORDS
    ).generate(positive_text)

    fig4, ax4 = plt.subplots(figsize=(10,5))

    ax4.imshow(positive_wordcloud)

    ax4.axis("off")

    st.pyplot(fig4)

else:

    st.warning(
        "No positive reviews available for wordcloud."
    )


# neutral review analysis

neutral_reviews = df[
    df["Sentiment"] == "Neutral"
]

st.header("Neutral Review Analysis")


# neutral wordcloud

st.subheader("Neutral Review WordCloud")

neutral_text = " ".join(
    neutral_reviews[
        "Translated_Review"
    ].dropna()
)

if len(neutral_text.strip()) > 0:

    neutral_wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=STOPWORDS
    ).generate(neutral_text)

    fig5, ax5 = plt.subplots(figsize=(10,5))

    ax5.imshow(neutral_wordcloud)

    ax5.axis("off")

    st.pyplot(fig5)

else:

    st.warning(
        "No neutral reviews available for wordcloud."
    )


# top complaint keywords

st.header("Top Complaint Keywords")

words = negative_text.lower().split()


# custom stopwords

custom_stopwords = STOPWORDS.union(
    {
        "want",
        "even",
        "really",
        "also",
        "please",
        "google",
        "photos",
        "photo",
        "app"
    }
)

filtered_words = []

for word in words:

    if (
        word not in custom_stopwords
        and len(word) > 3
    ):

        filtered_words.append(word)


# word frequency

word_counts = Counter(filtered_words)

top_words = word_counts.most_common(5)

hash_df = pd.DataFrame(
    top_words,
    columns=["Keyword", "Frequency"]
)

st.table(hash_df)


# polarity histogram

st.header("Polarity Distribution")

fig6, ax6 = plt.subplots(figsize=(8,5))

df["Sentiment_Polarity"].hist(
    ax=ax6
)

plt.title("Polarity Distribution")

plt.xlabel("Polarity")

plt.ylabel("Frequency")

st.pyplot(fig6)


# heatmap

st.header("Sentiment Correlation Heatmap")

sentiment_map = {
    "Positive": 1,
    "Neutral": 0,
    "Negative": -1
}

df["Sentiment_Score"] = df[
    "Sentiment"
].map(sentiment_map)

numeric_df = df[
    [
        "Sentiment_Polarity",
        "Sentiment_Subjectivity",
        "Sentiment_Score"
    ]
]

corr = numeric_df.corr()

fig7, ax7 = plt.subplots(figsize=(8,5))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax7
)

plt.title("Sentiment Correlation Heatmap")

st.pyplot(fig7)


# download processed data

st.header("Download Processed Data")

csv = df.to_csv(index=False)

st.download_button(
    label="Download CSV File",
    data=csv,
    file_name="processed_reviews.csv",
    mime="text/csv"
)


# final message

st.success(
    "Dashboard Analysis Completed Successfully."
)