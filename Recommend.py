import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

#Set Page Layout
st.set_page_config(layout="wide")

# title
st.markdown("<h1 style='text-align: center; font-family: monospace; color: #f5deb3;'>Book Recommendation Engine</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>_____________________________________________________________________</h3>", unsafe_allow_html=True)

with st.sidebar:
    ID = st.text_input('Enter Your ID:')
    st.write("Suggested Users : 7286, 38023, 107951, 201290, 274301")

books = pd.read_csv(r"D:\Study\Recommendations\Books.csv")
users = pd.read_csv(r"D:\Study\Recommendations\Users.csv")
rating = pd.read_csv(r"D:\Study\Recommendations\Ratings.csv")

#Data Cleaning
temp = books[(books['Year-Of-Publication'] == 'DK Publishing Inc') | (books['Year-Of-Publication'] == 'Gallimard')]
authors = []
books_titles = []

for title in temp['Book-Title']:
    author = title.split(';')[-1].split('"')[0]
    book = title.split(';')[0].split('\\')[0]

    authors.append(author)
    books_titles.append(book)

shift = temp.iloc[:, 2:].shift(periods=1, axis=1)
temp = temp.iloc[:, :1]
df = pd.concat([temp, shift], axis=1)
df = df.drop(columns="Book-Author")
df.insert(loc=1, column='Book-Title', value=books_titles)
df.insert(loc=2, column='Book-Author', value=authors)

books.drop(df.index, axis=0, inplace=True)
for i in df.index:
    books.loc[i] = list(df.loc[i].values)

books['Year-Of-Publication'] = books['Year-Of-Publication'].astype('int32')

book_rating = pd.merge(rating, books, how="inner", on="ISBN")
book_rating.drop(book_rating[book_rating['Book-Rating'] == 0].index, inplace = True)

def flatten(x):
    for e in x:
        if isinstance(e, list):
            yield from flatten(e)
        else:
            yield e

def get_columns(var,df):
    content = []
    for i in df.index.values:
        val = book_rating.loc[(book_rating == i).any(axis=1),var].values
        val1 = val[0]
        content.append(val1)
    return content

def popularity(var):
    # Group the books by variable, and calculate the number of ratings and average rating
    book_stats = book_rating.groupby(var).agg({'Book-Rating':['count','mean']})
    # Rename the columns
    book_stats.columns = ['num_ratings','avg_rating']
    # fetch the top 50
    book_stats = book_stats.sort_values(['num_ratings','avg_rating'], ascending=False).head(50)
    # sorting
    popular = book_stats.sort_values("avg_rating",ascending=False)
    return popular



if not ID:
    tab1, tab2, tab3 = st.tabs(["Popular Books", "Popular Authors", "Popular Publishers"])

    with tab1:
        popular_books = popularity("Book-Title")
        link = get_columns("Image-URL-M", popular_books)
        author = get_columns("Book-Author", popular_books)
        publisher = get_columns("Publisher", popular_books)
        popular_books["link"] = link
        popular_books["Book-Author"] = author
        popular_books["Publisher"] = publisher
        popular_books = popular_books.reset_index()
        popular_books['link'] = popular_books['link'].astype(str)
        popular_books['link'] = popular_books['link'].str.replace("'",'').str.replace(']', '').str.replace('[', '')

        for index, row in popular_books.iterrows():
            col1, col2, col3, col4 = st.columns([5,2,2,1])

            with col1:
                st.subheader(row["Book-Title"])

            with col4:
                st.image(row["link"])

            with col2:
                st.write("Author : ", row["Book-Author"])
                st.write("Publisher :  ", row["Publisher"])

            with col3:
                st.write("Average Rating : ",row["avg_rating"])
                st.write("As per ", row["num_ratings"],"votes")

            st.markdown(
                "<h5 style='text-align: left;'>_________________________________________________________________________________________________________________________________________________</h5>",
                unsafe_allow_html=True)


    with tab2:
        popular_authors = popularity("Book-Author")
        popular_authors = popular_authors.reset_index()
        #st.dataframe(popular_authors)
        for index, row in popular_authors.iterrows():
            st.subheader(row["Book-Author"])
            st.write("Average Rating : ",row["avg_rating"])
            st.write("As per ", row["num_ratings"], "votes")

            author_book = pd.DataFrame(book_rating.loc[book_rating['Book-Author'] == row["Book-Author"], ['Book-Title','Book-Rating']].groupby('Book-Title')["Book-Rating"].mean()).reset_index()
            with st.expander("Books from "+row["Book-Author"]):
                st.dataframe(author_book)

            st.markdown(
                "<h5 style='text-align: left;'>__________________________________________________________________________________________________________________________________________________</h5>",
                unsafe_allow_html=True)


    with tab3:
        popular_publishers = popularity("Publisher")
        popular_publishers = popular_publishers.reset_index()
        #st.dataframe(popular_authors)
        for index, row in popular_publishers.iterrows():
            st.subheader(row["Publisher"])
            st.write("Average Rating : ",row["avg_rating"])
            st.write("As per ", row["num_ratings"], "votes")

            publisher_book = pd.DataFrame(book_rating.loc[book_rating['Publisher'] == row["Publisher"], ['Book-Title','Book-Rating']].groupby('Book-Title')["Book-Rating"].mean()).reset_index()
            with st.expander("Books from "+row["Publisher"]):
                st.dataframe(publisher_book)

            st.markdown(
                "<h5 style='text-align: left;'>__________________________________________________________________________________________________________________________________________________</h5>",
                unsafe_allow_html=True)


else:
    ID = int(ID)
    if ID not in users["User-ID"].unique():
        st.sidebar.write("User does not exsist!")
    else:
        st.markdown(
            "<h3 style='text-align: center; font-family: monospace; color: white;'>Book Recommendations for You</h3>",
            unsafe_allow_html=True)
        user_data = users[(users["User-ID"] == ID)]
        st.sidebar.subheader("Your Profile Details:")
        for index, row in user_data.iterrows():
            st.sidebar.write("Your ID : ", row["User-ID"])
            st.sidebar.write("Location : ", row["Location"])
            st.sidebar.write("Age : ", row["Age"])
        user_rating = pd.merge(rating, users, how="right", on="User-ID")
        # user_rating = user_rating[user_rating['ISBN'].isin(books['ISBN'])]
        user_rating = pd.merge(user_rating, books, how="left", on="ISBN")
        df = user_rating[user_rating['Book-Rating'] != 0]
        counts = df["User-ID"].value_counts()
        df = df[df['User-ID'].isin(counts[counts > 100].index)]

        if ID not in df['User-ID']:
            new = user_rating[(user_rating['User-ID'] == ID)]
            new["Book-Rating"].replace("NaN", pd.NA, inplace=True)
            new["Book-Title"].fillna("Unknown", inplace=True)
            df1 = pd.concat([df, new])
        else:
            df1 = df
        users_matrix = df1.pivot_table(index=["User-ID"], columns=["Book-Title"], values="Book-Rating", dropna=False)
        # imputing the missing values with 0
        users_matrix.fillna(0, inplace=True)
        # cosin matrix
        user_sim = 1 - pairwise_distances(users_matrix.values, metric="cosine")
        np.fill_diagonal(user_sim, 0)
        # converting the matrix into dataframe
        user_sim_df = pd.DataFrame(user_sim)
        # set the index and column names to user ids
        user_sim_df.index = df1["User-ID"].unique()
        user_sim_df.columns = df1["User-ID"].unique()
        # d = int(ID)
        similar = user_sim_df.nlargest(2, ID).index.values
        # extracting the books which is already read and rated by the specified user

        book_read = user_rating.loc[(user_rating == ID).any(axis=1), 'Book-Title'].values
        # extracting the books which are rated by the similar users
        book_sugg = []
        for x in np.nditer(similar):
            book = df1.loc[(df1 == x).any(axis=1), 'Book-Title'].values
            if len(book) > 25:
                book1 = book[:25]
            else:
                book1 = book

            book_sugg.append(book1)

        # we will only recommend the books which are not read by the specified user yet
        this = []
        for bi in flatten(book_sugg):
            if bi not in book_read:
                read = bi
                this.append(read)

        Suggested = pd.DataFrame(np.concatenate(this, axis=0), columns=["Book-Title"])
        # isbn = []
        # for i in df["Book-Title"].values:
        #     val = .loc[(book_rating == i).any(axis=1), var].values
        #     val1 = val[0]
        #     content.append(val1)
        # #Suggested.set_index("Book-Title",inplace=True)
        # isbn = get_columns("ISBN",Suggested)
        # Suggested["ISBN"] = isbn
        suggested_df = pd.merge(Suggested, book_rating, how='inner', on="Book-Title")
        suggestions = suggested_df.groupby("Book-Title").agg({'Book-Rating': ['count', 'mean']})
        suggestions.columns = ['num_ratings', 'avg_rating']

        link = get_columns("Image-URL-M", suggestions)
        author = get_columns("Book-Author", suggestions)
        publisher = get_columns("Publisher", suggestions)
        suggestions["link"] = link
        suggestions["Book-Author"] = author
        suggestions["Publisher"] = publisher
        suggestions.reset_index(inplace=True)
        suggestions.sort_values(by="avg_rating",ascending=False,inplace=True)

        for index, row in suggestions.iterrows():
            col1, col2, col3, col4 = st.columns([5, 2, 2, 1])

            with col1:
                st.subheader(row["Book-Title"])

            with col4:
                st.image(row["link"])

            with col2:
                st.write("Author : ", row["Book-Author"])
                st.write("Publisher :  ", row["Publisher"])

            with col3:
                st.write("Average Rating : ", row["avg_rating"])
                st.write("As per ", row["num_ratings"], "votes")

            st.markdown(
                "<h5 style='text-align: left;'>_________________________________________________________________________________________________________________________________________________</h5>",
                unsafe_allow_html=True)







