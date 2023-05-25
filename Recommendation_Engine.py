import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_option_menu import option_menu
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


#Set Page Layout
st.set_page_config(layout="wide")

# title
st.markdown("<h1 style='text-align: center; font-family: monospace; color: #f5deb3;'>Book Recommendation Engine</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>_____________________________________________________________________</h3>", unsafe_allow_html=True)


with st.sidebar:
    selected = option_menu(None, ["Trending Books", "Books For You","Browse Books"],
                           styles={"nav-link-selected": {"background-color": "#262730"}},
                            menu_icon="cast", default_index=0)


users = pd.read_csv("Users.csv")
rating = pd.read_csv("Ratings.csv")
books = pd.read_csv("Books_detail.csv",low_memory=False)
bookurl = pd.read_csv("BookURL.csv")
books = pd.concat([books,bookurl],axis=1)


book_data_1 = pd.read_csv("Book_Data_1.csv")
book_data_2 = pd.read_csv("Book_Data_2.csv")
book_data = pd.concat([book_data_1,book_data_2],axis=1)

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

books_rating = pd.merge(books, rating, on="ISBN", how="left")
popular = books_rating[(books_rating["Book-Rating"] != 0)]


def get_columns(var,df):
    content = []
    for i in df.index.values:
        val = books_rating.loc[(books_rating == i).any(axis=1),var].values
        val1 = val[0]
        content.append(val1)
    return content


def get_book_data(df):
    data = book_data[book_data["Book-Title"].isin(df["Book-Title"])]
    return data

def popularity(var, top:int):

    popular_df = pd.DataFrame(popular.groupby(var).agg({'Book-Rating':['count','mean']}))
    popular_df.columns = ['num_ratings','avg_rating']
    popular_df = popular_df.sort_values(by=["num_ratings","avg_rating"],ascending=False).head(top)
    popular_df = popular_df.sort_values(by="avg_rating", ascending=False).head(top)
    return popular_df


def display_books(popular_data):
    num_cols = 5
    num_books = len(popular_data)
    num_rows = (num_books + num_cols - 1) // num_cols

    for i in range(num_rows):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            book_index = i * num_cols + j
            if book_index < num_books:
                book = popular_data.iloc[book_index]
                with cols[j]:

                    st.image(book["link"])
                    st.write(book["Book-Title"])
                    st.write("Author : ", book["Book-Author"])
                    st.write("Publisher :  ", book["Publisher"])
                    st.write("Average Rating : ", book["avg_rating"])
                    st.write("As per ", book["num_ratings"],"votes")
            else:
                cols[j].empty()

        st.markdown("<h5 style='text-align: left;'>_____________________________________________________________________________________________________________________________________</h5>",
                    unsafe_allow_html=True)

def flatten(x):
    for e in x:
        if isinstance(e, list):
            yield from flatten(e)
        else:
            yield e

if selected == "Trending Books" :
    tab1, tab2, tab3 = st.tabs(["Popular Books", "Popular Authors", "Popular Publishers"])

    with tab1:
        popular_books = popularity("Book-Title", top=100).reset_index().drop(columns=["num_ratings","avg_rating"])
        popular_books_df = get_book_data(popular_books)
        popular_books_df = popular_books_df.sort_values(by="avg_rating", ascending=False)

        display_books(popular_books_df)

    with tab2:
        popular_authors = popularity("Book-Author", top=10)
        popular_authors = popular_authors.reset_index()

        for index, row in popular_authors.iterrows():
            st.subheader(row["Book-Author"])
            st.write("Average Rating : ",row["avg_rating"])
            st.write("As per ", row["num_ratings"], "votes")

            author_book = pd.DataFrame(popular.loc[popular['Book-Author'] == row["Book-Author"], ['Book-Title']])
            author_books = get_book_data(author_book)
            author_books = author_books.sort_values(by=["num_ratings","avg_rating"], ascending=False).head(20)

            with st.expander("Popular Books from "+row["Book-Author"]):
                display_books(author_books)

    with tab3:
        popular_publishers = popularity("Publisher", top=10)
        popular_publishers = popular_publishers.reset_index()

        for index, row in popular_publishers.iterrows():
            st.subheader(row["Publisher"])
            st.write("Average Rating : ", row["avg_rating"])
            st.write("As per ", row["num_ratings"], "votes")

            publisher_book = pd.DataFrame(
                popular.loc[popular['Publisher'] == row["Publisher"], ['Book-Title']])
            publisher_books = get_book_data(publisher_book)
            publisher_books = publisher_books.sort_values(by=["num_ratings","avg_rating"], ascending=False).head(20)

            with st.expander("Popular Books from " + row["Publisher"]):
                display_books(publisher_books)


if selected == "Books For You" :
    with st.sidebar:
        numb = st.text_input('Enter Your ID:')
        st.write("Suggested Users : 7286, 38023, 107951, 201290, 274301")

    if not numb:
        st.write("Please Enter a valid User-ID:")
    else:
        st.subheader("Book Recommendations For You")
        user_id = int(numb)

        if user_id not in users["User-ID"].unique():
            st.sidebar.write("User does not exsist!")
        else:
            users_rating = pd.merge(books_rating, users, on="User-ID", how="right")
            user_counts = popular["User-ID"].value_counts()
            rating_df_books = popular[popular['User-ID'].isin(user_counts[user_counts > 50].index)]
            book_counts = rating_df_books["Book-Title"].value_counts()
            rating_df_books = rating_df_books[rating_df_books['Book-Title'].isin(book_counts[book_counts >= 10].index)]

            if user_id not in rating_df_books['User-ID']:
                new = users_rating[(users_rating['User-ID'] == user_id)]
                new["Book-Rating"].fillna(0, inplace=True)
                new["Book-Title"].fillna("Unknown", inplace=True)

                df1 = pd.concat([rating_df_books, new], ignore_index=True)
            else:
                df1 = rating_df_books.copy()

            users_matrix = pd.pivot_table(data=df1, index=["User-ID"], columns=["Book-Title"], values="Book-Rating",
                                      dropna=False)
            users_matrix.fillna(0, inplace=True)
            # cosin matrix
            user_sim = 1 - pairwise_distances(users_matrix.values, metric="cosine")
            np.fill_diagonal(user_sim, 0)
            # converting the matrix into dataframe
            user_sim_df = pd.DataFrame(user_sim)
            # set the index and column names to user ids
            user_sim_df.index = df1["User-ID"].unique()
            user_sim_df.columns = df1["User-ID"].unique()
            
            similar = user_sim_df.nlargest(2, user_id).index.values

            # extracting the books which is already read and rated by the specified user

            book_read = users_rating.loc[(users_rating == user_id).any(axis=1), 'Book-Title'].values
            # extracting the books which are rated by the similar users
            book_sugg = []
            for x in np.nditer(similar):

                book = books_rating.loc[(books_rating == x).any(axis=1), 'Book-Title'].values
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
            recommend_books = get_book_data(Suggested)
            display_books(recommend_books)



book_unique = books_rating["Book-Title"].unique()
vectorizer = TfidfVectorizer()

if selected == "Browse Books":
    with st.sidebar:
        input_book = st.text_input('Enter a Book Name:')

    if not input_book:
        st.write("Please Enter a valid Book Title!")

    else:
        num_parts = 30
        books_per_part = len(book_unique) // num_parts
        recommendations = []
        scores = []
        for i in range(num_parts):
            start_index = i*books_per_part
            end_index = start_index+books_per_part
    
            if i == num_parts-1:
                dataset = book_unique[start_index:].tolist()
        
            else:
                dataset = book_unique[start_index:end_index].tolist()
                
        
            dataset.append(input_book)
            book_vectors = vectorizer.fit_transform(dataset)
            
            # Compute the cosine similarity
            similarity_scores = cosine_similarity(book_vectors)

            # Get the index
            book_index = dataset.index(input_book)

            # Get the cosine similarity scores for the book
            similar_books = list(enumerate(similarity_scores[book_index]))

            # Sort the similar books by similarity score
            similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)
            top_books = similar_books[1:100]

            top_books_index = [t[0] for t in top_books]
            top_books_scores = [t[1] for t in top_books]

            # Get the top N similar books
            top_books_name = [dataset[i] for i in top_books_index]

            recommendations.append(top_books_name)
            scores.append(top_books_scores)
           
        flat_list1 = [item for sublist in recommendations for item in sublist]
        flat_list2 = [item for sublist in scores for item in sublist]

        book_output = pd.DataFrame({'Column1': flat_list1, 'Column2': flat_list2})
        book_output.columns = ["Book-Title","Scores"]
        book_output.drop_duplicates(inplace=True)
        book_output = book_output.sort_values(by="Scores", ascending=False).head(100)
        
        if input_book in book_unique:
            book_results=get_book_data(book_output)
            book_results = pd.merge(book_output, book_results, on="Book-Title", how="inner").sort_values(by=["Scores","num_ratings","avg_rating"], ascending=False)

        else:
            book_output = book_output.drop(book_output[book_output['Book-Title'] == input_book].index)
            book_results = get_book_data(book_output)
            
          
        st.subheader("Showing Results For : " +input_book)
        display_books(book_results)
