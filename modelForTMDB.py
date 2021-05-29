import difflib
import pandas as pd
# Using CountVectorizer for movie search
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reading the dateset
df2 = pd.read_csv('./dataset/tmdb.csv', error_bad_lines=False)

# Define CountVectorizer with 'english' as stop word
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])

# Define cosine_sim2
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df2 = df2.reset_index()
# Seperating the 'title' from the dataset
indices = pd.Series(df2.index, index=df2['title'])
# Putting all the movie title found in the dataset
all_title = [df2['title'][i] for i in range(len(df2['title']))]

# Defining the function to search the movies
def getRecommendations(title):
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Getting the top 10 movie recommendation
    sim_scores = sim_scores[1:11]
    movie_idx = [i[0] for i in sim_scores]
    tit = df2['title'].iloc[movie_idx]
    dat = df2['release_date'].iloc[movie_idx]
    return_df = pd.DataFrame(columns=['Title', 'Year'])
    return_df['Title'] = tit
    return_df['Year'] = dat
    return return_df

# User enter the movie title
title = input("Enter a movie title :: ")
# Converting the movie title 
title = title.title()
# If title not found 
if title not in all_title:
    print ("Sorry No Movie Found")
# If movie title found
else:
    # Getting the result and storing the result
    result = getRecommendations(title)
    # To print in proper format
    print ("Name :: Year")
    for i in range(len(result)):
        print ("{} : {}".format(result.iloc[i][0], result.iloc[i][1]))
    print ("Happy Binge Watching")