import flask
# from flask import *
from flask import Flask
from flask import render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
df = pd.read_csv('IMDB-Movie-Data.csv')
df['lowerTitle'] = pd.Series(str.lower(i) for i in df.Title)
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df.Description)
indices = pd.Series(df.index, index=df.lowerTitle)
all_titles = df.lowerTitle.to_list()


def get_recommendations(name):
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    idx = indices[str.lower(name)]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    title = df.Title.iloc[movie_indices]
    rating = df.Rating.iloc[movie_indices]
    recommend = pd.DataFrame(columns=['Title', 'Rating'])
    recommend['Title'] = title
    recommend['Rating'] = rating
    recommend = recommend.sort_values(by='Rating', ascending=False)
    return recommend


@app.route('/', methods=['GET', 'POST'])
def index():
    if flask.request.method == 'GET':
        return render_template('index.html')
    if flask.request.method == 'POST':
        movie_name = flask.request.form['name']
        movie_name = movie_name.lower()
        if movie_name not in all_titles:
            return render_template('negative.html', movie_name=movie_name)
        else:
            result_final = get_recommendations(movie_name)
            names = []
            rating = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                rating.append(result_final.iloc[i][1])

            return render_template('positive.html', movie_names=names, ratings=rating, search_name=movie_name)


if __name__ == '__main__':
    app.run(debug=True)
