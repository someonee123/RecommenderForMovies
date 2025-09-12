from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from flask import jsonify
from tmdbv3api import TMDb, Movie
import numpy as np
import pandas as pd
import requests

app = Flask(__name__)

# Postavljanje ključa za dohvaćanje postera o filmu.
tmdb = TMDb()
tmdb.api_key = '9055ab390ff30110ab56325ec3f7cb39'
tmdb_movie = Movie()
tmdb.language = 'en'

# Učitavanje podataka iz CSV datoteka, koje su dio arhive ml-latest-small.
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")


# Funkcija koja služi za dohvaćanje postera za film.
def get_movie_poster(title):
    try:
        # Ovo je potrebno, zato što svaki film u naslovu (unutar movies.csv) sadrži i godinu izlaska.
        clean_title = title.split(' (')[0]
        results = tmdb_movie.search(clean_title)
        if not results:
            return "https://via.placeholder.com/500x750?text=Poster+Not+Available"
        movie = results[0]
        if hasattr(movie, 'poster_path') and movie.poster_path:
            return f"https://image.tmdb.org/t/p/w500{movie.poster_path}"

        return "https://via.placeholder.com/500x750?text=Poster+Not+Available"
    except:
        print("Error getting poster.")
        return "https://via.placeholder.com/500x750?text=Error+Loading+Poster"


def get_user_user_recommendations(user_id, k_neigbours=30):
    movie_counts = ratings['movieId'].value_counts()
    filtered_data = ratings[
        ratings['movieId'].isin(movie_counts[movie_counts >= 20].index)
    ]

    model_train, model_test = train_test_split(
        filtered_data, test_size=0.30, random_state=42
    )

    user_data = model_train.pivot(
        index='userId', columns='movieId', values='rating'
    ).fillna(0)
    copy_train = model_train.copy()
    copy_train['rating'] = copy_train['rating'].apply(
        lambda x: 0 if x > 0 else 1)
    copy_train = copy_train.pivot(
        index='userId', columns='movieId', values='rating'
    ).fillna(1)

    user_similarity = cosine_similarity(user_data)
    user_similarity[np.isnan(user_similarity)] = 0

    for i in range(len(user_similarity)):
        sorted_indices = np.argsort(user_similarity[i])[::-1]
        user_similarity[i, sorted_indices[k_neigbours:]] = 0

    sum_weights = np.array([np.abs(user_similarity).sum(axis=1)]).T
    sum_weights[sum_weights == 0] = 1
    user_predicted_ratings = np.dot(user_similarity, user_data) / sum_weights

    min_original = ratings['rating'].min()
    max_original = ratings['rating'].max()
    current_min = user_predicted_ratings.min()
    current_max = user_predicted_ratings.max()

    user_predicted_ratings = min_original + (user_predicted_ratings - current_min) * \
        (max_original - min_original) / (current_max - current_min)

    user_final_ratings = np.multiply(user_predicted_ratings, copy_train)
    user_final_ratings = np.round(user_final_ratings, 1)

    # Provjera postoji li korisnik (problem hladnog starta).
    if user_id not in user_final_ratings.index:
        return pd.DataFrame()

    top_5_recommmendations = user_final_ratings.loc[user_id].sort_values(ascending=False)[
        0:5]
    top_5_data_frame = pd.DataFrame({
        'movieId': top_5_recommmendations.index,
        'predicted_rating': top_5_recommmendations.values
    })

    top_5_with_title = pd.merge(
        top_5_data_frame,
        movies[['movieId', 'title']],
        on='movieId'
    )

    top_5_with_title['poster_url'] = top_5_with_title['title'].apply(
        get_movie_poster)

    return top_5_with_title


def get_item_item_recommendations(user_id, k_neighbours=30):
    try:
        # Filtriranje podataka
        movie_counts = ratings['movieId'].value_counts()

        filtered_ratings = ratings[
            ratings['movieId'].isin(movie_counts[movie_counts >= 20].index)
        ]
        train, test = train_test_split(
            filtered_ratings, test_size=0.3, random_state=42)

        item_data = train.pivot(
            index='movieId', columns='userId', values='rating'
        ).fillna(0)

        dummy_train = train.copy()
        dummy_train['rating'] = dummy_train['rating'].apply(
            lambda x: 0 if x > 0 else 1)
        dummy_train = dummy_train.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(1)

        item_similarity = cosine_similarity(item_data)
        item_similarity[np.isnan(item_similarity)] = 0

        for i in range(len(item_similarity)):
            sorted_indices = np.argsort(item_similarity[i])[::-1]
            item_similarity[i, sorted_indices[k_neighbours:]] = 0

        # Predviđanje ocjena.
        sum_weights = np.array([np.abs(item_similarity).sum(axis=1)]).T
        sum_weights[sum_weights == 0] = 1
        item_predicted = np.dot(item_similarity, item_data) / sum_weights

        # Skaliranje ocjena na raspon 0.5-5.
        min_rating = 0.5
        max_rating = 5.0
        pred_min = item_predicted.min()
        pred_max = item_predicted.max()

        if pred_max != pred_min:
            item_predicted = min_rating + (item_predicted - pred_min) * \
                (max_rating - min_rating) / (pred_max - pred_min)
        else:
            item_predicted = np.full_like(
                item_predicted, (min_rating + max_rating)/2)

        user_ratings = item_predicted.T
        user_final_ratings = pd.DataFrame(
            np.multiply(user_ratings, dummy_train),
            index=dummy_train.index,
            columns=dummy_train.columns
        )

        if user_id not in user_final_ratings.index:
            return pd.DataFrame()

        recommendations = user_final_ratings.loc[user_id].sort_values(ascending=False)[
            :5]

        result = pd.DataFrame({
            'movieId': recommendations.index,
            'predicted_rating': recommendations.values
        }).merge(
            movies[['movieId', 'title']],
            on='movieId',
            how='left'
        ).dropna(subset=['predicted_rating'])

        # Dodavanje postera
        if 'get_movie_poster' in globals():
            result['poster_url'] = result['title'].apply(
                lambda x: get_movie_poster(x) if pd.notnull(x) else None)

        return result

    except:
        print(f"Greška! Molim Vas pokušajte ponovno.")
        return pd.DataFrame()


# Validacija podataka. Imamo dvije funkcije koje računaju odgovarajuće metrike za user-user i item-item dio.
def evaluate_user_user():
    user_counts = ratings['userId'].value_counts()
    movie_counts = ratings['movieId'].value_counts()

    filtered_ratings = ratings[
        ratings['userId'].isin(user_counts[user_counts >= 20].index)
    ]

    model_train, model_test = train_test_split(
        filtered_ratings, test_size=0.30, random_state=42)

    test_user_features = model_test.pivot(
        index='userId', columns='movieId', values='rating').fillna(0)

    dummy_test = model_test.copy()
    dummy_test['rating'] = dummy_test['rating'].apply(
        lambda x: 1 if x > 0 else 0)
    dummy_test = dummy_test.pivot(
        index='userId', columns='movieId', values='rating').fillna(0)

    test_user_similarity = cosine_similarity(test_user_features)
    test_user_similarity[np.isnan(test_user_similarity)] = 0
    user_predicted_ratings_test = np.dot(
        test_user_similarity, test_user_features)
    test_user_final_rating = np.multiply(
        user_predicted_ratings_test, dummy_test)

    X = test_user_final_rating.copy()
    X = X[X > 0]
    scaler = MinMaxScaler(feature_range=(0.5, 5))
    scaler.fit(X)
    pred = scaler.transform(X)
    total_non_nan = np.count_nonzero(~np.isnan(pred))
    test = model_test.pivot(index='userId', columns='movieId', values='rating')
    diff_sqr_matrix = (test - pred)**2
    sum_of_squares_err = diff_sqr_matrix.sum().sum()
    rmse = np.sqrt(sum_of_squares_err/total_non_nan)
    mae = np.abs(pred - test).sum().sum()/total_non_nan
    return float(rmse), float(mae)


def evaluate_item_item():
    user_counts = ratings['userId'].value_counts()
    movie_counts = ratings['movieId'].value_counts()

    filtered_ratings = ratings[
        ratings['movieId'].isin(movie_counts[movie_counts >= 20].index)
    ]

    model_train, model_test = train_test_split(
        filtered_ratings, test_size=0.30, random_state=42)
    test_item_features = model_test.pivot(
        index='movieId', columns='userId', values='rating').fillna(0)

    dummy_test = model_test.copy()
    dummy_test['rating'] = dummy_test['rating'].apply(
        lambda x: 1 if x > 0 else 0)
    dummy_test = dummy_test.pivot(
        index='userId', columns='movieId', values='rating').fillna(0)

    test_item_similarity = cosine_similarity(test_item_features)
    test_item_similarity[np.isnan(test_item_similarity)] = 0
    item_predicted_ratings_test = np.dot(
        test_item_features.T, test_item_similarity)
    test_item_final_rating = np.multiply(
        item_predicted_ratings_test, dummy_test)

    X = test_item_final_rating.copy()
    X = X[X > 0]
    scaler = MinMaxScaler(feature_range=(0.5, 5))
    scaler.fit(X)
    pred = scaler.transform(X)
    total_non_nan = np.count_nonzero(~np.isnan(pred))
    test = model_test.pivot(index='userId', columns='movieId', values='rating')
    diff_sqr_matrix = (test - pred)**2
    sum_of_squares_err = diff_sqr_matrix.sum().sum()
    rmse = np.sqrt(sum_of_squares_err/total_non_nan)
    mae = np.abs(pred - test).sum().sum()/total_non_nan
    return float(rmse), float(mae)


# Dio programa vezan za korisničko sučelje.
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            user_id = int(request.form['user_id'])
            algorithm = request.form.get('algorithm', 'user_user')

            if algorithm == 'user_user':
                recommendations = get_user_user_recommendations(user_id)
            else:
                recommendations = get_item_item_recommendations(user_id)

            if recommendations.empty:
                return render_template('index.html',
                                       # Problem hladnog starta.
                                       error="Korisnik nije pronađen u našoj bazi! Molim Vas pokušajte ponovno.")

            return render_template('results.html',
                                   recommendations=recommendations.to_dict(
                                       'records'),
                                   user_id=user_id,
                                   algorithm=algorithm)
        except ValueError:
            return render_template('index.html',
                                   error="Molim Vas pokušajte s valjanom korisničkom oznakom!")

    return render_template('index.html')


@app.route('/evaluate', methods=['POST'])
def evaluate():
    algorithm = request.form.get('algorithm', 'user_user')

    try:
        if algorithm == 'user_user':
            rmse, mae = evaluate_user_user()
        else:
            rmse, mae = evaluate_item_item()

        return jsonify({
            'status': 'success',
            'rmse': rmse,
            'mae': mae
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
