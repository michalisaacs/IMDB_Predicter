import pandas as pd
import numpy as np
import ast
def prepare_data(df: pd.DataFrame, crew_lookup_df: pd.DataFrame = None, is_train: bool = False, mappings: dict = None) -> pd.DataFrame:

    X = df.copy()
    
    if crew_lookup_df is None:
        crew_lookup_df = globals().get('crew_lookup_df', pd.DataFrame(columns=['tconst', 'directors']))
    if mappings is None:
        mappings = globals().get('trained_mappings', {})

    y_input = X['averageRating'].copy() if 'averageRating' in X.columns else None
    X.drop(columns=['averageRating', 'BoxOffice', 'budget', 'numVotes', 'Unnamed: 0', 'writers', 'originalTitle'], errors='ignore', inplace=True)
    
    if 'startYear' in X.columns:
        X['startYear'] = pd.to_numeric(X['startYear'], errors='coerce').fillna(2000)

    # -----------------------------------------------------------------
    # actor reputation feature
    # -----------------------------------------------------------------
    def parse_actor_list(val) -> list:
        if pd.isna(val) or val is None: return []
        if isinstance(val, list): return val
        if isinstance(val, str):
            val_cleaned = val.strip()
            if val_cleaned.startswith('[') and val_cleaned.endswith(']'):
                try: return ast.literal_eval(val_cleaned)
                except (ValueError, SyntaxError):
                    val_cleaned = val_cleaned.strip('[]').replace("'", "").replace('"', "")
                    return [actor.strip() for actor in val_cleaned.split(',') if actor.strip()]
            if val_cleaned: return [actor.strip() for actor in val_cleaned.split(',') if actor.strip()]
        return []

    actor_col = 'lead_actors_ids' if 'lead_actors_ids' in X.columns else 'cast'
    X['cleaned_actor_list'] = X[actor_col].apply(parse_actor_list) if actor_col in X.columns else [[] for _ in range(len(X))]
    
    if is_train and y_input is not None:
        actor_history = {}
        df_temp = X.copy()
        df_temp['target_rating'] = y_input
        df_exploded = df_temp.explode('cleaned_actor_list').dropna(subset=['cleaned_actor_list', 'startYear', 'target_rating'])
        
        for _, row in df_exploded.iterrows():
            actor = row['cleaned_actor_list']
            if actor not in actor_history: actor_history[actor] = []
            actor_history[actor].append((int(row['startYear']), float(row['target_rating'])))
            
        mappings['actor'] = {'scores': actor_history, 'global_mean': float(y_input.mean())}

    actor_repo = mappings.get('actor', {'scores': {}, 'global_mean': 5.5})
    
    def calculate_weighted_actor_score(row):
        actors = row['cleaned_actor_list']
        movie_year = row['startYear']
        if not actors or pd.isna(movie_year): return actor_repo['global_mean']
        
        min_year, max_year = int(movie_year) - 10, int(movie_year) - 1
        actor_scores = []
        
        for actor in actors:
            history = actor_repo['scores'].get(actor, [])
            relevant_ratings = [rating for year, rating in history if min_year <= year <= max_year]
            actor_scores.append(np.mean(relevant_ratings) if relevant_ratings else None)
            
        known_scores = [score for score in actor_scores if score is not None]
        if known_scores:
            N = len(known_scores)
            weights = [N - i for i in range(N)]
            return np.average(known_scores, weights=weights)
        return actor_repo['global_mean']

    X['weighted_actor_reputation_10yrs'] = X.apply(calculate_weighted_actor_score, axis=1)
    X.drop(columns=['cleaned_actor_list'], inplace=True, errors='ignore')

    # -----------------------------------------------------------------
    # director reputation feature
    # -----------------------------------------------------------------
    X = pd.merge(X, crew_lookup_df[['tconst', 'directors']], on='tconst', how='left')
    
    def _get_director_list(val):
        if pd.isna(val): return []
        return [d.strip() for d in str(val).split(',') if d.strip()]
        
    X['director_ids_list'] = X['directors'].apply(_get_director_list)
    X['year_numeric'] = pd.to_numeric(X['startYear'], errors='coerce')

    if is_train and y_input is not None:
        director_repo_dict = {}
        df_temp = X[['director_ids_list', 'year_numeric']].copy()
        df_temp['target_rating'] = y_input.values
        df_exploded = df_temp.explode('director_ids_list').dropna(subset=['director_ids_list'])
        
        for director_id, group in df_exploded.groupby('director_ids_list'):
            director_repo_dict[director_id] = group[['year_numeric', 'target_rating']].to_dict(orient='records')
            
        director_repo_dict['GLOBAL_DEFAULT'] = float(y_input.mean())
        mappings['director'] = director_repo_dict

    director_repo = mappings.get('director', {'GLOBAL_DEFAULT': 5.5})
    global_default = director_repo.get('GLOBAL_DEFAULT', 5.5)
    movie_scores = []
    
    for _, row in X.iterrows():
        current_year = row['year_numeric']
        directors_in_movie = row['director_ids_list']
        if not directors_in_movie or pd.isna(current_year):
            movie_scores.append(global_default)
            continue
            
        current_directors_scores = []
        for d_id in directors_in_movie:
            if d_id in director_repo:
                past_movies = [m['target_rating'] for m in director_repo[d_id] if (current_year - 10) <= m['year_numeric'] < current_year]
                if past_movies: current_directors_scores.append(np.mean(past_movies))
                
        movie_scores.append(np.mean(current_directors_scores) if current_directors_scores else global_default)
        
    X['weighted_director_reputation_10yrs'] = movie_scores
    X.drop(columns=['directors', 'director_ids_list', 'year_numeric'], inplace=True, errors='ignore')

    # ----------------------
    # movie runtime feature (short/medium/long) - 3 categories
    # -----------------------
    if is_train:
        mappings['runtime'] = {'median_value': X['runtimeMinutes'].median() if X['runtimeMinutes'].notna().any() else 90.0}
    
    runtimes = pd.to_numeric(X['runtimeMinutes'], errors='coerce').fillna(mappings.get('runtime', {}).get('median_value', 90.0))
    X['runtime_short'] = runtimes.apply(lambda x: 1 if 60 <= x <= 90 else 0)
    X['runtime_medium'] = runtimes.apply(lambda x: 1 if 90 < x <= 140 else 0)
    X['runtime_long'] = runtimes.apply(lambda x: 1 if 140 < x <= 300 else 0)

    # -----------------
    # genre features - one-hot encoding for top genres + missing/other flags
    # --------------------
    def _clean_genres_to_list(val):
        if pd.isna(val) or val is None: return []
        return [cleaned_g for g in str(val).split(',') if (cleaned_g := g.strip().replace('[', '').replace(']', '').replace("'", "").replace('"', ""))]

    X['cleaned_genre_list'] = X['genres'].apply(_clean_genres_to_list)
    
    if is_train:
        all_genres = [g for row in X['cleaned_genre_list'] for g in row]
        mappings['genre'] = {'top_genres': pd.Series(all_genres).value_counts().nlargest(11).index.tolist()}
        
    top_genres_list = mappings.get('genre', {}).get('top_genres', [])
    X['genre_missing'] = X['cleaned_genre_list'].apply(lambda x: 1 if len(x) == 0 else 0)
    
    for genre in top_genres_list:
        X[f'genre_{genre.lower()}'] = X['cleaned_genre_list'].apply(lambda x: 1 if genre in x else 0)
        
    X['genre_other'] = X['cleaned_genre_list'].apply(lambda x: 1 if len(x) > 0 and not any(g in top_genres_list for g in x) else 0)
    X.drop(columns=['cleaned_genre_list'], inplace=True, errors='ignore')

    # -----------------------------------------------------------------
    # language features - is_english + is_unknown
    # -----------------------------------------------------------------
    def check_language_status(row):
        lang = str(row.get('Language', '')).lower().strip()
        # אם השפה חסרה או מסומנת כלא ידועה
        if pd.isna(row.get('Language')) or lang in ['nan', 'unknown', '']:
            return 1, 0  # unknown=1, is_english=0
        
        is_eng = 1 if 'english' in lang else 0
        return 0, is_eng  # unknown=0, לפי הבדיקה

    status_results = X.apply(check_language_status, axis=1)
    X['is_language_unknown'] = [res[0] for res in status_results]
    X['is_english_language'] = [res[1] for res in status_results]

    if is_train:
        globals()['trained_mappings'] = mappings


    final_cols_order = [f'genre_{g.lower()}' for g in top_genres_list] + \
                       ['genre_missing', 'genre_other', 'runtime_short', 'runtime_medium', 'runtime_long', \
                        'weighted_actor_reputation_10yrs', 'weighted_director_reputation_10yrs', \
                        'is_english_language', 'is_language_unknown']
                       
    X_final = X.reindex(columns=final_cols_order, fill_value=0.0).fillna(0.0)
    return X_final