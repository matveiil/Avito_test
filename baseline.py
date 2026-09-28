import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import gc

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', ' ', text).lower()
    return re.sub(r'\s+', ' ', text).strip()

def main():
    print("1. Загрузка данных")
    items = pd.read_parquet('data/benchmark_items.parquet')
    queries = pd.read_parquet('data/benchmark_queries.parquet')

    print("2. Предобработка текстов")
    items['index_text'] = (
        items['item_title_raw'].fillna('') + ' ' + 
        items['item_title_raw'].fillna('') + ' ' + 
        items['item_infm_params_text'].fillna('') + ' ' + 
        items['item_description_raw'].fillna('').str[:150]
    ).apply(clean_text)

    queries['query_text'] = (
        queries['search_query'].fillna('') + ' ' + 
        queries['search_infm_params_text'].fillna('')
    ).apply(clean_text)

    print("3. TF-IDF")
    vectorizer = TfidfVectorizer(
        analyzer='word', 
        ngram_range=(1, 2), 
        min_df=2, 
        max_df=0.8, 
        sublinear_tf=True
    )
    item_vectors = vectorizer.fit_transform(items['index_text'])
    query_vectors = vectorizer.transform(queries['query_text'])

    print("4. Вычисление текстовой релевантности...")
    scores = query_vectors.dot(item_vectors.T).toarray().astype(np.float32)

    print("5. Применение логических фильтров и тай-брейкеров")
    
    q_loc = queries['search_location_id'].fillna(-1).to_numpy()[:, None]
    i_loc = items['item_location_id'].fillna(-1).to_numpy()[None, :]
    loc_match = (q_loc == i_loc) & (q_loc != -1)
    scores[loc_match] += 1.0
    del loc_match, q_loc, i_loc
    gc.collect()

    q_cat = queries['search_category'].fillna('q_none').astype(str).to_numpy()[:, None]
    i_cat = items['item_category_id'].fillna('i_none').astype(str).to_numpy()[None, :]
    cat_match = (q_cat == i_cat) & (q_cat != 'q_none')
    scores[cat_match] += 0.05
    del cat_match, q_cat, i_cat
    gc.collect()

    rating = items['item_rating'].fillna(0).to_numpy() / 5.0
    reviews = np.log1p(items['item_rating_reviews_count'].fillna(0).to_numpy())
    reviews_norm = reviews / (reviews.max() + 1e-9)

    quality_boost = (rating * 0.01 + reviews_norm * 0.01).astype(np.float32)
    scores += quality_boost
    
    del quality_boost, rating, reviews, reviews_norm
    gc.collect()

    print("6. Извлечение Топ-50")
    top_50_indices = np.argpartition(-scores, 50, axis=1)[:, :50]
    
    item_ids_array = items['item_id'].to_numpy()
    predictions = []

    for i in range(len(queries)):
        top_idx = top_50_indices[i]
        sorted_top = top_idx[np.argsort(-scores[i, top_idx])]
        predictions.append(item_ids_array[sorted_top])

    print("7. Сохранение answer.csv")
    answer = pd.DataFrame({
        'query_id': queries['query_id'],
        'answer': [' '.join(top) for top in predictions],
    })

    answer.to_csv('answer.csv', index=False)
    print("Готово.")

if __name__ == '__main__':
    main()