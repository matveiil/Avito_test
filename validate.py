import pandas as pd
import re
import sys

def validate_submission(answer_csv_path='answer.csv', 
                        queries_parquet_path='data/benchmark_queries.parquet', 
                        items_parquet_path='data/benchmark_items.parquet'):
    
    print(f"Валидация файла: {answer_csv_path} ...")
    
    try:
        # Читаем как строки, чтобы pandas не отрезал нули и не сделал float из hex
        sub = pd.read_csv(answer_csv_path, dtype=str)
        queries = pd.read_parquet(queries_parquet_path)
        items = pd.read_parquet(items_parquet_path)
    except FileNotFoundError as e:
        print(f"Ошибка загрузки файлов: {e}")
        sys.exit(1)
        
    assert list(sub.columns) == ['query_id', 'answer'], "Неверные колонки. Ожидается ['query_id', 'answer']"
    assert len(sub) == len(queries), f"Строк {len(sub)}, ожидается {len(queries)}"
    assert sub['query_id'].nunique() == len(queries), "Найдены дублирующиеся query_id"
    assert set(sub['query_id']) == set(queries['query_id']), "Набор query_id не совпадает с benchmark_queries"
    
    valid_items = set(items['item_id'])
    
    for _, row in sub.iterrows():
        q_id = row['query_id']
        ans = str(row['answer']).strip()
        
        if not ans or ans == 'nan':
            continue 
            
        item_list = ans.split(' ')
        
        assert len(item_list) <= 50, f"[{q_id}]: В ответе {len(item_list)} объявлений (макс 50)"
        assert len(item_list) == len(set(item_list)), f"[{q_id}]: Дубликаты item_id в ответе"
        
        for item in item_list:
            assert len(item) == 16, f"[{q_id}]: Длина item_id '{item}' != 16"
            assert bool(re.match(r'^[0-9a-f]{16}$', item)), f"[{q_id}]: Формат '{item}' не hex"
            assert item in valid_items, f"[{q_id}]: Объявления '{item}' нет в benchmark_items"

    print("Успешно")

if __name__ == "__main__":
    validate_submission()