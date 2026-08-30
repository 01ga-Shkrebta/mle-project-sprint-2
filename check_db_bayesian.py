import psycopg2

conn = psycopg2.connect(
    host='rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net',
    port='6432',
    dbname='playground_mle_20260624_4fe40d5155',
    user='mle_20260624_4fe40d5155_freetrack',
    password='d6d18edc4151436aa05e10149a641507',
    sslmode='require'
)

cur = conn.cursor()

# 1. Проверяем записи в runs
cur.execute("""
    SELECT run_uuid, name, status, lifecycle_stage
    FROM runs
    WHERE name LIKE '%bayesian%' OR name LIKE '%model_bayesian%'
    ORDER BY start_time DESC
    LIMIT 10
""")
rows = cur.fetchall()
print('=== RUNS ===')
print('Найдено записей:', len(rows))
for row in rows:
    print(row)

# 2. Проверяем теги
cur.execute("""
    SELECT t.run_uuid, t.key, t.value
    FROM tags t
    JOIN runs r ON r.run_uuid = t.run_uuid
    WHERE r.name LIKE '%bayesian%'
    LIMIT 20
""")
rows = cur.fetchall()
print('\n=== TAGS ===')
for row in rows:
    print(row)

cur.close()
conn.close()
