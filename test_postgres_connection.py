import psycopg

try:
    with psycopg.connect(
        host="localhost",
        port=5432,
        dbname="supply_chain_ai",
        user="postgres",
        password="L9032303406lc@"
    ) as connection:

        with connection.cursor() as cursor:
            cursor.execute("SELECT version()")
            result = cursor.fetchone()

            print("Connected successfully!")
            print(result[0])

except Exception as error:
    print("Database connection failed:", error)