import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Carrega variáveis de ambiente do .env
load_dotenv()

DB_HOST = os.getenv("HOST_PGSQL")
DB_PORT = os.getenv("PORT_PGSQL")
DB_NAME = os.getenv("NAME_PGSQL")
DB_USER = os.getenv("USER_PGSQL")
DB_PASS = os.getenv("PASS_PGSQL")

def connect_db():
    """Conecta ao PostgreSQL e retorna o connection (autocommit=True)."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        print("Erro na conexão com o DB:", e)
        return None

def create_tables(conn):
    """Cria tabelas users e messages, se não existirem."""
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            username TEXT NOT NULL UNIQUE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );""")
    print("✅ Tabelas verificadas/criadas com sucesso.")

def get_or_create_user(conn, name, email, username):
    """Retorna o id do usuário existente ou insere um novo."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM users WHERE email = %s OR username = %s",
            (email, username)
        )
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute(
            "INSERT INTO users (name, email, username) VALUES (%s, %s, %s) RETURNING id",
            (name, email, username)
        )
        return cur.fetchone()[0]

def store_message(conn, user_id, role, content):
    """Armazena uma mensagem (pergunta, resposta GPT ou resposta Agent)."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO messages (user_id, role, content) VALUES (%s, %s, %s)",
            (user_id, role, content)
        )


def get_conversation_history(conn, user_id=None, limit=None):
    """
    Busca o histórico de mensagens.
    Se user_id for None, retorna todo o histórico;
    caso contrário, retorna apenas as mensagens daquele usuário.
    Se limit for fornecido, retorna apenas as últimas `limit` mensagens.
    """
    query_all = "SELECT role, content FROM messages ORDER BY id ASC"
    query_user = "SELECT role, content FROM messages WHERE user_id = %s ORDER BY id ASC"

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if user_id is None:
            cur.execute(query_all)
        else:
            cur.execute(query_user, (user_id,))
        rows = cur.fetchall()

    if limit is not None:
        # Retorna apenas as últimas 'limit' mensagens
        rows = rows[-limit:]

    return rows
