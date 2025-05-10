# AGENT-BRAIN

## Visão Geral

AGENT-BRAIN é um framework Python projetado para construir um "cérebro" de conversação autônomo. Nele, um módulo professor baseado em GPT-3.5 fornece respostas contextuais especializadas, enquanto o AgentBrain-ML atua como aluno, registrando e armazenando cada interação. À medida que acumula massa de dados suficiente, esse agente evolui por meio de aprendizado incremental, refinando seus próprios modelos internos. Com esse aprendizado contínuo, ele se torna totalmente independente, capaz de assumir o papel do professor sempre que o GPT-3.5 estiver offline ou indisponível. Assim, o Agent-Brain consolida um depósito de conhecimento próprio e garante continuidade e resiliência no atendimento, mesmo sem conexão externa.

---

## Índice

* [Pré-requisitos](#pré-requisitos)
* [Instalação](#instalação)
* [Variáveis de Ambiente](#variáveis-de-ambiente)
* [Estrutura do Projeto](#estrutura-do-projeto)
* [Componentes Principais](#componentes-principais)
* [Configuração do Banco de Dados](#configuração-do-banco-de-dados)
* [Execução Local](#execução-local)
* [Docker Compose](#docker-compose)
* [Templates de Prompt](#templates-de-prompt)
* [Fallback com SBERT](#fallback-com-sbert)
* [Logs e Depuração](#logs-e-depuração)
* [Extensões e Customizações](#extensões-e-customizações)

---

## Pré-requisitos

* Python 3.10+ com suporte a `venv`
* PostgreSQL 12+ (local ou containerizado)
* Acesso à Internet para chamadas API da OpenAI

---

## Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/zack-ia/agent-brain.git
   cd agent-brain
   ```
2. Crie e ative um ambiente virtual:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Instale dependências:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Variáveis de Ambiente

Configure um arquivo `.env` na raiz do projeto (gitignored) com as chaves:

```dotenv
# OpenAI
OPENAI_API_KEY=seu_token_openai

# PostgreSQL (caso use local ou container)
HOST_PGSQL=localhost
PORT_PGSQL=5432
NAME_PGSQL=agent_brain_db
USER_PGSQL=usuario_pg
PASS_PGSQL=senha_pg
```

Essas variáveis alimentam tanto a conexão ao PostgreSQL quanto a autenticação à API da OpenAI.

---

## Estrutura do Projeto

```text
AGENT-BRAIN/
├── agent_script/
│   ├── agent_brain.py       # Loop principal e integração GPT-ML
│   └── load_braindata.py    # Módulo de conexão e persistência no PostgreSQL
├── data/
│   └── info_data.json       # JSON com configurações do escritório
├── brain_data/
│   └── docker-compose.yaml  # Definição do serviço PostgreSQL
├── requirements.txt         # Dependências Python do projeto
└── .gitignore
```

---

## Componentes Principais

### 1. `agent_script/agent_brain.py`

* **Carga de ambiente** via `python-dotenv` e setup de logging.
* **Inicialização**:

  * Carrega `OPENAI_API_KEY` (GRAVE erro se ausente).
  * Lê `info_data.json` para compor o prompt do professor.
  * Conecta ao PostgreSQL (módulo `load_braindata`).
  * Carrega modelo SBERT (`paraphrase-multilingual-MiniLM-L12-v2`) para fallback local.
* **Registro de Usuário** interativo via terminal.
* **Fluxo de Conversação**:

  1. Armazena pergunta do usuário.
  2. Gera resposta do **professor** usando `gpt-3.5-turbo` com `PROFESSOR_TEMPLATE` e max\_tokens=300.
  3. Armazena resposta do professor.
  4. Gera resposta do **AgentBrain-ML** usando `gpt-3.5-turbo` + histórico e max\_tokens=150.
  5. Em caso de falha no Agent, usa `fallback_agent_response` baseado em similaridade SBERT.
  6. Armazena e exibe resposta do agente.

### 2. `agent_script/load_braindata.py`

* Conexão PostgreSQL via `psycopg2` com `RealDictCursor`.
* Criação automática de tabelas:

  * `users(id, name, email, username, created_at)`
  * `messages(id, user_id, role, content, created_at)`
* Funções CRUD:

  * `get_or_create_user`
  * `store_message`
  * `get_conversation_history`

---

## Configuração do Banco de Dados

Se estiver usando localmente, garanta um banco vazio conforme variáveis `.env`. O script `create_tables` valida/cria as tabelas no primeiro run.

---

## Execução Local

```bash
source venv/bin/activate
cd agent_script
python agent_brain.py
```

Siga o prompt para cadastrar um usuário e inicie a conversa interativa.

---

## Docker Compose

O serviço PostgreSQL está definido em `brain_data/docker-compose.yaml`:

```yaml
version: '3.8'
services:
  agent_brain:
    image: postgres:17.2
    container_name: agent_brain
    environment:
            POSTGRES_USER: ${USER_PGSQL}
            POSTGRES_PASSWORD: ${PASS_PGSQL}
            POSTGRES_DB: ${NAME_PGSQL}
    ports:
      - '127.0.0.1:5432:5432'
    volumes:
      - agent_brain:/var/lib/postgresql/data/
volumes:
  agent_brain:
```

Para iniciar:

```bash
docker-compose -f brain_data/docker-compose.yaml up -d
```

---

## Templates de Prompt

* **PROFESSOR\_TEMPLATE**: injeta dados estáticos do Escritório (`nome`, `endereco`, etc.) para o GPT-3.5-turbo.
* **AGENT\_SYSTEM\_TEMPLATE**: orienta o AgentBrain-ML a não vazar metadados e focar apenas na pergunta.

Personalize `data/info_data.json` para alterar contexto institucional.

---

## Fallback com SBERT

Caso a chamada à OpenAI falhe no Agent, o módulo de fallback:

1. Recupera último `HISTORY_LIMIT` pares `(pergunta, resposta)` de `messages`.
2. Calcula embeddings com SBERT.
3. Retorna a resposta do par com maior similaridade de cosseno.

Ajuste `HISTORY_LIMIT` em `agent_brain.py` para controlar profundidade.

---

## Logs e Depuração

* Formato de log: `%%(asctime)s %%(levelname)s %%(message)s`.
* Níveis configurados em `basicConfig(level=logging.INFO)`.
* Erros de conexão ou de API são logados com `logger.error`.

---

## Extensões e Customizações

* **Modelos**: troque `gpt-3.5-turbo` por versões customizadas ou `gpt-4` se disponível.
* **DB**: substitua `psycopg2` por `asyncpg` para performance assíncrona.
* **Fallback ML**: experimente outros SentenceTransformers ou índice FAISS para buscas de similaridade em larga escala.
* **Interface**: adapte `chat_loop` para uma API REST (FastAPI/Flask) ou frontend web.

---

> **Zack:** Projeto AGENT-BRAIN – Arquitetura de Conversação Híbrida.
> **Data:** 10 de Maio de 2025
