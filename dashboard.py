# --- IMPORTAÇÕES INICIAIS ---
import streamlit as st
import pandas as pd
import plotly.express as px
import hashlib
from streamlit_extras.metric_cards import style_metric_cards 
from streamlit_extras.colored_header import colored_header 

# --- NOVAS IMPORTAÇÕES (do script ML) ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# --- NOVAS IMPORTAÇÕES (BANCO DE DADOS) ---
from sqlalchemy import create_engine
# --- CONEXÃO COM O BANCO DE DADOS ---
@st.cache_resource
def conectar_mysql():
    user = st.secrets["db"]["user"]
    password = st.secrets["db"]["password"]
    host = st.secrets["db"]["host"]
    port = st.secrets["db"]["port"]
    database = st.secrets["db"]["database"]

    url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(url)
    return engine

@st.cache_data
def carregar_todas_tabelas(_engine):
    try:
        tabelas = pd.read_sql("SHOW TABLES;", engine)
        lista_tabelas = tabelas.iloc[:, 0].tolist()

        dados = {}
        for tb in lista_tabelas:
            try:
                df = pd.read_sql(f"SELECT * FROM `{tb}`;", engine)
                dados[tb] = df
            except:
                pass  # apenas ignora se alguma tabela der erro

        return dados

    except Exception as e:
        return {}
engine = conectar_mysql()
dados = carregar_todas_tabelas(engine)


# ===========================================================
# 🎨 FUNÇÃO DE CSS CUSTOMIZADO (UI KIT CANNOLI)
# ===========================================================
def load_custom_css():
    """
    Carrega o CSS customizado baseado no UI Kit da Cannoli.
    """
    # Importa as fontes Poppins e Inter
    fonts_import = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
"""

    # Tokens de Design (do seu UI Kit JSON)
    tokens = """
:root {
    --font-primary: 'Poppins', sans-serif;
    --font-secondary: 'Inter', sans-serif;
    
    --color-primary: #FF7A00;
    --color-secondary: #FFE4CC;
    --color-dark: #4B2E1E;
    --color-background: #F9FAFB;
    --color-white: #FFFFFF;
    --color-text-primary: #1E293B;
    --color-text-secondary: #6B7280;
    --color-success: #10B981;
    --color-error: #EF4444;
    --color-border: #E5E7EB;
    
    --radius-medium: 10px;
    --radius-large: 12px;
    --shadow-card: 0 2px 8px rgba(0,0,0,0.08);
}
"""
    
    # Estilos Globais e de Componentes
    styles = """
/* 1. ESTILOS GERAIS */
body, .stApp {
    font-family: var(--font-primary);
    color: var(--color-text-primary);
    background-color: var(--color-background); /* Fundo Suave #F9FAFB */
}
h1, h2, h3 {
    font-family: var(--font-primary);
    color: var(--color-dark);
    font-weight: 700;
}
h1 { font-size: 32px; }
h2 { font-size: 24px; }
h3 { font-size: 20px; font-weight: 600; }

/* Remove o CSS padrão do Streamlit que injeta */
.stApp > header {
    display: none;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--color-white);
}

/* 2. COMPONENTES: CARDS (Seções) */
[data-testid="stExpander"] {
    background-color: var(--color-white);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-large);
    box-shadow: var(--shadow-card);
    margin-bottom: 20px;
}
[data-testid="stExpander"] > details > summary {
    padding: 16px 24px; /* Padding do Título */
    font-size: 18px;
    font-weight: 600;
    color: var(--color-dark);
}
[data-testid="stExpander"] > details > div {
    padding: 0 24px 24px 24px; /* Padding do Conteúdo */
}

/* 3. COMPONENTES: CARDS DE MÉTRICA (KPIs) */
[data-testid="stMetric"] {
    background-color: var(--color-white);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-large);
    box-shadow: var(--shadow-card);
    padding: 24px;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-primary);
    color: var(--color-text-secondary);
    font-size: 16px;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-secondary); /* Inter para números */
    font-size: 30px;
    font-weight: 700;
    color: var(--color-dark);
}

/* 4. COMPONENTES: BOTÕES */
[data-testid="stButton"] button {
    background-color: var(--color-primary);
    color: var(--color-white);
    border: none;
    padding: 12px 20px;
    border-radius: 8px;
    font-weight: 500;
    font-size: 16px;
    transition: background-color 0.2s;
}
[data-testid="stButton"] button:hover {
    background-color: #E96B00; /* Hover (conforme seu UI Kit) */
    color: var(--color-white);
}

/* 5. COMPONENTES: TABELAS */
[data-testid="stDataFrame"] thead th {
    background-color: transparent;
    color: var(--color-dark);
    font-weight: 600;
    font-size: 14px;
}
[data-testid="stDataFrame"] tbody tr:nth-of-type(even) {
    background-color: #F9FAFB; /* Linha alternada */
}
[data-testid="stDataFrame"] tbody tr:hover {
    background-color: #FFF1E6; /* Hover Laranja */
    color: var(--color-dark);
}
/* NOVO: Garante que o texto dentro do cabeçalho use a fonte primária */
[data-testid="stDataFrame"] .st-bd { 
    font-family: var(--font-primary);
    font-weight: 600;
}
"""
    
    st.markdown(f"<style>{fonts_import}{tokens}{styles}</style>", unsafe_allow_html=True)

# ===========================================================
# 🧩 CONFIGURAÇÃO GERAL
# ===========================================================
st.set_page_config(
    page_title="Cannoli DataVision", layout="wide")

# --- NOVO: CHAMA A FUNÇÃO DE CSS ---
load_custom_css()

# ===========================================================
# 🔐 LOGIN / AUTENTICAÇÃO (via cadastro.csv)
# ===========================================================
def hash_pwd(password: str) -> str:
    return hashlib.sha256(str(password).encode("utf-8")).hexdigest()

@st.cache_data
def load_cadastro(path: str = "data/cadastro.csv") -> pd.DataFrame:
    try:
        cad = pd.read_csv(path, sep=",", encoding="utf-8")
    except UnicodeDecodeError:
        cad = pd.read_csv(path, sep=",", encoding="latin-1")
    rename_map = {
        "id_restaurate": "id_restaurante",
        "nome_restarante": "nome_restaurante",
        "usuario": "login",
        "password": "senha"
    }
    cad.rename(columns={k: v for k, v in rename_map.items()
                      if k in cad.columns}, inplace=True)
    if "id_restaurante" in cad.columns:
        cad["id_restaurante"] = pd.to_numeric(
            cad["id_restaurante"], errors="coerce").astype("Int64")
    cad["password_hash"] = cad["senha"].astype(str).apply(hash_pwd)
    cad["role"] = cad["role"].astype(
        str).str.strip().str.lower()
    cad["nome_restaurante"] = cad.get(
        "nome_restaurante", pd.Series(dtype="object")).astype(str)
    return cad

cadastro = load_cadastro()
ID_TO_NAME = cadastro.set_index("id_restaurante")["nome_restaurante"].to_dict()
NAME_TO_ID = {v: k for k, v in ID_TO_NAME.items()}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None

def authenticate(username: str, password: str):
    row = cadastro.loc[cadastro["login"].astype(str) == str(username)]
    if row.empty:
        return False, None
    row = row.iloc[0]
    if hash_pwd(password) == row["password_hash"]:
        user = {
            "username": username,
            "role": row["role"],
            "id_restaurante": None if pd.isna(row.get("id_restaurante")) else int(row.get("id_restaurante")),
            "nome_restaurante": None if pd.isna(row.get("nome_restaurante")) else str(row.get("nome_restaurante")),
            "display": str(row.get("nome_restaurante")) if row["role"] == "restaurant" else "Admin"
        }
        return True, user
    return False, None

def login_form():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        # st.image("logo_cannoli.png", use_column_width=True) 
        st.title("Login de Acesso")
        with st.form("login_form"):
            username = st.text_input("Usuário")
            password = st.text_input("Senha", type="password")
            submit = st.form_submit_button("Entrar")
            if submit:
                ok, user = authenticate(username, password)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.user = user
                    st.success(f"✅ Logado como {user['display']}")
                    st.rerun()
                else:
                    st.error("Usuário ou senha incorretos")

def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()

if not st.session_state.logged_in:
    login_form()
    st.stop()

user = st.session_state.user

# ===========================================================
# 📊 DADOS (Carregamento principal)
# ===========================================================
@st.cache_data
def load_data():
    CQ_PATH = 'data/CampaignQueue_semicolon.csv'
    CAMPAIGN_PATH = 'data/Campaign_semicolon.csv'
    CUSTOMER_PATH = 'data/Customer_semicolon.csv'
    ORDER_PATH = 'data/Order_semicolon.csv'
    df1 = pd.read_csv(CQ_PATH, sep=';', encoding='latin-1')
    df2 = pd.read_csv(CAMPAIGN_PATH, sep=';', encoding='latin-1')
    df3 = pd.read_csv(CUSTOMER_PATH, sep=';', encoding='latin-1')
    df4 = pd.read_csv(ORDER_PATH, sep=';', encoding='latin-1')
    return df1, df2, df3, df4

df1, df2, df3, df4 = load_data()

def to_dt(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

# --- PRÉ-PROCESSAMENTO ---
to_dt(df1, ['scheduledAt', 'sendAt', 'createdAt', 'updatedAt'])
to_dt(df3, ['dateOfBirth', 'enrichedAt', 'createdAt', 'updatedAt'])
to_dt(df4, ['createdAt', 'updatedAt'])

for _df, col in [(df1, 'customer_id'), (df3, 'customer_id'), (df4, 'customer_id'), (df4, 'id_restaurante')]:
    if col in _df.columns:
        _df[col] = pd.to_numeric(_df[col], errors='coerce')

try:
    df2['id'] = pd.to_numeric(df2['id'], errors='coerce').astype('Int64')
except KeyError:
    st.error("Erro Crítico: O arquivo 'Campaign_semicolon.csv' não possui uma coluna 'id'. Verifique o arquivo.")
    st.stop()
except Exception as e:
    st.error(f"Erro ao processar 'Campaign_semicolon.csv': {e}")
    st.stop()

df1_sent = df1[df1['sendAt'].notnull()].copy()
df4_cols = ['customer_id', 'id_restaurante',
            'createdAt', 'totalAmount', 'status', 'salesChannel']
df4_subset = df4[df4_cols].copy()
df4_subset.rename(columns={'createdAt': 'createdAt_order'}, inplace=True)
df1_sent_subset = df1_sent[['customer_id',
                            'id_restaurante', 'sendAt', 'campaignId']].copy()
df1_sent_subset.rename(columns={'sendAt': 'sendAt_campaign'}, inplace=True)
df4_subset = df4_subset.merge(
    cadastro[['id_restaurante', 'nome_restaurante']],
    on='id_restaurante',
    how='left'
)
df_orders_campaigns = pd.merge(
    df4_subset,
    df1_sent_subset,
    on=['customer_id', 'id_restaurante'],
    how='left'
)
if 'nome_restaurante' not in df_orders_campaigns.columns or df_orders_campaigns['nome_restaurante'].isna().any():
    df_orders_campaigns['nome_restaurante'] = df_orders_campaigns['id_restaurante'].map(
        ID_TO_NAME)
df_orders_campaigns['time_diff'] = df_orders_campaigns['createdAt_order'] - \
    df_orders_campaigns['sendAt_campaign']

# ===========================================================
# 🔎 FILTROS (Sidebar)
# ===========================================================
# st.sidebar.image("logo_cannoli.png", use_column_width=True) 
st.sidebar.title(f"Painel {user['display']}")
st.sidebar.markdown(f"Conectado como: **{user['username']}** ({user['role']})")
if st.sidebar.button("Sair"):
    logout()
st.sidebar.markdown("---")

restaurantes = (
    cadastro.dropna(subset=['id_restaurante', 'nome_restaurante'])
    .sort_values('nome_restaurante')
    ['nome_restaurante'].unique().tolist()
)

# --- INÍCIO DA SEÇÃO DE FILTROS ---
with st.sidebar.expander("Filtros de Análise", expanded=True):
    
    # --- Filtro de Restaurante ---
    if user["role"] == "admin":
        restaurante_selecionado = st.selectbox(
            "Selecionar Restaurante", restaurantes)
        id_restaurante_selecionado = NAME_TO_ID.get(restaurante_selecionado)
    else:  # restaurant
        restaurante_selecionado = user["nome_restaurante"]
        id_restaurante_selecionado = user["id_restaurante"]
        st.info(f"Restaurante: **{restaurante_selecionado}**")

    st.markdown("---") # Separador
    st.caption("Filtros para a 'Aba 1: Visão Geral'") # Mais curto

    # --- Filtro de Canal ---
    canais = sorted(df_orders_campaigns['salesChannel'].dropna().unique().tolist())
    canal_selecionado = st.multiselect(
        "Canais de Venda:", # Label mais curto
        canais, 
        default=canais
    )
    
    # --- Filtro de Janela ---
    janelas = [7, 14, 30, 60]
    janela_selecionada = st.radio(
        "Janela de Dias (Impacto):", # Label mais curto
        janelas, 
        horizontal=True
    )
    
    st.markdown("---") # Separador

    # --- Filtro de Campanha (Lógica já filtra por restaurante) ---
    # 1. Obter IDs de campanhas do restaurante (de df1, que tem 'campaignId')
    camp_ids_restaurante = df1[df1['id_restaurante'] == id_restaurante_selecionado]['campaignId'].unique()
    
    # 2. Obter os nomes dessas campanhas (de df2, usando 'id' e 'name')
    df_nomes_campanhas = df2[df2['id'].isin(camp_ids_restaurante)][['id', 'name']].dropna()
    df_nomes_campanhas['name'] = df_nomes_campanhas['name'].astype(str)
    
    # 3. Criar lista de nomes para o filtro
    lista_nomes = sorted(df_nomes_campanhas['name'].unique().tolist())
    
    # 4. Lógica do Filtro
    if not lista_nomes:
        st.info("Este restaurante não possui campanhas nomeadas.")
        ids_campanhas_selecionadas = []
    else:
        campanhas_selecionadas_nomes = st.multiselect(
            "Campanhas Específicas:\n(Padrão: todas)", # <-- Label com quebra de linha
            lista_nomes, 
            default=lista_nomes
        )
        # 5. Obter os IDs dos nomes selecionados para filtrar
        ids_campanhas_selecionadas = df_nomes_campanhas[
            df_nomes_campanhas['name'].isin(campanhas_selecionadas_nomes)
        ]['id'].unique()
        
# --- FIM DA SEÇÃO DE FILTROS ---

# ===========================================================
# 🤖 FUNÇÕES DE MACHINE LEARNING E ESTRATÉGIA
# ===========================================================

# Constantes do ML
TEST_WEEKS, TEST_MONTHS = 4, 3
FORECAST_WEEKS, FORECAST_MONTHS = 4, 3
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#888", "axes.labelcolor": "#000000",
    "xtick.color": "#000000", "ytick.color": "#000000",
    "grid.color": "#CCCCCC", "text.color": "#000000",
    "savefig.facecolor": "#FFFFFF", "savefig.edgecolor": "#FFFFFF",
    "legend.facecolor": "#FFFFFF", "legend.edgecolor": "#FFFFFF"
})
fmt_moeda = lambda x, pos: f"R${x:,.0f}".replace(
    ",", "X").replace(".", ",").replace("X", ".")

def setup_axes(ax, freq):
    ax.grid(True, alpha=.25)
    ax.set_xlabel("Período")
    ax.set_ylabel("Vendas (R$)")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_moeda))
    if freq == "W":
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))

def make_series(orders, freq):
    df = orders.copy()
    if 'status' in df:
        df = df[df['status'] == 'CONCLUDED']
    if 'createdAt' not in df:
        raise ValueError("createdAt ausente em Order")
    df = df.set_index('createdAt').sort_index()
    rule = 'W-SUN' if freq == 'W' else 'MS'
    s = df['totalAmount'].resample(rule).sum().rename(
        'sales').asfreq(rule, fill_value=0.0)
    out = s.reset_index().rename(columns={'createdAt': 'period'})
    if freq == 'W':
        out['time_idx'] = (out['period'] - out['period'].min()).dt.days // 7
    else:
        b = out['period'].iloc[0]
        out['time_idx'] = (out['period'].dt.year -
                             b.year) * 12 + (out['period'].dt.month - b.month)
    return out

def camps_by_period(cq, freq):
    df = cq[cq['sendAt'].notna()].set_index('sendAt').sort_index()
    rule = 'W-SUN' if freq == 'W' else 'MS'
    c = df['campaignId'].resample(rule).nunique().rename(
        'num_campaigns').asfreq(rule, fill_value=0)
    return c.reset_index().rename(columns={'sendAt': 'period'})

def fit_eval_forecast(serie, camps, test_last, horizon, freq):
    df = serie.merge(camps, on='period', how='left')
    df['num_campaigns'] = df['num_campaigns'].fillna(0.0)
    feats = ['time_idx', 'num_campaigns']
    df = df.dropna()
    if len(df) <= test_last + 2:
        raise ValueError(
            "Série de dados muito curta para criar a previsão. (Poucos meses/semanas com dados)")
    tr, te = df.iloc[:-test_last], df.iloc[-test_last:]
    m = LinearRegression().fit(tr[feats], tr['sales'])
    pred = m.predict(te[feats])
    r2 = r2_score(te['sales'], pred)
    mae = mean_absolute_error(te['sales'], pred)
    last_idx = int(df['time_idx'].max())
    mean_c = float(df['num_campaigns'].tail(max(3, test_last)).mean())
    fut = pd.DataFrame({'time_idx': [last_idx + i for i in
                                    range(1, horizon + 1)], 'num_campaigns': mean_c})
    rule = 'W-SUN' if freq == 'W' else 'MS'
    fut['period'] = pd.date_range(
        start=df['period'].max(), periods=horizon + 1, freq=rule)[1:]
    fut['y_hat'] = m.predict(fut[['time_idx', 'num_campaigns']])
    return {'df': df, 'test': te, 'y_pred': pred, 'r2': r2, 'mae': mae, 'future': fut}

def plot_line(res, title, freq):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(res['df']['period'], res['df']['sales'], lw=2.2, label='Vendas Reais')
    tp = res['test'][['period']].copy()
    tp['y_pred'] = res['y_pred']
    ax.plot(tp['period'], tp['y_pred'], '--o', ms=4, lw=2.0, label='Previsto (Teste)')
    ax.plot(res['future']['period'], res['future']
            ['y_hat'], '-o', ms=5, lw=2.0, label='Projeção')
    ax.fill_between(res['future']['period'], res['future']['y_hat'], alpha=.18)
    if len(res['future']):
        lr = res['future'].iloc[-1]
        ax.annotate(fmt_moeda(lr['y_hat'], None), xy=(lr['period'], lr['y_hat']),
                    xytext=(0, 10), textcoords='offset points', fontsize=9, ha='center')
    setup_axes(ax, freq)
    ax.set_title(title, fontsize=13)
    ax.legend(loc='upper left', frameon=False)
    plt.tight_layout()
    return fig

def gerar_estrategia_ia(resultados_ml, nome_restaurante):
    insights = []
    try:
        future_df = resultados_ml['future']
        historical_df = resultados_ml['df']
        r2 = resultados_ml['r2']
        projecao_inicial = future_df['y_hat'].iloc[0]
        projecao_final = future_df['y_hat'].iloc[-1]
        tendencia = "estável"
        if projecao_final > projecao_inicial * 1.05:
            tendencia = "de alta"
        elif projecao_final < projecao_inicial * 0.95:
            tendencia = "de baixa"
        insights.append(f"A projeção para {nome_restaurante} nos próximos períodos é **{tendencia}**.")
        correlacao = historical_df['sales'].corr(historical_df['num_campaigns'])
        if tendencia == "de baixa":
            insights.append("#### 🚨 Alerta de Ação\nA previsão indica uma possível queda nas vendas. Este é um momento crucial para intervir e reverter o cenário.")
        elif tendencia == "de alta":
            insights.append("#### 🚀 Oportunidade\nA previsão é positiva! É uma ótima hora para acelerar o crescimento e superar as metas.")
        if correlacao > 0.4:
            insights.append(
                "#### 📈 Estratégia Recomendada (Foco em Volume)\n"
                "Nossos dados históricos mostram uma **ligação positiva** entre o *número* de campanhas enviadas e o total de vendas. "
                "Para o seu restaurante, mais campanhas parecem gerar mais vendas.\n\n"
                "**Ações:**\n"
                "* **Acelere:** Se a tendência for de baixa, considere aumentar o número de campanhas para reverter o cenário.\n"
                "* **Mantenha o Ritmo:** Se a tendência for de alta, mantenha ou aumente a frequência de campanhas para capturar o máximo de vendas."
            )
        else:
             insights.append(
                "#### 🎯 Estratégia Recomendada (Foco em Qualidade)\n"
                "Nossos dados mostram que simplesmente enviar *mais* campanhas pode não ser a resposta (correlação baixa). O foco deve ser na *qualidade* e *relevância*.\n\n"
                "**Ações:**\n"
                "* **Segmente:** Crie campanhas para públicos específicos (ex: clientes que não compram há 30 dias, clientes que só compram sobremesa).\n"
                "* **Teste Ofertas:** Varie suas campanhas. Tente combos, descontos progressivos, ou frete grátis em vez de % de desconto.\n"
                "* **Temas:** Crie eventos temáticos (ex: 'Semana do Combo Família', 'Festival de Sobremesas') para gerar urgência."
            )
        if r2 < 0.3:
             insights.append(
                f"#### ⚠️ Nota de Confiança (R²: {r2:.2f})\n"
                "O modelo teve dificuldade em prever suas vendas (R² baixo). Isso sugere que suas vendas são muito voláteis ou, mais provável, dependem de fatores que não estamos medindo (como o *tipo* de oferta, clima, ou eventos locais). "
                "Isso **reforça** a necessidade de focar em **testes e qualidade** (Estratégia 🎯) em vez de apenas volume."
            )
        
        return "\n\n".join(insights)
        
    except Exception as e:
        return f"Erro ao gerar estratégia: {e}. Verifique se há dados suficientes."

@st.cache_data
def get_ml_forecast(_all_orders_data, _all_cq_data, restaurant_id, freq):
    orders_ml_input = _all_orders_data[_all_orders_data['id_restaurante'] == restaurant_id].copy()
    cq_ml_input = _all_cq_data[_all_cq_data['id_restaurante'] == restaurant_id].copy()
    if orders_ml_input.empty:
        raise ValueError(f"Não há dados de 'Pedidos' (Orders) para o restaurante ID {restaurant_id} para treinar o modelo.")
    if freq == 'W':
        test_last, horizon = TEST_WEEKS, FORECAST_WEEKS
    else:
        test_last, horizon = TEST_MONTHS, FORECAST_MONTHS
    serie = make_series(orders_ml_input, freq)
    camps = camps_by_period(cq_ml_input, freq)
    if serie.empty or len(serie) < (test_last + 2):
         raise ValueError(f"Dados insuficientes (poucas semanas/meses com vendas) para o restaurante ID {restaurant_id} após a agregação.")
    results = fit_eval_forecast(serie, camps, test_last, horizon, freq)
    return results

# ===========================================================
# 🖥️ ESTRUTURA DO DASHBOARD (COM ABAS)
# ===========================================================
st.title(f"Análise de Campanhas: {restaurante_selecionado}")
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "📈 1. Visão Geral do Impacto", 
    "🤖 2. Previsão de Vendas (ML)", 
    "🧠 3. Análise e Estratégia"
])

# --- CONTEÚDO DA ABA 1: VISÃO GERAL ---
with tab1:
    
    # --- CÁLCULO DOS DADOS (PRÉ-FILTRO) ---
    df_pre_filtered = df_orders_campaigns[
        (df_orders_campaigns['id_restaurante'] == id_restaurante_selecionado) &
        (df_orders_campaigns['salesChannel'].isin(canal_selecionado)) &
        (df_orders_campaigns['status'] == 'CONCLUDED') &
        (df_orders_campaigns['time_diff'] >= pd.Timedelta(seconds=0)) &
        (df_orders_campaigns['time_diff'] <= pd.Timedelta(days=janela_selecionada)) &
        (df_orders_campaigns['campaignId'].isin(ids_campanhas_selecionadas)) # <-- FILTRO DE CAMPANHA DA SIDEBAR
    ].copy()

    df_filtered = df_pre_filtered.copy()
    
    impact = df_filtered.groupby(['campaignId', 'salesChannel']).agg(
        total_pedidos=('customer_id', 'count'),
        valor_total_vendas=('totalAmount', 'sum')
    ).reset_index()
    
    # --- Merge para buscar os Nomes das Campanhas ---
    df_nomes_camp = df2[['id', 'name']].drop_duplicates()
    impact = impact.merge(
        df_nomes_camp, 
        left_on='campaignId', # ID de df1/df4
        right_on='id',        # ID de df2
        how='left'
    )
    impact['campaignName'] = impact['name'].fillna(impact['campaignId'].astype(str)) # Usa o ID como fallback
    
    # Mapeamento de nomes de canal
    channel_rename_map = {
        'SOCIAL_MEDIA': 'Mídia Social',
        'DELIVERY_APP': 'DELIVERYVP',
        'WHATSAPP': 'WhatsApp',
        'IFOOD': 'IFOOD',
        '99FOOD': '99FOOD',
        'SITE': 'SITE',
        'EPIDOCA': 'EPIDOCA',
        'DELIVERYVP': 'DELIVERYVP'
    }
    impact['canal_display'] = impact['salesChannel'].map(channel_rename_map).fillna(impact['salesChannel'])

    CORES_PROFISSIONAIS = {
        'IFOOD': '#EA1D2C',
        '99FOOD': '#FFC700',
        'WhatsApp': '#25D366',
        'SITE': '#FF7A00',         
        'EPIDOCA': '#808080',   
        'App de Delivery': '#242551',
        'DELIVERYVP': '#242551'
    }
    
    # --- MÉTRICAS (VISÃO GERAL) ---
    st.subheader("Resultados da Análise de Impacto")
    st.caption(f"Filtrando pedidos concluídos em até {janela_selecionada} dias após uma campanha.")
    
    total_pedidos = int(impact['total_pedidos'].sum()) if not impact.empty else 0
    total_vendas = float(impact['valor_total_vendas'].sum()
                         ) if not impact.empty else 0.0
    media_ticket = (total_vendas / total_pedidos) if total_pedidos > 0 else 0.0
    num_campanhas = int(impact['campaignId'].nunique()) if not impact.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🛒 Total de Pedidos (Filtrados)", f"{total_pedidos:,}".replace(",", "."))
    col2.metric("💰 Valor Total de Vendas (Filtrados)", f"R$ {total_vendas:,.2f}".replace(
        ",", "X").replace(".", ",").replace("X", "."))
    col3.metric("🎯 Ticket Médio (Filtrado)", f"R$ {media_ticket:,.2f}".replace(
        ",", "X").replace(".", ",").replace("X", "."))
    col4.metric("📦 Nº de Campanhas (com impacto)", f"{num_campanhas}")

    st.markdown("---")

    # --- GRÁFICOS (VISÃO GERAL) ---
    with st.expander("📊 Análise de Volume (Pedidos e Vendas)", expanded=True):
        # === GRÁFICO 1: PEDIDOS ===
        fig_pedidos = px.bar(
            impact, x='campaignName',
            y='total_pedidos', 
            color='canal_display',
            barmode='group',
            title='Total de Pedidos por Campanha', 
            text='total_pedidos',
            color_discrete_map=CORES_PROFISSIONAIS 
        )
        fig_pedidos.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)', 
            font_color='#1E293B', 
            xaxis_title='Campanha',
            yaxis_title='Total de Pedidos',
            title_x=0.5, 
            legend_title_text='Canal',
            font=dict(family="Inter, sans-serif", size=12),
            yaxis=dict(showgrid=True, gridcolor='lightgray')
        )
        fig_pedidos.update_traces(
            texttemplate='%{y}',
            textposition='outside',
            textfont_color='#1E293B',
            textfont_size=11,
            marker_line_width=0
        )
        st.plotly_chart(fig_pedidos, use_container_width=True)

        st.markdown("---")

        # === GRÁFICO 2: VENDAS ===
        fig_vendas = px.bar(
            impact, x='campaignName',
            y='valor_total_vendas', 
            color='canal_display',
            barmode='group',
            title='Valor Total de Vendas por Campanha', 
            text='valor_total_vendas',
            color_discrete_map=CORES_PROFISSIONAIS
        )
        fig_vendas.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)', 
            font_color='#1E293B',
            xaxis_title='Campanha',
            yaxis_title='Valor Total (R$)',
            title_x=0.5, 
            legend_title_text='Canal',
            font=dict(family="Inter, sans-serif", size=12),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='lightgray', tickformat='R$ .2s')
        )
        fig_vendas.update_traces(
            texttemplate='R$ %{y:,.2f}', 
            textposition='outside',
            textfont_color='#1E293B',
            textfont_size=11,
            marker_line_width=0
        )
        st.plotly_chart(fig_vendas, use_container_width=True)

    with st.expander("🏆 Análise de Canais e Ranking", expanded=True):

        # === GRÁFICO 3: PIZZA ===
        fig_pizza = px.pie(
            impact, 
            names='canal_display',
            values='valor_total_vendas',
            title='Participação dos Canais nas Vendas',
            color_discrete_map=CORES_PROFISSIONAIS 
        )
        fig_pizza.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)', 
            font_color='#1E293B',
            title_x=0.5, 
            showlegend=True,
            legend_title_text='Canal',
            font=dict(family="Inter, sans-serif", size=12)
        )
        fig_pizza.update_traces(
            textinfo='percent+value', 
            hovertemplate='%{label}<br>Valor: R$ %{value:,.2f}<br>Participação: %{percent}<extra></extra>',
            marker=dict(line=dict(color='#FFFFFF', width=1))
        )
        st.plotly_chart(fig_pizza, use_container_width=True)
    
        st.markdown("---")

        # === GRÁFICO 4: RANKING HORIZONTAL ===
        impact_rank = impact.sort_values(by='valor_total_vendas', ascending=True)
        fig_rank = px.bar(
            impact_rank, y='campaignName',
            x='valor_total_vendas', 
            color='canal_display',
            orientation='h',
            text='valor_total_vendas', title='Ranking de Campanhas por Valor Total de Vendas',
            color_discrete_map=CORES_PROFISSIONAIS
        )
        fig_rank.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)', 
            font_color='#1E293B',
            xaxis_title='Valor Total de Vendas (R$)', yaxis_title='Campanha',
            title_x=0.5, 
            legend_title_text='Canal',
            font=dict(family="Inter, sans-serif", size=12),
            xaxis=dict(showgrid=True, gridcolor='lightgray', tickformat='R$ .2s'), 
            yaxis=dict(showgrid=False)
        )
        fig_rank.update_traces(
            texttemplate='R$ %{x:,.2f}', 
            textposition='outside',
            textfont_color='#1E293B',
            textfont_size=11,
            marker_line_width=0
        )
        st.plotly_chart(fig_rank, use_container_width=True)

# --- CONTEÚDO DA ABA 2: PREVISÃO (ML) ---
with tab2:
    st.info("Esta análise usa o **histórico completo** do restaurante. Os filtros 'Canal de Venda' e 'Janela de Dias' da sidebar são ignorados aqui.")
    
    sub_tab_mensal, sub_tab_semanal = st.tabs(["Projeção Mensal", "Projeção Semanal"])

    with sub_tab_mensal:
        try:
            with st.expander("🤖 Previsão Mensal", expanded=True):
                res_m = get_ml_forecast(df4, df1, id_restaurante_selecionado, 'M')
                fig_m = plot_line(res_m,
                                  f'Mensal – Vendas Reais x Previstas ({restaurante_selecionado})',
                                  'M')
                st.pyplot(fig_m)
                
                st.subheader("Métricas do Modelo (Mensal)")
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("R² (Score do Modelo)", f"{res_m['r2']:.3f}",
                              help="Quanto mais próximo de 1.0, melhor o modelo explica os dados passados.")
                col_m2.metric("MAE (Erro Médio Absoluto)", f"R$ {res_m['mae']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
                              help="O erro médio (para mais ou para menos) das previsões de teste.")
                
                st.subheader("Dados da Projeção Futura (Mensal)")
                df_futuro_m = res_m['future'][['period', 'y_hat']].rename(
                    columns={'period': 'Mês', 'y_hat': 'Venda Prevista (R$)'})
                st.dataframe(df_futuro_m.style.format({'Venda Prevista (R$)': 'R$ {:,.2f}'}), use_container_width=True)
        except ValueError as e:
            st.warning(f"Não foi possível gerar a previsão mensal: {e}")
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado ao gerar a previsão mensal: {e}")
            st.exception(e)

    with sub_tab_semanal:
        try:
            with st.expander("🤖 Previsão Semanal", expanded=True):
                res_w = get_ml_forecast(df4, df1, id_restaurante_selecionado, 'W')
                fig_w = plot_line(res_w,
                                  f'Semanal – Vendas Reais x Previstas ({restaurante_selecionado})',
                                  'W')
                st.pyplot(fig_w)

                st.subheader("Métricas do Modelo (Semanal)")
                col_w1, col_w2 = st.columns(2)
                col_w1.metric("R² (Score do Modelo)", f"{res_w['r2']:.3f}",
                              help="Quanto mais próximo de 1.0, melhor o modelo explica os dados passados.")
                col_w2.metric("MAE (Erro Médio Absoluto)", f"R$ {res_w['mae']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
                              help="O erro médio (para mais ou menos) das previsões de teste.")
                
                st.subheader("Dados da Projeção Futura (Semanal)")
                df_futuro_w = res_w['future'][['period', 'y_hat']].rename(
                    columns={'period': 'Semana (Fim)', 'y_hat': 'Venda Prevista (R$)'})
                st.dataframe(df_futuro_w.style.format({'Venda Prevista (R$)': 'R$ {:,.2f}'}), use_container_width=True)
        except ValueError as e:
            st.warning(f"Não foi possível gerar a previsão semanal: {e}")
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado ao gerar aG previsão semanal: {e}")
            st.exception(e)

# --- CONTEÚDO DA ABA 3: ESTRATÉGIA (IA) ---
with tab3:
    st.info("Esta análise usa o **histórico completo** do restaurante. Os filtros 'Canal de Venda' e 'Janela de Dias' da sidebar são ignorados aqui.")
    
    with st.expander("🧠 Estratégia Baseada na Projeção Mensal", expanded=True):
        try:
            res_m = get_ml_forecast(df4, df1, id_restaurante_selecionado, 'M')
            estrategia_texto_m = gerar_estrategia_ia(res_m, restaurante_selecionado)
            st.markdown(estrategia_texto_m)
        except Exception as e:
            st.warning(f"Não foi possível gerar a estratégia mensal: {e}")

    with st.expander("🧠 Estratégia Baseada na Projeção Semanal", expanded=True):
        try:
            res_w = get_ml_forecast(df4, df1, id_restaurante_selecionado, 'W')
            estrategia_texto_w = gerar_estrategia_ia(res_w, restaurante_selecionado)
            st.markdown(estrategia_texto_w)
        except Exception as e:
            st.warning(f"Não foi possível gerar a estrategia semanal: {e}")