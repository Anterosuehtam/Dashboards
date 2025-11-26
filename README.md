# 📊 Cannoli DataVision - Dashboard de Análises e Previsões de Vendas

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red)
![MySQL](https://img.shields.io/badge/Database-MySQL-orange)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)

> Um dashboard interativo full-stack para gestão de restaurantes, integrando análise de dados históricos, previsão de vendas com Machine Learning e gestão multi-usuário segura.

---

## 🖼️ Visão do Projeto

O **Cannoli DataVision** é uma solução de Business Intelligence (BI) desenvolvida para transformar dados brutos de pedidos e campanhas de marketing em insights acionáveis. Diferente de dashboards estáticos, este projeto implementa um fluxo completo de engenharia de dados:

1.  **Autenticação Segura:** Sistema de Login/Cadastro com hash de senhas.
2.  **Banco de Dados Real:** Conexão com MySQL para persistência de dados.
3.  **Machine Learning:** Algoritmos preditivos para estimar faturamento futuro.
4.  **Multi-Tenancy:** Painel Admin (visão global) e Painel Restaurante (visão isolada).

---

## 🛠️ Tecnologias Utilizadas

O projeto foi construído utilizando uma stack moderna de Data Science e Engenharia de Software:

### 🔹 Frontend & Aplicação
* **[Streamlit](https://streamlit.io/):** Framework principal para construção da interface web interativa.
* **[Plotly Express](https://plotly.com/python/):** Criação de gráficos dinâmicos e interativos para melhor experiência de UX.
* **CSS Customizado:** Estilização avançada para identidade visual profissional.

### 🔹 Backend & Banco de Dados
* **[MySQL](https://www.mysql.com/):** Banco de dados relacional para armazenar usuários, pedidos e campanhas.
* **[SQLAlchemy](https://www.sqlalchemy.org/):** ORM utilizado para gerenciar conexões e queries de forma segura e "Pythonica".
* **Pandas:** Manipulação, limpeza e transformação de dados (ETL).

### 🔹 Inteligência Artificial (Machine Learning)
* **[Scikit-Learn](https://scikit-learn.org/):** Utilizado para criar modelos de **Regressão Linear**.
    * *Objetivo:* Analisar a correlação entre campanhas de marketing enviadas e o volume de vendas para prever faturamento futuro (semanal e mensal).

### 🔹 Segurança
* **Hashlib (SHA-256):** Criptografia de senhas para garantir que credenciais nunca sejam armazenadas em texto simples.
* **Session State:** Gerenciamento de sessão para controle de acesso restrito (Admin vs. Usuário).

---

## 🚀 Funcionalidades Principais

* **🔐 Sistema de Login Seguro:**
    * Cadastro de novos restaurantes com validação de duplicidade.
    * Login com verificação de hash.
* **📈 Visão Geral (Dashboard):**
    * KPIs de Vendas, Tickets Médios e Total de Pedidos.
    * Gráficos de funil de vendas por canal (iFood, Site, WhatsApp).
* **🤖 Previsão de Vendas (AI):**
    * Projeção automática de vendas para as próximas 4 semanas.
    * Mensagens de estratégia geradas automaticamente com base na tendência (Alta/Baixa).
* **⚙️ Painel Admin:**
    * Capacidade de visualizar e filtrar dados de qualquer restaurante cadastrado.

---

## 📦 Como Rodar Localmente

Pré-requisitos: Python 3.9+ e MySQL instalado.

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/Anterosuehtam/Dashboards.git](https://github.com/Anterosuehtam/Dashboards.git)
    cd Dashboards
    ```

2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure o Banco de Dados:**
    * Crie um arquivo `.streamlit/secrets.toml` na raiz do projeto.
    * Adicione suas credenciais do MySQL:
    ```toml
    [db]
    user = "seu_usuario"
    password = "sua_senha"
    host = "localhost"
    port = 3306
    database = "cannollifoods"
    ```

4.  **Execute a aplicação:**
    ```bash
    streamlit run dashboard.py
    ```

---

## 👨‍💻 Autor

Desenvolvido por **Matheus Antero**.

* [LinkedIn](https://www.linkedin.com/in/matheus-antero-/)
* [GitHub](https://github.com/Anterosuehtam)

---
