# 📊 Tech Challenge – Fase 4 | Dashboard de Previsão de Mercado

Este projeto corresponde à **Fase 4 do Tech Challenge (FIAP / POSTECH)** e tem
como objetivo realizar o **deploy de um dashboard de previsão de mercado**
utilizando **Streamlit** e **Machine Learning** (Random Forest) aplicado a
indicadores técnicos calculados a partir de dados históricos de preço.

---

## 🎯 Objetivo

Disponibilizar um **dashboard interativo** que permita ao usuário:

- Visualizar a série histórica de preços com indicadores técnicos relevantes.
- Consultar a **previsão de movimento** do mercado (ALTA ou BAIXA) para o próximo pregão.
- Acompanhar um **resumo executivo** da previsão, com nível de confiança e razões técnicas.
- Entender, via aba de **performance do modelo**, quais features mais influenciam a decisão.
- Monitorar indicadores técnicos atuais em formato de cards e tabelas.

---

## 🧠 Modelo Utilizado

- **Tipo de modelo:** `RandomForestClassifier` (scikit-learn).
- **Variável-alvo (`target`):** indicador binário se o retorno do próximo dia é positivo (1 = ALTA, 0 = BAIXA).
- **Features técnicas principais (9):**
  - `sma_5`, `sma_20`, `sma_50` – Médias móveis de curto, médio e longo prazo.
  - `rsi` – Relative Strength Index (14 períodos).
  - `macd`, `macd_signal` – MACD e linha de sinal.
  - `volatility` – Volatilidade dos retornos em janela de 20 dias.
  - `bb_upper`, `bb_lower` – Bandas de Bollinger superior e inferior (20 períodos).
- **Hiperparâmetros:**
  - `n_estimators=100` (100 árvores de decisão).
  - `max_depth=10` (profundidade máxima das árvores).
  - `random_state=42` (reprodutibilidade).
- **Pré-processamento:** padronização das features com `StandardScaler`.
- **Treino:** usa ~245 amostras do histórico, deixando as últimas 5 linhas reservadas.
- **Saída do modelo:**
  - Classe prevista: **ALTA** ou **BAIXA**.
  - Probabilidades por classe, convertidas em **confiança (%)** exibida no painel.

As métricas apresentadas no dashboard foram obtidas durante a validação
realizada na **Fase 2 do Tech Challenge**.

---

## 📊 Métricas e Indicadores Exibidos no Painel

O dashboard não mostra apenas a saída do modelo, mas uma visão analítica
completa, dividida em abas.

### Indicadores Técnicos Calculados

A partir da coluna `close` do arquivo `Unified_Data.csv`, são calculados:

- Médias móveis simples: `SMA 5`, `SMA 20`, `SMA 50`.
- `RSI (14 períodos)` com zonas de sobrecompra/sobrevenda.
- `MACD`, `MACD Signal` e `MACD Histogram`.
- Bandas de Bollinger: `bb_middle`, `bb_upper`, `bb_lower`.
- `return`: retorno percentual diário.
- `volatility`: desvio padrão do retorno em janela de 20 dias.

---

## 📁 Estrutura do Projeto

```text
.
├── app_vfinal.py                         # Aplicação principal Streamlit (dashboard final)
├── Unified_Data.csv                      # Base de dados histórica usada pelo painel
├── requirements.txt                      # Dependências do projeto
├── README.md                             # Este arquivo
├── Dashboard_Documentacao_Completa.md    # Documentação técnica e executiva do dashboard
└── Apresentacao_Executiva_5min.md        # Roteiro de apresentação executiva (5 minutos) - VIDEO!
```

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.13
- **Web App:** Streamlit
- **Manipulação de dados:** pandas, numpy
- **Machine Learning:** scikit-learn (RandomForestClassifier, StandardScaler)
- **Visualização:** Plotly (graph_objects, express)
- **Tratamento de erros:** traceback, warnings
- **Controle de versão:** Git / GitHub

---

## 🧩 Estrutura Lógica do Código

No arquivo `app_vfinal.py`, o fluxo principal segue:

1. **Carregamento de dados (`load_data`)**:
   - Lê o CSV `Unified_Data.csv`.
   - Converte a coluna `date` para datetime.
   - Ordena por data.
   - Trata erros de IO com mensagens descritivas.

2. **Criação de features (`create_features`)**:
   - Calcula indicadores técnicos (SMAs, RSI, MACD, Bandas de Bollinger, volatilidade, retorno, target).
   - Trata valores ausentes com forward/backward fill.
   - Retorna dataframe com 19 colunas de features.

3. **Treino do modelo (`train_model`)**:
   - Seleciona 9 features principais.
   - Padroniza com `StandardScaler`.
   - Treina um `RandomForestClassifier` com 100 árvores.
   - Retorna (`model`, `scaler`, `feature_cols`).

4. **Geração da previsão (`get_prediction_and_reasons`)**:
   - Usa a última linha do dataframe.
   - Faz `transform` das features com o scaler.
   - Usa `predict` e `predict_proba` para gerar classe (ALTA/BAIXA) e confiança.
   - Monta lista de razões com base em regras de negócio dos indicadores.

5. **Renderização do dashboard**:
   - **Topo:** card com previsão, confiança e última cotação.
   - **Abas:**
     - `Análise Técnica`: gráficos de série histórica, RSI, MACD.
     - `Indicadores Atuais`: métricas em cards + tabela dos últimos 10 dias.
     - `Performance`: métricas do modelo + gráfico de importância das features.
     - `Resumo`: narrativa executiva da previsão, razões e dados da última linha.

---

## 🌐 Deploy

O deploy da aplicação foi realizado utilizando o **Streamlit Cloud**, com
integração direta ao repositório do GitHub.

---

## 👨‍🎓 Projeto Acadêmico

Projeto desenvolvido para fins acadêmicos no curso **POSTECH – FIAP**,
como parte do **Tech Challenge – Fase 4**.

### Grupo 3

- Desenvolvedores:
  - Jarbas Ten Caten (jtcaten@bb.com.br)
  - Paulo Sérgio Xavier Santos (paulosxs@bb.com.br)

- Link do app no Streamlit: https://techchallengefase4grupo3-2pnp7dtlwuameybvkdybny.streamlit.app/
- Link do vídeo de apresentação no Youtube: https://www.youtube.com/watch?v=pPLdJ6tAB4Y

---

**Data:** Janeiro de 2026  
**Status:** ✅ Pronto para Produção

