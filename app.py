import os
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# APP STREAMLIT - PREVISÃO PONTUAL COM MODELO RFC
# VERSÃO CORRIGIDA PARA INPUT MANUAL
# ============================================================
# Requisitos:
#   pip install streamlit pandas scikit-learn joblib openpyxl
#
# Como correr:
#   streamlit run app_streamlit_rfc_manual_corrigida.py
#
# Espera encontrar o modelo guardado como:
#   modelo_rfc.joblib
#
# IMPORTANTE:
# Esta versão NÃO usa pd.get_dummies(drop_first=True) diretamente numa só linha,
# porque isso faz desaparecer as colunas dummy e pode gerar sempre a mesma previsão.
# ============================================================


st.set_page_config(
    page_title="Previsão Vendas RFC",
    page_icon="🌲",
    layout="centered"
)


@st.cache_resource
def carregar_modelo(caminho_modelo: str):
    return joblib.load(caminho_modelo)


def preparar_dados_para_previsao_manual(df: pd.DataFrame, modelo_export: dict) -> pd.DataFrame:
    """
    Prepara uma linha manual para o modelo.

    Correção principal:
    - Não usamos pd.get_dummies(drop_first=True) numa única linha.
    - Criamos diretamente todas as colunas finais do treino com 0.
    - Preenchemos as colunas numéricas.
    - Para cada categórica, ativamos manualmente a dummy correta, se existir.

    Isto replica o resultado final que o modelo viu no treino.
    """

    df = df.copy()

    feature_columns = modelo_export["feature_columns"]
    colunas_categoricas = modelo_export.get("colunas_categoricas", [])
    colunas_para_remover = modelo_export.get(
        "colunas_para_remover",
        ["Variavel_Alvo", "Quantidade Vendida", "Data", "Produto"]
    )
    colunas_historico = modelo_export.get(
        "colunas_historico",
        [
            "Dias_Desde_Ultima_Venda",
            "Dias_Desde_Ultima_Compra",
            "Dias_Desde_Ultima_Prod",
            "Cluster_Produto",
        ]
    )
    colunas_quantidades = modelo_export.get(
        "colunas_quantidades",
        [
            "Qt_Comprada",
            "Qt_Produzida",
            "Qtd_Vendida_Ultima_Venda",
            "Media_Qtd_Ultimas_3_Vendas",
        ]
    )

    # Tratamento de nulos igual ao notebook
    for col in colunas_historico:
        if col in df.columns:
            df[col] = df[col].fillna(-1)

    for col in colunas_quantidades:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Remover colunas que não entram no modelo
    df_features = df.drop(columns=colunas_para_remover, errors="ignore")

    # Criar DataFrame final com todas as colunas do treino a 0
    X_final = pd.DataFrame(0, index=df_features.index, columns=feature_columns)

    # 1) Copiar colunas numéricas / não categóricas que existam diretamente no treino
    for col in df_features.columns:
        if col in colunas_categoricas:
            continue

        if col in X_final.columns:
            X_final[col] = pd.to_numeric(df_features[col], errors="coerce").fillna(0)

    # 2) Codificar categóricas manualmente
    # No treino foi usado pd.get_dummies(..., drop_first=True).
    # Portanto, se a dummy não existir, significa que a categoria é provavelmente
    # a categoria base/dropada, ficando tudo a 0 para essa variável.
    for col in colunas_categoricas:
        if col not in df_features.columns:
            continue

        valor = str(df_features.loc[df_features.index[0], col])
        dummy_col = f"{col}_{valor}"

        if dummy_col in X_final.columns:
            X_final.loc[df_features.index[0], dummy_col] = 1

    return X_final


def converter_para_numero(valor):
    try:
        if float(valor).is_integer():
            return int(valor)
    except Exception:
        pass
    return valor


# ============================================================
# INTERFACE
# ============================================================

st.title("🌲 Previsão Vendas com o Random Forest Classifier")
st.write("Preencha os campos abaixo para gerar uma previsão pontual.")

CAMINHO_MODELO = "modelo_rfc.joblib"

if not os.path.exists(CAMINHO_MODELO):
    st.error("Não encontrei o ficheiro `modelo_rfc.joblib` na pasta da app.")
    st.info("Coloca o `modelo_rfc.joblib` na mesma pasta deste ficheiro Python.")
    st.stop()

try:
    modelo_export = carregar_modelo(CAMINHO_MODELO)
except Exception as erro:
    st.error("Erro ao carregar o modelo `modelo_rfc.joblib`.")
    st.exception(erro)
    st.stop()

rfc = modelo_export["model"]
scaler = modelo_export.get("scaler")
feature_columns = modelo_export["feature_columns"]
campos_entrada = modelo_export.get("campos_entrada", feature_columns)
colunas_categoricas = modelo_export.get("colunas_categoricas", [])
valores_categoricos = modelo_export.get("valores_categoricos", {})

with st.expander("Informação do modelo", expanded=False):
    st.write(f"**Modelo:** `{type(rfc).__name__}`")
    st.write(f"**Número de variáveis após pré-processamento:** `{len(feature_columns)}`")
    st.write(f"**Scaler guardado:** `{'Sim' if scaler is not None else 'Não'}`")

st.subheader("Dados para previsão")

with st.form("formulario_previsao"):
    dados_input = {}

    for col in campos_entrada:
        if col in colunas_categoricas:

            # Tratamento visual específico para Cluster_Produto
            if col == "Cluster_Produto":

                opcoes_originais = valores_categoricos.get(col, [])

                # Esconder o valor -1
                opcoes_validas = [
                    valor for valor in opcoes_originais
                    if str(valor).strip() not in ["-1", "-1.0"]
                ]

                # Garantir que aparecem pelo menos as opções 0 e 1
                if len(opcoes_validas) == 0:
                    opcoes_validas = ["0", "1"]

                mapa_visual_cluster = {
                    "0": "Baixa Frequência",
                    "0.0": "Baixa Frequência",
                    "1": "Alta Frequência",
                    "1.0": "Alta Frequência"
                }

                dados_input[col] = st.selectbox(
                    label="Frequência de Venda",
                    options=opcoes_validas,
                    format_func=lambda valor: mapa_visual_cluster.get(
                        str(valor).strip(),
                        str(valor)
                    )
                )

                # Restantes colunas categóricas
            else:
                opcoes = valores_categoricos.get(col, [])

                if len(opcoes) > 0:
                    dados_input[col] = st.selectbox(col, opcoes)
                else:
                    dados_input[col] = st.text_input(col, value="")
        else:
            dados_input[col] = st.number_input(col, value=0.0, step=1.0)

    botao_prever = st.form_submit_button("Gerar previsão")

if botao_prever:
    dados_input = {col: converter_para_numero(valor) for col, valor in dados_input.items()}
    df_novo = pd.DataFrame([dados_input])

    st.divider()
    st.subheader("Dados introduzidos")
    st.dataframe(df_novo, use_container_width=True)

    try:
        X_novo = preparar_dados_para_previsao_manual(df_novo, modelo_export)

        # Aplicar StandardScaler se o modelo foi treinado com scaler
        if scaler is not None:
            X_novo_modelo = scaler.transform(X_novo)
        else:
            X_novo_modelo = X_novo

        previsao = rfc.predict(X_novo_modelo)[0]

        # Ir buscar o valor da classe FE1
        classe_fe1 = str(dados_input.get("FE1", "")).strip().upper()

        # Regra dos dias
        if classe_fe1.startswith("PA"):
            dias_previsao = 30
        else:
            dias_previsao = 60

        st.subheader("Resultado")
        if previsao == 1:
            st.success(f"Previsão RFC: Classe 1 - Vende nos próximos {dias_previsao} Dias")
        else:
            st.error(f"Previsão RFC: Classe 0 - Não vende nos proximos {dias_previsao} Dias")

        if hasattr(rfc, "predict_proba"):
            probabilidades = rfc.predict_proba(X_novo_modelo)[0]
            df_probs = pd.DataFrame({
                "Classe": rfc.classes_,
                "Probabilidade": probabilidades
            })
            df_probs["Probabilidade"] = df_probs["Probabilidade"].map(lambda x: f"{x:.2%}")

            st.subheader("Probabilidades")
            st.dataframe(df_probs, use_container_width=True)

        with st.expander("Ver dados após pré-processamento", expanded=False):
            st.write("Colunas finais enviadas para o modelo antes do scaler:")
            st.dataframe(X_novo, use_container_width=True)

            # Mostrar apenas as colunas diferentes de zero para facilitar debug
            st.write("Colunas com valor diferente de zero:")
            colunas_ativas = X_novo.loc[:, (X_novo != 0).any(axis=0)]
            st.dataframe(colunas_ativas, use_container_width=True)

    except Exception as erro:
        st.error("Ocorreu um erro ao gerar a previsão.")
        st.exception(erro)
