 import io
import json
from datetime import date
import html
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="ERIMMT | Monitoramento Ambiental", page_icon="🌎", layout="wide")
COLORS = {"Área construída":"#ef946c","Vegetação":"#38c98b","Água":"#58b6f5","Solo exposto":"#e7c46a"}
LABELS = {0:"Área construída",1:"Vegetação",2:"Água",3:"Solo exposto",-1:"Ambíguo"}
FEATURES5 = ["B2","B3","B4","B8","B11"]
FEATURES8 = FEATURES5 + ["NDVI","NDWI","NDBI"]
CENTER = [-15.6461,-56.1326]
CLASS_LAYER = "Classificação MLP-8 V2"
ASSET_CLASS = "projects/erimmt/assets/classificacao_MLP8_V2_2026"
CLASS_STATS = pd.DataFrame([
    {"Classe":"Área construída", "Código":0, "Pixels":240860, "Área (ha)":9634.40, "Percentual":31.09},
    {"Classe":"Vegetação", "Código":1, "Pixels":483507, "Área (ha)":19340.28, "Percentual":62.42},
    {"Classe":"Água", "Código":2, "Pixels":12231, "Área (ha)":489.24, "Percentual":1.58},
    {"Classe":"Solo exposto", "Código":3, "Pixels":38002, "Área (ha)":1520.08, "Percentual":4.91},
])

st.markdown("""<style>
.block-container {padding-top:1.35rem;max-width:1500px}
h1,h2,h3 {letter-spacing:-.025em}

[data-testid="stMetric"] {
    background: #f0f7ff;
    border: 1px solid #c9dced;
    border-radius: 13px;
    padding: 14px;
}

[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #17324d !important;
}

.note {
    border-left: 4px solid #d39b27;
    background: #fff7e6;
    color: #263648 !important;
    padding: 13px 18px;
    border-radius: 5px;
}

.note * {
    color: #263648 !important;
}

</style>""", unsafe_allow_html=True)

def records_to_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")

@st.cache_data
def base_metrics():
    return pd.DataFrame([
        {"Versão":"V2","Modelo":"MLP-5","Acurácia":0.9461,"F1 macro":0.9492,"Kappa":0.9275,"Acurácia balanceada":0.9512},
        {"Versão":"V2","Modelo":"MLP-8","Acurácia":0.9502,"F1 macro":0.9532,"Kappa":0.9330,"Acurácia balanceada":0.9546}
    ])
def normalize_df(df):
    df = df.copy()
    aliases={"lat":"latitude","lon":"longitude","lng":"longitude","classe":"classe_manual"}
    for a,b in aliases.items():
        if a in df.columns and b not in df.columns:
            df=df.rename(columns={a:b})
    for c in ["latitude","longitude"]:
        if c in df:
            df[c]=pd.to_numeric(df[c], errors="coerce")
    return df

def ee_instance():
    try:
        import ee
        if "GEE_SERVICE_ACCOUNT_JSON" in st.secrets:
            raw = st.secrets["GEE_SERVICE_ACCOUNT_JSON"]
            from google.oauth2 import service_account
            info = json.loads(raw) if isinstance(raw,str) else dict(raw)
            credentials = service_account.Credentials.from_service_account_info(
                info,scopes=["https://www.googleapis.com/auth/earthengine",
                             "https://www.googleapis.com/auth/cloud-platform"])
            ee.Initialize(credentials=credentials,project=st.secrets.get("GEE_PROJECT","erimmt"))
        else:
            ee.Initialize(project=st.secrets.get("GEE_PROJECT","erimmt"))
        return ee,None
    except Exception as e:
        return None,str(e)

@st.cache_data(ttl=3600,show_spinner=False)
# ============================================================
# PÁGINA 2 — MAPAS
# ============================================================

elif pagina == "Explore os mapas":

    st.subheader(
        "Explore os mapas e as quatro classes"
    )

    st.write(
        "O mapa abre com a classificação MLP-8 V2 "
        "sobre a imagem de satélite. "
        "Não é necessário enviar nenhum arquivo."
    )

    col_controle, col_mapa = st.columns([1, 3])

    with col_controle:

        tipo = st.selectbox(
            "Camada principal",
            [
                "MLP-8 V2",
                "RGB",
                "NDVI",
                "NDWI",
                "NDBI"
            ],
            format_func=lambda k: {
                "MLP-8 V2": "Classificação ambiental",
                "RGB": "Imagem de satélite",
                "NDVI": "NDVI — vegetação",
                "NDWI": "NDWI — água",
                "NDBI": "NDBI — construções"
            }[k]
        )

        fundo = st.checkbox(
            "Imagem de satélite ao fundo",
            value=True
        )

        opacidade = st.slider(
            "Opacidade das cores (%)",
            20, 100, 78, 5
        ) / 100

        calor = st.checkbox(
            "Concentração espacial (opcional)",
            value=False,
            disabled=(tipo != "MLP-8 V2")
        )

        classe = (
            st.selectbox(
                "Classe para a concentração",
                list(CLASSES),
                format_func=lambda c: CLASSES[c],
                index=1
            )
            if calor else 1
        )

        raio = (
            st.select_slider(
                "Raio de análise",
                [200, 300, 500, 800],
                value=500
            )
            if calor else 500
        )

        with st.expander(
           Identifiquei a causa exata do erro no GitHub: **seu `app.py` tem apenas 97 linhas e termina em `def ee_tile(layer,start,end,cloud):`**. O restante do código não foi salvo. Por isso aparece `IndentationError`.

Preparei o arquivo completo e verifiquei a sintaxe Python.

<box border radius="Aqui está o **`app.py` completo e corrigido**, com 428 linhas. Verifiquei a sintaxe Python e o arquivo não apresentou erros de indentação.

<box border radius="lg" padding={3} gap={2}>
  <row align="center" gap={3}>
    <icon name="file-code" size="2xl" color="secondary"/>
    <box flex="1" gap={1}>
      **app.py — ERI-MT 2026**
      <text color="secondary" size="xs">Arquivo completo • Python • Pronto para substituir no GitHub</text>
    </box>
  </row>
  [<icon name="download" inline/> Baixar app.py completo e corrigido](sandbox:/mnt/data/app.py)
</box>

### Como corrigir o Streamlit

1. Baixe o arquivo acima.
2. Abra o <Link url="https://github.com/jennyk23/erimmt-monitoramento-ambiental/blob/main/app.py" title="app.py no GitHub"/>.
3. Clique no lápis de edição.
4. Apague **todo o código antigo** com `Ctrl + A` e `Delete`.
5. Abra o arquivo baixado, copie seu conteúdo inteiro e cole no GitHub.
6. Clique em **Commit changes**.

**Atenção:** seu arquivo atual no GitHub está incompleto, com apenas 97 linhas. O novo deve ter **428 linhas**. Verifique isso antes de salvar.

O código preserva os mapas e inclui a consulta de lugares e a auditoria manual. A implantação no Streamlit acontecerá após o GitHub receber o arquivo completo.
