
import json
import html
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import folium

from folium.plugins import MarkerCluster
from streamlit_folium import st_folium


# ==========================================================
# ERIMMT 2026 — MONITORAMENTO AMBIENTAL COM IA
# ==========================================================

st.set_page_config(
    page_title="ERIMMT | Monitoramento Ambiental",
    page_icon="🌎",
    layout="wide"
)

CENTER = [-15.6461, -56.1326]

CLASS_LAYER = "Classificação MLP-8 V2"

ASSET_CLASS = (
    "projects/erimmt/assets/"
    "classificacao_MLP8_V2_2026"
)

COLORS = {
    "Área construída": "#ef946c",
    "Vegetação": "#38c98b",
    "Água": "#58b6f5",
    "Solo exposto": "#e7c46a"
}

LABELS = {
    0: "Área construída",
    1: "Vegetação",
    2: "Água",
    3: "Solo exposto",
    -1: "Ambíguo"
}

FEATURES5 = ["B2", "B3", "B4", "B8", "B11"]

FEATURES8 = FEATURES5 + [
    "NDVI", "NDWI", "NDBI"
]

CLASS_STATS = pd.DataFrame([
    {
        "Classe": "Área construída",
        "Pixels": 240860,
        "Área (ha)": 9634.40,
        "Percentual": 31.09
    },
    {
        "Classe": "Vegetação",
        "Pixels": 483507,
        "Área (ha)": 19340.28,
        "Percentual": 62.42
    },
    {
        "Classe": "Água",
        "Pixels": 12231,
        "Área (ha)": 489.24,
        "Percentual": 1.58
    },
    {
        "Classe": "Solo exposto",
        "Pixels": 38002,
        "Área (ha)": 1520.08,
        "Percentual": 4.91
    }
])


# ==========================================================
# ESTILO VISUAL — ALTO CONTRASTE
# ==========================================================

st.markdown("""
<style>
.block-container {
    padding-top: 1.35rem;
    max-width: 1500px;
}

h1, h2, h3 {
    letter-spacing: -0.025em;
}

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
</style>
""", unsafe_allow_html=True)


# ==========================================================
# FUNÇÕES AUXILIARES
# ==========================================================

def records_to_csv(df):
    return df.to_csv(
        index=False
    ).encode("utf-8-sig")


@st.cache_data
def base_metrics():
    return pd.DataFrame([
        {
            "Versão": "V2",
            "Modelo": "MLP-5",
            "Acurácia": 0.9461,
            "F1 macro": 0.9492,
            "Kappa": 0.9275,
            "Acurácia balanceada": 0.9512
        },
        {
            "Versão": "V2",
            "Modelo": "MLP-8",
            "Acurácia": 0.9502,
            "F1 macro": 0.9532,
            "Kappa": 0.9330,
            "Acurácia balanceada": 0.9546
        }
    ])


def normalize_df(df):
    df = df.copy()

    aliases = {
        "lat": "latitude",
        "lon": "longitude",
        "lng": "longitude",
        "classe": "classe_manual"
    }

    for antigo, novo in aliases.items():
        if antigo in df.columns and novo not in df.columns:
            df = df.rename(columns={antigo: novo})

    for coluna in ["latitude", "longitude"]:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(
                df[coluna],
                errors="coerce"
            )

    return df


# ==========================================================
# GOOGLE EARTH ENGINE — AUTENTICAÇÃO
# ==========================================================

def ee_instance():
    try:
        import ee

        projeto = st.secrets.get(
            "GEE_PROJECT", "erimmt"
        )

        if "GEE_SERVICE_ACCOUNT_JSON" in st.secrets:

            raw = st.secrets[
                "GEE_SERVICE_ACCOUNT_JSON"
            ]

            from google.oauth2 import service_account

            info = (
                json.loads(raw)
                if isinstance(raw, str)
                else dict(raw)
            )

            credentials = (
                service_account.Credentials
                .from_service_account_info(
                    info,
                    scopes=[
                        "https://www.googleapis.com/auth/earthengine",
                        "https://www.googleapis.com/auth/cloud-platform"
                    ]
                )
            )

            ee.Initialize(
                credentials=credentials,
                project=projeto
            )

        else:
            ee.Initialize(project=projeto)

        return ee, None

    except Exception as exc:
        return None, str(exc)


# ==========================================================
# CAMADAS REAIS DO EARTH ENGINE
# ==========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def ee_tile(layer, start, end, cloud):

    ee, error = ee_instance()

    if ee is None:
        return None, error

    # CLASSIFICAÇÃO FIXA PRODUZIDA NO COLAB

    if layer == CLASS_LAYER:

        try:
            classification = ee.Image(
                ASSET_CLASS
            ).select(0)

            classification = classification.updateMask(
                classification.gte(0).And(
                    classification.lte(3)
                )
            )

            mapa = classification.getMapId({
                "min": 0,
                "max": 3,
                "palette": [
                    "ef946c",
                    "38c98b",
                    "58b6f5",
                    "e7c46a"
                ]
            })

            return (
                mapa["tile_fetcher"].url_format,
                "Classificação MLP-8 V2 carregada."
            )

        except Exception as exc:
            return None, str(exc)

    # SENTINEL-2 E ÍNDICES ESPECTRAIS

    point = ee.Geometry.Point([
        CENTER[1],
        CENTER[0]
    ])

    roi = point.buffer(10000)

    images = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                cloud
            )
        )
    )

    def prep(im):
        scl = im.select("SCL")

        valid = (
            scl.neq(0)
            .And(scl.neq(1))
            .And(scl.neq(3))
            .And(scl.neq(7))
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )

        return im.updateMask(valid)

    try:

        quantidade = images.size().getInfo()

        if quantidade == 0:
            return None, "Nenhuma imagem encontrada."

        imagem = (
            images.map(prep)
            .median()
            .clip(roi)
        )

        reflectancia = (
            imagem.select([
                "B2", "B3", "B4", "B8", "B11"
            ])
            .divide(10000)
        )

        ndvi = reflectancia.normalizedDifference(
            ["B8", "B4"]
        ).rename("NDVI")

        ndwi = reflectancia.normalizedDifference(
            ["B3", "B8"]
        ).rename("NDWI")

        ndbi = reflectancia.normalizedDifference(
            ["B11", "B8"]
        ).rename("NDBI")

        if layer == "Satélite RGB":

            vis = imagem.select([
                "B4", "B3", "B2"
            ])

            params = {
                "min": 0,
                "max": 3000,
                "gamma": 1.2
            }

        else:

            vis = {
                "NDVI": ndvi,
                "NDWI": ndwi,
                "NDBI": ndbi
            }[layer]

            if layer == "NDVI":

                params = {
                    "min": -1,
                    "max": 1,
                    "palette": [
                        "#9b4e24",
                        "#eee6be",
                        "#23865b"
                    ]
                }

            else:

                params = {
                    "min": -1,
                    "max": 1,
                    "palette": [
                        "#a97e53",
                        "#f1eac3",
                        "#287dc0"
                    ]
                }

        mapa = vis.getMapId(params)

        return (
            mapa["tile_fetcher"].url_format,
            f"{quantidade} cenas selecionadas."
        )

    except Exception as exc:
        return None, str(exc)


# ==========================================================
# LEGENDAS EXPLICATIVAS
# ==========================================================

def legenda_da_camada(camada):

    if camada == CLASS_LAYER:

        return {
            "titulo": "Classificação ambiental — MLP-8 V2",
            "tipo": "categorias",
            "itens": list(COLORS.items()),
            "descricao": (
                "As cores representam classes atribuídas "
                "pela inteligência artificial. "
                "Áreas transparentes não possuem classificação. "
                "Resultado exploratório, sem validação "
                "manual independente concluída."
            )
        }

    if camada == "Satélite RGB":

        return {
            "titulo": "Sentinel-2 — Cores naturais",
            "tipo": "categorias",
            "itens": [
                ("B4 — Vermelho", "#e86c62"),
                ("B3 — Verde", "#49ac75"),
                ("B2 — Azul", "#5aa5e9")
            ],
            "descricao": (
                "A imagem combina as bandas B4, B3 e B2 "
                "para representar a superfície em cores "
                "próximas das naturais. "
                "Esta legenda identifica canais espectrais, "
                "não classes ambientais."
            )
        }

    descricoes = {
        "NDVI": (
            "Valores positivos elevados geralmente "
            "indicam maior presença e vigor da vegetação. "
            "Valores próximos de zero podem representar "
            "solo exposto ou áreas construídas. "
            "Valores negativos podem representar água "
            "ou outras superfícies não vegetadas."
        ),
        "NDWI": (
            "Valores positivos elevados podem indicar "
            "água superficial. A interpretação depende "
            "das características espectrais e ambientais. "
            "O índice não substitui a validação dos "
            "corpos hídricos."
        ),
        "NDBI": (
            "Valores positivos elevados podem estar "
            "associados a superfícies construídas. "
            "Solo exposto também pode apresentar "
            "valores elevados, exigindo cautela."
        )
    }

    paleta_ndvi = [
        "#9b4e24",
        "#eee6be",
        "#23865b"
    ]

    paleta_outros = [
        "#a97e53",
        "#f1eac3",
        "#287dc0"
    ]

    return {
        "titulo": f"{camada} — Índice espectral",
        "tipo": "continuo",
        "itens": (
            paleta_ndvi
            if camada == "NDVI"
            else paleta_outros
        ),
        "descricao": descricoes.get(
            camada, "Índice espectral contínuo."
        )
    }


def painel_legenda(camada):

    legenda = legenda_da_camada(camada)

    titulo = html.escape(legenda["titulo"])
    descricao = html.escape(legenda["descricao"])

    if legenda["tipo"] == "continuo":

        gradiente = ", ".join(
            legenda["itens"]
        )

        corpo = (
            '<div style="height:22px;'
            'border-radius:6px;'
            f'background:linear-gradient(to right,{gradiente});'
            'border:1px solid #bcc8d3;"></div>'
            '<div style="display:flex;'
            'justify-content:space-between;'
            'font-size:13px;color:#17324d;'
            'margin-top:5px;">'
            '<span><b>−1</b> — baixo</span>'
            '<span><b>0</b> — intermediário</span>'
            '<span><b>+1</b> — alto</span>'
            '</div>'
        )

    else:

        corpo = (
            '<div style="display:flex;'
            'flex-wrap:wrap;gap:8px 20px;">'
        )

        for nome, cor in legenda["itens"]:

            corpo += (
                '<span style="display:inline-flex;'
                'align-items:center;gap:7px;'
                'font-size:14px;color:#17324d;">'
                '<span style="display:inline-block;'
                'width:18px;height:18px;'
                f'background:{cor};'
                'border:1px solid #8295a6;'
                'border-radius:3px;"></span>'
                f'{html.escape(nome)}</span>'
            )

        corpo += '</div>'

    return (
        '<div style="background:#f6faff;'
        'border:1px solid #c9dced;'
        'border-radius:12px;'
        'padding:15px 18px;'
        'color:#17324d;'
        'font-family:Arial,sans-serif;'
        'line-height:1.55;">'
        f'<div style="font-weight:700;'
        f'font-size:17px;'
        f'margin-bottom:9px;">{titulo}</div>'
        f'{corpo}'
        '<p style="margin:10px 0 0;'
        'font-size:14px;">'
        f'{descricao}</p>'
        '</div>'
    )


# ==========================================================
# MENU LATERAL
# ==========================================================

with st.sidebar:

    st.header("ERIMMT • 2026")

    st.caption(
        "Explorador de sensoriamento remoto "
        "e inteligência artificial"
    )

    secao = st.radio(
        "Navegação",
        [
            "Visão geral",
            "Mapa e índices",
            "Auditoria manual",
            "Modelos e previsões",
            "Dados e metodologia"
        ]
    )

    st.divider()

    st.caption(
        "Área: Várzea Grande, MT • raio 10 km"
    )

    st.caption(
        "Sentinel-2 SR Harmonized "
        "• Janeiro a junho de 2026 • 20 m"
    )

    st.warning(
        "Resultados com pseudo-rótulos "
        "não substituem validação de campo.",
        icon="⚠️"
    )


st.title("Monitoramento ambiental com IA")

st.caption(
    "ERIMMT 2026 | Interface web baseada "
    "no notebook de pesquisa"
)


# ==========================================================
# PÁGINA 1 — VISÃO GERAL
# ==========================================================

if secao == "Visão geral":

    st.subheader(
        "Resultados exploratórios — Versão V2"
    )

    met = base_metrics()

    best = met.loc[
        met["Modelo"] == "MLP-8"
    ].iloc[0]

    a, b, c, d = st.columns(4)

    a.metric(
        "Acurácia no teste V2",
        f'{best["Acurácia"]:.2%}'
    )

    b.metric(
        "F1 macro no teste V2",
        f'{best["F1 macro"]:.2%}'
    )

    c.metric(
        "Kappa V2",
        f'{best["Kappa"]:.3f}'
    )

    d.metric(
        "Pontos para anotação manual",
        "200"
    )

    st.markdown(
        '<div class="note">'
        '<b>Interpretação:</b> as métricas V2 '
        'medem concordância com pseudo-rótulos '
        'em teste espacial. Não representam '
        'validação independente em campo.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### Comparação de arquiteturas"
    )

    st.dataframe(
        met,
        hide_index=True,
        use_container_width=True
    )

    chart = met.melt(
        id_vars=["Modelo"],
        value_vars=[
            "Acurácia",
            "F1 macro",
            "Acurácia balanceada"
        ],
        var_name="Métrica",
        value_name="Valor"
    )

    fig = px.bar(
        chart,
        x="Métrica",
        y="Valor",
        color="Modelo",
        barmode="group",
        range_y=[0, 1],
        text_auto=".1%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Pseudo-rótulos por classe")

    a, b = st.columns(2)

    dist = pd.DataFrame({
        "Classe": list(COLORS),
        "V1": [
            230502, 449197, 10957, 15219
        ],
        "V2": [
            226563, 449197, 10957, 19158
        ]
    })

    dist_m = dist.melt(
        "Classe",
        var_name="Versão",
        value_name="Pixels"
    )

    fig2 = px.bar(
        dist_m,
        x="Classe",
        y="Pixels",
        color="Versão",
        barmode="group"
    )

    a.plotly_chart(
        fig2,
        use_container_width=True
    )

    b.info(
        "Na revisão V2, aproximadamente "
        "3.938 pixels foram redirecionados "
        "de área construída para solo exposto."
    )

    b.dataframe(
        dist,
        hide_index=True,
        use_container_width=True
    )

    st.markdown(
        "### Comparação externa — Dynamic World V1"
    )

    st.write(
        "Em 1.023 amostras, a concordância "
        "de MLP-8 com Dynamic World foi "
        "**49,07% de acurácia** e "
        "**0,5056 de F1 macro**. "
        "Dynamic World não é verdade-terreno."
    )


# ==========================================================
# PÁGINA 2 — MAPA E ÍNDICES
# ==========================================================

elif secao == "Mapa e índices":

    st.subheader("Mapa interativo")

    left, right = st.columns([1, 3])

    with left:

        mode = st.selectbox(
            "Camada",
            [
                "Satélite RGB",
                "NDVI",
                "NDWI",
                "NDBI",
                CLASS_LAYER
            ]
        )

        is_class = mode == CLASS_LAYER

        live = st.toggle(
            "Consultar Google Earth Engine",
            value=True
        )

        if is_class:

            st.info(
                "Este mapa MLP-8 V2 é fixo. "
                "Alterar datas não recalcula "
                "a classificação."
            )

            start = date(2026, 1, 1)
            end = date(2026, 6, 16)
            cloud = 10

        else:

            start = st.date_input(
                "Data inicial",
                date(2026, 1, 1)
            )

            end = st.date_input(
                "Data final (exclusiva)",
                date(2026, 6, 16)
            )

            cloud = st.slider(
                "Máximo de nuvens (%)",
                0, 80, 10
            )

        opacity = st.slider(
            "Opacidade da camada",
            0, 100, 85, 5
        ) / 100

        data = st.file_uploader(
            "Pontos em CSV (latitude, longitude)",
            type="csv",
            key="map_csv"
        )

        df = None

        if data:

            try:

                df = normalize_df(
                    pd.read_csv(data)
                )

                if not {
                    "latitude", "longitude"
                }.issubset(df.columns):

                    st.error(
                        "O CSV precisa de latitude "
                        "e longitude."
                    )

                    df = None

                else:

                    df = df.dropna(
                        subset=[
                            "latitude",
                            "longitude"
                        ]
                    )

                    st.caption(
                        f"{len(df)} pontos carregados"
                    )

            except Exception as exc:

                st.error(
                    f"CSV inválido: {exc}"
                )

    # CONSTRUÇÃO DO MAPA

    m = folium.Map(
        location=CENTER,
        zoom_start=11,
        tiles="OpenStreetMap",
        control_scale=True
    )

    folium.Circle(
        location=CENTER,
        radius=10000,
        color="#19a999",
        fill=False,
        tooltip="Área de estudo — 10 km"
    ).add_to(m)

    if live:

        if not is_class and end <= start:

            st.error(
                "A data final precisa ser "
                "posterior à inicial."
            )

        else:

            with st.spinner(
                "Consultando Google Earth Engine..."
            ):

                tile, msg = ee_tile(
                    mode,
                    str(start),
                    str(end),
                    cloud
                )

            if tile:

                folium.TileLayer(
                    tiles=tile,
                    attr=(
                        "Google Earth Engine / Copernicus"
                    ),
                    name=mode,
                    overlay=True,
                    control=True,
                    opacity=opacity
                ).add_to(m)

                st.success(msg)

            else:

                st.error(
                    "Camada indisponível: " + msg
                )

    if df is not None:

        grupo = MarkerCluster(
            name="Pontos enviados"
        )

        for _, r in df.head(2000).iterrows():

            titulo = html.escape(
                str(r.get("id_ponto", "Ponto"))
            )

            folium.Marker(
                [
                    float(r.latitude),
                    float(r.longitude)
                ],
                tooltip=titulo,
                popup=titulo
            ).add_to(grupo)

        grupo.add_to(m)

    folium.LayerControl().add_to(m)

    with right:

        st_folium(
            m,
            height=590,
            use_container_width=True,
            returned_objects=[]
        )

    # LEGENDA AUTOMÁTICA PARA TODAS AS CAMADAS

    st.markdown(
        "### Legenda e interpretação do mapa"
    )

    st.markdown(
        painel_legenda(mode),
        unsafe_allow_html=True
    )

    st.caption(
        "O contorno turquesa delimita a área "
        "de estudo de 10 km de raio. "
        "A opacidade pode alterar a aparência "
        "das cores pela mistura com o mapa-base."
    )

    if is_class:

        st.subheader(
            "Distribuição espacial — MLP-8 V2"
        )

        st.dataframe(
            CLASS_STATS,
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Baixar dados da classificação (CSV)",
            records_to_csv(CLASS_STATS),
            "erimmt_areas_mlp8_v2.csv",
            mime="text/csv"
        )

        st.caption(
            "Cobertura válida: 774.600 pixels "
            "de 785.395 no círculo (98,63%). "
            "Os percentuais são calculados "
            "somente sobre pixels classificados. "
            "A validação manual independente "
            "ainda está pendente."
        )

    else:

        st.caption(
            "Os índices NDVI, NDWI e NDBI "
            "apresentam valores espectrais "
            "contínuos. Não devem ser "
            "interpretados como classificações "
            "automáticas de uso e cobertura do solo."
        )


# ==========================================================
# PÁGINA 3 — AUDITORIA MANUAL
# ==========================================================

elif secao == "Auditoria manual":

    st.subheader(
        "Referência manual independente"
    )

    st.write(
        "Importe o CSV de referência manual "
        "produzido no Colab."
    )

    upload = st.file_uploader(
        "Arquivo CSV de anotação",
        type="csv",
        key="audit"
    )

    if upload:

        try:

            df = normalize_df(
                pd.read_csv(
                    upload,
                    dtype={"id_ponto": str}
                ).fillna("")
            )

            if not {
                "latitude", "longitude"
            }.issubset(df.columns):

                st.error(
                    "Faltam latitude e longitude."
                )

            else:

                if "id_ponto" not in df:

                    df.insert(
                        0,
                        "id_ponto",
                        [
                            f"REF_{i:03d}"
                            for i in range(
                                1, len(df) + 1
                            )
                        ]
                    )

                for coluna in [
                    "classe_manual",
                    "confianca",
                    "data_imagem",
                    "observacoes"
                ]:

                    if coluna not in df:
                        df[coluna] = ""

                if (
                    "audit_table" not in st.session_state
                    or st.session_state.get(
                        "audit_filename"
                    ) != upload.name
                ):

                    st.session_state[
                        "audit_table"
                    ] = df

                    st.session_state[
                        "audit_filename"
                    ] = upload.name

                edit = st.data_editor(
                    st.session_state["audit_table"],
                    num_rows="fixed",
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "classe_manual":
                            st.column_config.SelectboxColumn(
                                "Classe manual",
                                options=[
                                    "", "0", "1",
                                    "2", "3", "-1"
                                ]
                            ),
                        "confianca":
                            st.column_config.SelectboxColumn(
                                "Confiança",
                                options=[
                                    "", "Alta",
                                    "Média", "Baixa"
                                ]
                            ),
                        "observacoes":
                            st.column_config.TextColumn(
                                "Observações",
                                width="large"
                            )
                    },
                    key="annotation_editor"
                )

                valid = (
                    edit["classe_manual"]
                    .astype(str)
                    .isin([
                        "0", "1", "2", "3", "-1"
                    ])
                    .sum()
                )

                st.progress(
                    valid / max(1, len(edit)),
                    text=(
                        f"{valid}/{len(edit)} "
                        "pontos anotados"
                    )
                )

                st.download_button(
                    "Baixar anotações em CSV",
                    records_to_csv(edit),
                    "referencia_manual_anotada.csv",
                    "text/csv"
                )

                st.info(
                    "Salve periodicamente o CSV. "
                    "As alterações não são "
                    "gravadas automaticamente."
                )

        except Exception as exc:

            st.error(
                f"Erro de leitura: {exc}"
            )

    else:

        st.info(
            "O notebook informa que 200 pontos "
            "foram preparados para anotação. "
            "Importe o CSV para visualizar "
            "e preencher os dados."
        )


# ==========================================================
# PÁGINA 4 — MODELOS E PREVISÕES
# ==========================================================

elif secao == "Modelos e previsões":

    st.subheader(
        "Inferência com modelos treinados"
    )

    st.write(
        "Envie o CSV de bandas espectrais "
        "e os arquivos `.keras` e `.joblib` "
        "exportados pelo Colab."
    )

    model = st.selectbox(
        "Modelo",
        ["MLP-8", "MLP-5"]
    )

    csv = st.file_uploader(
        "CSV de bandas e índices",
        type="csv",
        key="pred_csv"
    )

    keras_file = st.file_uploader(
        f"Arquivo {model} (.keras)",
        type="keras",
        key="model_upload"
    )

    scaler_file = st.file_uploader(
        f"Arquivo {model} (_scaler.joblib)",
        type="joblib",
        key="scaler_upload"
    )

    st.caption(
        "Carregue apenas modelos e scalers "
        "de origem confiável. Arquivos joblib "
        "podem executar código."
    )

    if st.button(
        "Executar classificação",
        type="primary"
    ):

        if not all([
            csv, keras_file, scaler_file
        ]):

            st.error(
                "Envie os três arquivos."
            )

        else:

            try:

                import os
                import tempfile
                import joblib

                from tensorflow.keras.models import (
                    load_model
                )

                features = (
                    FEATURES8
                    if model == "MLP-8"
                    else FEATURES5
                )

                data = pd.read_csv(csv)

                missing = [
                    x for x in features
                    if x not in data
                ]

                if missing:

                    st.error(
                        "Faltam colunas: "
                        + ", ".join(missing)
                    )

                else:

                    with tempfile.TemporaryDirectory() as d:

                        mp = os.path.join(
                            d, "modelo.keras"
                        )

                        sp = os.path.join(
                            d, "escalador.joblib"
                        )

                        with open(mp, "wb") as f:
                            f.write(
                                keras_file.getvalue()
                            )

                        with open(sp, "wb") as f:
                            f.write(
                                scaler_file.getvalue()
                            )

                        neural = load_model(
                            mp,
                            compile=False
                        )

                        scaler = joblib.load(sp)

                        x = data[features].apply(
                            pd.to_numeric,
                            errors="coerce"
                        )

                        valid = x.notna().all(axis=1)

                        out = data.copy()

                        out["classe_prevista"] = ""
                        out["confianca_modelo"] = np.nan

                        if valid.any():

                            probs = neural.predict(
                                scaler.transform(
                                    x.loc[valid]
                                ).astype("float32"),
                                verbose=0
                            )

                            out.loc[
                                valid,
                                "classe_prevista"
                            ] = [
                                LABELS[int(v)]
                                for v in probs.argmax(
                                    axis=1
                                )
                            ]

                            out.loc[
                                valid,
                                "confianca_modelo"
                            ] = probs.max(axis=1)

                        st.success(
                            f"{valid.sum()} "
                            "linhas classificadas."
                        )

                        st.dataframe(
                            out.head(300),
                            use_container_width=True
                        )

                        st.download_button(
                            "Baixar previsões",
                            records_to_csv(out),
                            "previsoes.csv",
                            "text/csv"
                        )

            except ImportError:

                st.error(
                    "TensorFlow não está instalado "
                    "no servidor."
                )

            except Exception as exc:

                st.error(
                    f"Falha de inferência: {exc}"
                )

    st.caption(
        "A confiança softmax não é uma "
        "probabilidade calibrada de acerto. "
        "Esta ferramenta classifica linhas "
        "de CSV, não imagens inteiras."
    )


# ==========================================================
# PÁGINA 5 — DADOS E METODOLOGIA
# ==========================================================

else:

    st.subheader(
        "Dados e metodologia científica"
    )

    st.markdown("""
**Projeto:** ERIMMT 2026 — Monitoramento ambiental
na Baixada Cuiabana com sensoriamento remoto
e inteligência artificial.

**Região:** área de estudo centrada em
Várzea Grande, Mato Grosso, com raio de 10 km.

**Fonte de imagens:** Sentinel-2 SR Harmonized.

**Período de referência:** janeiro a junho de 2026.

**Resolução espacial:** 20 metros.

**Índices ambientais:**

- NDVI = (B8 − B4) / (B8 + B4)
- NDWI = (B3 − B8) / (B3 + B8)
- NDBI = (B11 − B8) / (B11 + B8)

**Modelo de IA:** rede neural MLP-8 V2,
com oito variáveis espectrais e quatro
classes ambientais.

**Asset da classificação:**

`projects/erimmt/assets/classificacao_MLP8_V2_2026`

**Resultados espaciais:**

- Pixels classificados: 774.600
- Cobertura válida: 98,63%
- Área classificada: 30.984 hectares
- Classes: área construída, vegetação,
  água e solo exposto

**Limitações metodológicas:**

Os resultados da MLP-8 V2 são exploratórios,
pois as métricas de treinamento e teste
foram obtidas com pseudo-rótulos.

A validação manual independente ainda
não foi concluída.

Os mapas dos índices espectrais são
representações contínuas e não equivalem
a classificações supervisionadas.

O raster de classificação é fixo e não
é atualizado automaticamente quando
o usuário modifica as datas.
""")

    st.info(
        "Para a apresentação científica, "
        "diferencie os resultados exploratórios "
        "da validação independente."
    )
