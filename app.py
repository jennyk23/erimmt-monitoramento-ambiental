
import json
import html
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st
import folium
from streamlit_folium import st_folium


# ============================================================
# ERI-MT 2026 - MONITORAMENTO AMBIENTAL
# ============================================================

st.set_page_config(
    page_title="ERI-MT 2026 | Monitoramento ambiental",
    page_icon="🌎",
    layout="wide"
)

CENTER = [-15.6461, -56.1326]

ASSET = (
    "projects/erimmt/assets/"
    "classificacao_MLP8_V2_2026"
)

CLASSES = {
    0: "Área construída",
    1: "Vegetação",
    2: "Água",
    3: "Solo exposto"
}

CORES = {
    0: "#ef946c",
    1: "#38c98b",
    2: "#58b6f5",
    3: "#e7c46a"
}

ATRIBUTOS_5 = ["B2", "B3", "B4", "B8", "B11"]

DADOS_AREAS = pd.DataFrame([
    {
        "Classe": "Área construída",
        "Pixels": 240860,
        "Área (ha)": 9634.40,
        "Percentual (%)": 31.09
    },
    {
        "Classe": "Vegetação",
        "Pixels": 483507,
        "Área (ha)": 19340.28,
        "Percentual (%)": 62.42
    },
    {
        "Classe": "Água",
        "Pixels": 12231,
        "Área (ha)": 489.24,
        "Percentual (%)": 1.58
    },
    {
        "Classe": "Solo exposto",
        "Pixels": 38002,
        "Área (ha)": 1520.08,
        "Percentual (%)": 4.91
    }
])


# ============================================================
# APARÊNCIA
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 1.25rem;
    }

    [data-testid="stMetric"] {
        background: #f0f7ff;
        border: 1px solid #c9dced;
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stMetric"] * {
        color: #17324d !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def csv_bytes(df):
    return df.to_csv(
        index=False
    ).encode("utf-8-sig")


def normalizar(df):
    df = df.copy()

    aliases = {
        "lat": "latitude",
        "lon": "longitude",
        "lng": "longitude"
    }

    for atual, desejado in aliases.items():
        if atual in df.columns and desejado not in df.columns:
            df = df.rename(
                columns={atual: desejado}
            )

    for coluna in ["latitude", "longitude"]:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(
                df[coluna],
                errors="coerce"
            )

    return df


def coordenadas_validas(df):
    return (
        not df.empty
        and {"latitude", "longitude"}.issubset(df.columns)
        and df["latitude"].between(-90, 90).all()
        and df["longitude"].between(-180, 180).all()
    )


# ============================================================
# AUTENTICAÇÃO GOOGLE EARTH ENGINE
# ============================================================

def earth_engine():
    import ee

    projeto = st.secrets.get(
        "GEE_PROJECT",
        "erimmt"
    )

    if "GEE_SERVICE_ACCOUNT_JSON" in st.secrets:

        from google.oauth2 import service_account

        bruto = st.secrets[
            "GEE_SERVICE_ACCOUNT_JSON"
        ]

        info = (
            json.loads(bruto)
            if isinstance(bruto, str)
            else dict(bruto)
        )

        credenciais = (
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
            credentials=credenciais,
            project=projeto
        )

    else:
        ee.Initialize(project=projeto)

    return ee


# ============================================================
# IMAGENS E CAMADAS DO EARTH ENGINE
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def imagem_tile(
    tipo,
    inicio="2026-01-01",
    fim="2026-06-16",
    nuvens=10
):

    ee = earth_engine()

    # Classificação fixa MLP-8 V2

    if tipo == "MLP-8 V2":

        img = ee.Image(ASSET).select(0)

        img = img.updateMask(
            img.gte(0).And(img.lte(3))
        )

        vis = {
            "min": 0,
            "max": 3,
            "palette": [
                "ef946c",
                "38c98b",
                "58b6f5",
                "e7c46a"
            ]
        }

        return img.getMapId(vis)[
            "tile_fetcher"
        ].url_format

    # Composição Sentinel-2

    ponto = ee.Geometry.Point([
        CENTER[1],
        CENTER[0]
    ])

    roi = ponto.buffer(10000)

    colecao = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterBounds(roi)
        .filterDate(inicio, fim)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                nuvens
            )
        )
    )

    if colecao.size().getInfo() == 0:
        raise ValueError(
            "Não há cenas Sentinel-2 para esse filtro."
        )

    def mascara(im):

        scl = im.select("SCL")

        valido = (
            scl.neq(0)
            .And(scl.neq(1))
            .And(scl.neq(3))
            .And(scl.neq(7))
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )

        return im.updateMask(valido)

    img = (
        colecao.map(mascara)
        .median()
        .clip(ponto.buffer(13000))
    )

    if tipo == "RGB":

        return (
            img.select(["B4", "B3", "B2"])
            .getMapId({
                "min": 0,
                "max": 3000,
                "gamma": 1.2
            })["tile_fetcher"]
            .url_format
        )

    # Índices espectrais

    reflectancia = (
        img.select(ATRIBUTOS_5)
        .divide(10000)
    )

    bandas = {
        "NDVI": ("B8", "B4"),
        "NDWI": ("B3", "B8"),
        "NDBI": ("B11", "B8")
    }

    if tipo not in bandas:
        raise ValueError(
            "Camada desconhecida."
        )

    indice = reflectancia.normalizedDifference(
        list(bandas[tipo])
    )

    if tipo == "NDVI":
        cores = [
            "9b4e24",
            "eee6be",
            "23865b"
        ]
    else:
        cores = [
            "a97e53",
            "f1eac3",
            "287dc0"
        ]

    return indice.getMapId({
        "min": -1,
        "max": 1,
        "palette": cores
    })["tile_fetcher"].url_format


# ============================================================
# CONCENTRAÇÃO ESPACIAL POR CLASSE
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def tile_densidade(codigo, raio):

    ee = earth_engine()

    img = ee.Image(ASSET).select(0)

    validos = (
        img.gte(0)
        .And(img.lte(3))
    )

    camada = (
        img.eq(int(codigo))
        .updateMask(validos)
    )

    media = (
        camada.focalMean(
            radius=int(raio),
            kernelType="circle",
            units="meters"
        )
        .updateMask(validos)
    )

    return media.getMapId({
        "min": 0,
        "max": 1,
        "palette": [
            "274bdb",
            "33c6ed",
            "56d77d",
            "ffdf55",
            "f04c3b"
        ]
    })["tile_fetcher"].url_format


# ============================================================
# CONSULTAS AO RASTER
# ============================================================

def consultar_ponto(latitude, longitude):

    ee = earth_engine()

    img = (
        ee.Image(ASSET)
        .select(0)
        .rename("classe")
        .unmask(255)
    )

    resultado = img.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=ee.Geometry.Point([
            float(longitude),
            float(latitude)
        ]),
        scale=20,
        maxPixels=10000
    ).getInfo()

    valor = resultado.get("classe")

    if valor is None:
        return None

    codigo = int(valor)

    return (
        codigo
        if codigo in CLASSES
        else None
    )


def consultar_lote(df):

    ee = earth_engine()

    imagem = (
        ee.Image(ASSET)
        .select(0)
        .rename("classe")
        .unmask(255)
    )

    itens = []

    for i, linha in enumerate(
        df.itertuples(index=False)
    ):

        feature = ee.Feature(
            ee.Geometry.Point([
                float(linha.longitude),
                float(linha.latitude)
            ]),
            {"linha": int(i)}
        )

        itens.append(feature)

    amostras = imagem.sampleRegions(
        collection=ee.FeatureCollection(itens),
        scale=20,
        geometries=False,
        tileScale=4
    ).getInfo()["features"]

    codigos = {}

    for item in amostras:
        props = item["properties"]

        codigos[int(props["linha"])] = (
            props.get("classe")
        )

    saida = df.copy().reset_index(drop=True)

    resultados = []

    for i in range(len(saida)):

        valor = codigos.get(i)

        if valor is None:
            resultados.append("Sem classificação")
        else:
            resultados.append(
                CLASSES.get(
                    int(valor),
                    "Sem classificação"
                )
            )

    saida["Classe MLP-8 V2"] = resultados

    return saida


# ============================================================
# CONSTRUÇÃO DOS MAPAS
# ============================================================

def mapa_base(centro=None, zoom=12):

    if centro is None:
        centro = CENTER

    return folium.Map(
        location=centro,
        zoom_start=zoom,
        tiles=None,
        control_scale=True
    )


def adicionar_tile(
    mapa,
    tipo,
    nome,
    opacidade=1.0,
    inicio="2026-01-01",
    fim="2026-06-16",
    nuvens=10
):

    try:

        url = imagem_tile(
            tipo,
            inicio,
            fim,
            nuvens
        )

        folium.TileLayer(
            tiles=url,
            attr="Google Earth Engine / Copernicus Sentinel-2",
            name=nome,
            overlay=True,
            control=True,
            opacity=opacidade
        ).add_to(mapa)

        return True

    except Exception as erro:

        st.warning(
            f"Camada {nome} indisponível: {erro}"
        )

        return False


def legenda_classe():

    elementos = []

    for codigo, nome in CLASSES.items():

        elementos.append(
            '<span style="display:inline-flex;'
            'align-items:center;'
            'margin:5px 18px 5px 0;'
            'gap:7px">'
            f'<span style="background:{CORES[codigo]};'
            'width:17px;height:17px;'
            'display:inline-block;'
            'border-radius:3px"></span>'
            f'{html.escape(nome)}'
            '</span>'
        )

    return (
        '<div style="background:#f0f7ff;'
        'padding:13px;border:1px solid #c9dced;'
        'border-radius:10px;color:#17324d">'
        '<b>Classes da inteligência artificial</b>'
        '<div>'
        + "".join(elementos)
        + '</div></div>'
    )


def legenda_indice(nome):

    if nome == "RGB":
        return (
            "**RGB:** imagem em cores próximas das "
            "naturais, formada pelas bandas B4, B3 e B2."
        )

    if nome == "NDVI":
        cores = "#9b4e24,#eee6be,#23865b"
    else:
        cores = "#a97e53,#f1eac3,#287dc0"

    textos = {
        "NDVI": (
            "Valores altos geralmente indicam "
            "maior presença de vegetação."
        ),
        "NDWI": (
            "Valores altos podem indicar "
            "água superficial."
        ),
        "NDBI": (
            "Valores altos podem estar associados "
            "a construções ou solo exposto."
        )
    }

    st.markdown(
        f'<div style="height:20px;'
        f'border-radius:5px;'
        f'background:linear-gradient(to right,{cores})">'
        f'</div>'
        '<div style="display:flex;'
        'justify-content:space-between">'
        '<small>−1</small>'
        '<small>0</small>'
        '<small>+1</small>'
        '</div>',
        unsafe_allow_html=True
    )

    return (
        f"**{nome}:** {textos[nome]} "
        "A escala é contínua e não equivale "
        "às quatro classes da MLP."
    )


# ============================================================
# MENU LATERAL
# ============================================================

with st.sidebar:

    st.header("ERI-MT 2026")

    pagina = st.radio(
        "O que você deseja fazer?",
        [
            "Início",
            "Explore os mapas",
            "Consulte um lugar",
            "Auditoria manual",
            "Modelos e resultados",
            "Metodologia"
        ]
    )

    st.caption(
        "Área-piloto: Várzea Grande/Cuiabá "
        "— raio de 10 km"
    )

    st.info(
        "Os resultados são exploratórios "
        "e não substituem validação de campo."
    )


# ============================================================
# CABEÇALHO
# ============================================================

st.title(
    "Monitoramento ambiental da Baixada Cuiabana"
)

st.caption(
    "Sentinel-2 • Google Earth Engine • "
    "Inteligência Artificial • ERI-MT 2026"
)


# ============================================================
# PÁGINA 1 - INÍCIO
# ============================================================

if pagina == "Início":

    st.write(
        "Este projeto identifica padrões exploratórios "
        "de cobertura do solo em uma área-piloto "
        "da Baixada Cuiabana. Você pode navegar "
        "pelos mapas e consultar a classe "
        "indicada para um lugar."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Pixels classificados",
        "774.600"
    )

    c2.metric(
        "Cobertura válida",
        "98,63%"
    )

    c3.metric(
        "Classes ambientais",
        "4"
    )

    c4.metric(
        "Pontos para auditoria",
        "200"
    )

    st.subheader(
        "Distribuição da classificação MLP-8 V2"
    )

    st.dataframe(
        DADOS_AREAS,
        hide_index=True,
        use_container_width=True
    )

    figura = px.pie(
        DADOS_AREAS,
        names="Classe",
        values="Pixels",
        color="Classe",
        color_discrete_map={
            CLASSES[i]: CORES[i]
            for i in CLASSES
        },
        hole=0.48
    )

    st.plotly_chart(
        figura,
        use_container_width=True
    )

    with st.expander(
        "Métricas da pesquisa e como interpretá-las"
    ):

        st.dataframe(
            pd.DataFrame([
                {
                    "Modelo": "MLP-5 V2",
                    "Acurácia interna": "94,61%",
                    "F1 macro": "0,9492"
                },
                {
                    "Modelo": "MLP-8 V2",
                    "Acurácia interna": "95,02%",
                    "F1 macro": "0,9532"
                }
            ]),
            hide_index=True,
            use_container_width=True
        )

        st.warning(
            "As métricas indicam concordância "
            "com pseudo-rótulos, não acurácia "
            "validada independentemente em campo."
        )

    st.success(
        "Para começar, escolha "
        "'Explore os mapas' no menu lateral."
    )


# ============================================================
# PÁGINA 2 - MAPAS
# ============================================================

elif pagina == "Explore os mapas":

    st.subheader(
        "Explore os mapas e as quatro classes"
    )

    st.write(
        "O mapa abre com a classificação MLP-8 V2 "
        "sobre a imagem do satélite Sentinel-2. "
        "Não é necessário enviar nenhum arquivo."
    )

    controles, visualizacao = st.columns([1, 3])

    with controles:

        opcoes = {
            "MLP-8 V2": "Classificação ambiental",
            "RGB": "Imagem de satélite",
            "NDVI": "NDVI — vegetação",
            "NDWI": "NDWI — água",
            "NDBI": "NDBI — construções"
        }

        tipo = st.selectbox(
            "O que deseja visualizar?",
            list(opcoes),
            format_func=lambda x: opcoes[x]
        )

        fundo = st.checkbox(
            "Imagem de satélite ao fundo",
            value=True
        )

        opacidade = st.slider(
            "Opacidade das cores (%)",
            20,
            100,
            78,
            5
        ) / 100

        calor = st.checkbox(
            "Concentração espacial (opcional)",
            value=False,
            disabled=(tipo != "MLP-8 V2")
        )

        if calor:

            classe = st.selectbox(
                "Classe para análise",
                list(CLASSES),
                index=1,
                format_func=lambda x: CLASSES[x]
            )

            raio = st.select_slider(
                "Raio de análise (metros)",
                [200, 300, 500, 800],
                value=500
            )

        else:
            classe = 1
            raio = 500

        with st.expander(
            "Filtros das imagens de satélite"
        ):

            inicio = st.date_input(
                "Data inicial",
                date(2026, 1, 1)
            )

            fim = st.date_input(
                "Data final (exclusiva)",
                date(2026, 6, 16)
            )

            nuvens = st.slider(
                "Máximo de nuvens (%)",
                0,
                80,
                10
            )

            st.caption(
                "Os filtros não recalculam "
                "a classificação MLP-8 V2."
            )

    mapa = mapa_base()

    if fim <= inicio:

        st.error(
            "A data final deve ser posterior "
            "à data inicial."
        )

    else:

        if fundo or tipo == "RGB":

            adicionar_tile(
                mapa,
                "RGB",
                "Sentinel-2 RGB",
                1.0,
                str(inicio),
                str(fim),
                nuvens
            )

        if tipo != "RGB":

            adicionar_tile(
                mapa,
                tipo,
                tipo,
                opacidade,
                str(inicio),
                str(fim),
                nuvens
            )

        if calor and tipo == "MLP-8 V2":

            try:

                url = tile_densidade(
                    classe,
                    raio
                )

                folium.TileLayer(
                    tiles=url,
                    attr="Google Earth Engine",
                    name=(
                        "Concentração: "
                        + CLASSES[classe]
                    ),
                    overlay=True,
                    opacity=0.75
                ).add_to(mapa)

            except Exception as erro:

                st.warning(
                    "Não foi possível exibir "
                    f"a concentração: {erro}"
                )

    folium.Circle(
        location=CENTER,
        radius=10000,
        color="#008f96",
        weight=2,
        fill=False,
        tooltip="Área de estudo: raio de 10 km"
    ).add_to(mapa)

    folium.LayerControl(
        collapsed=False
    ).add_to(mapa)

    with visualizacao:

        st_folium(
            mapa,
            height=700,
            use_container_width=True,
            returned_objects=[]
        )

    st.subheader(
        "O que significam as cores?"
    )

    if tipo == "MLP-8 V2":

        st.markdown(
            legenda_classe(),
            unsafe_allow_html=True
        )

        if calor:

            st.info(
                "Concentração espacial: azul indica "
                "menor proporção local da classe; "
                "amarelo e vermelho indicam maior "
                "proporção. Não representa "
                "temperatura ou risco ambiental."
            )

        st.dataframe(
            DADOS_AREAS,
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Baixar tabela em CSV",
            csv_bytes(DADOS_AREAS),
            "areas_mlp8_v2.csv",
            mime="text/csv"
        )

    else:

        st.info(
            legenda_indice(tipo)
        )

    st.caption(
        "A classificação é exploratória "
        "e ainda não possui validação "
        "manual independente concluída."
    )


# ============================================================
# PÁGINA 3 - CONSULTAR UM LUGAR
# ============================================================

elif pagina == "Consulte um lugar":

    st.subheader(
        "Descubra a classificação de um lugar"
    )

    st.write(
        "Clique no mapa e consulte a classe "
        "indicada pela inteligência artificial. "
        "Você não precisa conhecer programação."
    )

    aba_mapa, aba_lista, aba_ajuda = st.tabs([
        "Consultar no mapa",
        "Consultar vários pontos",
        "Como funciona?"
    ])

    with aba_mapa:

        mapa = mapa_base()

        adicionar_tile(
            mapa,
            "RGB",
            "Sentinel-2 RGB"
        )

        adicionar_tile(
            mapa,
            "MLP-8 V2",
            "Classificação MLP-8 V2",
            0.72
        )

        folium.Circle(
            location=CENTER,
            radius=10000,
            color="#008f96",
            fill=False
        ).add_to(mapa)

        folium.LayerControl().add_to(mapa)

        escolha = st_folium(
            mapa,
            height=520,
            use_container_width=True,
            returned_objects=["last_clicked"],
            key="consulta_click"
        )

        selecionado = (
            (escolha or {}).get("last_clicked")
        )

        if selecionado:

            st.caption(
                "Local selecionado: "
                f'{selecionado["lat"]:.6f}, '
                f'{selecionado["lng"]:.6f}'
            )

        with st.expander(
            "Prefiro digitar as coordenadas"
        ):

            lat_manual = st.number_input(
                "Latitude",
                value=-15.6461,
                format="%.6f"
            )

            lon_manual = st.number_input(
                "Longitude",
                value=-56.1326,
                format="%.6f"
            )

            usar_manual = st.checkbox(
                "Usar coordenadas digitadas"
            )

        if st.button(
            "Consultar a classe deste lugar",
            type="primary"
        ):

            if not usar_manual and not selecionado:

                st.warning(
                    "Clique primeiro no mapa "
                    "ou escolha coordenadas digitadas."
                )

            else:

                if usar_manual:
                    lat = lat_manual
                    lon = lon_manual
                else:
                    lat = selecionado["lat"]
                    lon = selecionado["lng"]

                try:

                    codigo = consultar_ponto(
                        lat,
                        lon
                    )

                    if codigo is None:

                        st.warning(
                            "Não há classificação válida "
                            "nesse ponto."
                        )

                    else:

                        st.success(
                            "A MLP-8 V2 indicou: "
                            f"**{CLASSES[codigo]}**"
                        )

                        st.markdown(
                            legenda_classe(),
                            unsafe_allow_html=True
                        )

                        st.caption(
                            "O resultado foi consultado "
                            "no raster existente. "
                            "Não representa uma nova "
                            "previsão nem confirmação "
                            "por observação de campo."
                        )

                except Exception as erro:

                    st.error(
                        f"Falha na consulta: {erro}"
                    )

    with aba_lista:

        st.write(
            "Você também pode consultar vários "
            "locais usando uma planilha CSV "
            "com latitude e longitude."
        )

        exemplo = pd.DataFrame({
            "id_ponto": [
                "EX1",
                "EX2",
                "EX3"
            ],
            "latitude": [
                -15.6461,
                -15.6500,
                -15.6360
            ],
            "longitude": [
                -56.1326,
                -56.1400,
                -56.1290
            ]
        })

        st.download_button(
            "Baixar CSV de exemplo",
            csv_bytes(exemplo),
            "coordenadas_exemplo.csv",
            mime="text/csv"
        )

        arquivo = st.file_uploader(
            "Enviar CSV com até 80 pontos",
            type="csv",
            key="csv_consulta"
        )

        if arquivo is not None:

            try:

                df = normalizar(
                    pd.read_csv(arquivo)
                )

                if (
                    not coordenadas_validas(df)
                    or len(df) > 80
                ):

                    st.error(
                        "Informe de 1 a 80 pontos "
                        "com latitude e longitude válidas."
                    )

                else:

                    st.dataframe(
                        df.head(10),
                        hide_index=True,
                        use_container_width=True
                    )

                    if st.button(
                        "Consultar lista de pontos"
                    ):

                        with st.spinner(
                            "Consultando classificação..."
                        ):

                            resultado = consultar_lote(
                                df
                            )

                            st.session_state[
                                "consulta_resultado"
                            ] = resultado

            except Exception as erro:

                st.error(
                    f"Não foi possível consultar: {erro}"
                )

        if "consulta_resultado" in st.session_state:

            resultado = st.session_state[
                "consulta_resultado"
            ]

            st.dataframe(
                resultado,
                hide_index=True,
                use_container_width=True
            )

            st.download_button(
                "Baixar resultado",
                csv_bytes(resultado),
                "resultado_mlp8.csv",
                mime="text/csv"
            )

    with aba_ajuda:

        st.info(
            "O sistema consulta a classificação "
            "MLP-8 V2 armazenada no Earth Engine. "
            "Não solicita arquivos .keras ou .joblib "
            "e não executa novos treinamentos."
        )


# ============================================================
# PÁGINA 4 - AUDITORIA MANUAL
# ============================================================

elif pagina == "Auditoria manual":

    st.subheader(
        "Aprenda a conferir a classificação"
    )

    st.write(
        "Observe a imagem de satélite de um ponto "
        "e registre o que você identifica. "
        "A classe da IA não aparece nesta etapa "
        "para não influenciar sua observação."
    )

    orientacoes = pd.DataFrame([
        {
            "Código": "0",
            "Classe": "Área construída",
            "O que observar": "Telhados e pavimentação"
        },
        {
            "Código": "1",
            "Classe": "Vegetação",
            "O que observar": "Árvores, matas e vegetação"
        },
        {
            "Código": "2",
            "Classe": "Água",
            "O que observar": "Rios, lagos e reservatórios"
        },
        {
            "Código": "3",
            "Classe": "Solo exposto",
            "O que observar": "Terra sem cobertura vegetal"
        },
        {
            "Código": "-1",
            "Classe": "Indeterminado",
            "O que observar": "Nuvem, sombra ou dúvida"
        }
    ])

    with st.expander(
        "Como reconhecer as classes?",
        expanded=True
    ):

        st.dataframe(
            orientacoes,
            hide_index=True,
            use_container_width=True
        )

    exemplo = pd.DataFrame({
        "id_ponto": ["EXEMPLO_1", "EXEMPLO_2"],
        "latitude": [-15.6461, -15.6500],
        "longitude": [-56.1326, -56.1400]
    })

    st.download_button(
        "Baixar modelo CSV para auditoria",
        csv_bytes(exemplo),
        "modelo_auditoria.csv",
        mime="text/csv"
    )

    arquivo = st.file_uploader(
        "Enviar CSV com pontos a auditar",
        type="csv",
        key="csv_auditoria"
    )

    demonstracao = st.checkbox(
        "Praticar com dois pontos ilustrativos",
        value=(arquivo is None)
    )

    if arquivo is None and not demonstracao:

        st.info(
            "Envie a base de pontos do Colab "
            "ou ative a demonstração."
        )

    else:

        try:

            import hashlib

            if arquivo is not None:

                dados_brutos = normalizar(
                    pd.read_csv(arquivo)
                )

                chave = hashlib.sha256(
                    arquivo.getvalue()
                ).hexdigest()

            else:

                dados_brutos = exemplo.copy()
                chave = "demo"

            if not coordenadas_validas(dados_brutos):

                st.error(
                    "O CSV precisa conter latitude "
                    "e longitude válidas."
                )

            else:

                if (
                    st.session_state.get(
                        "chave_auditoria"
                    ) != chave
                ):

                    dados_brutos = (
                        dados_brutos
                        .reset_index(drop=True)
                    )

                    if "id_ponto" not in dados_brutos:

                        dados_brutos.insert(
                            0,
                            "id_ponto",
                            [
                                f"P{i+1:03}"
                                for i in range(
                                    len(dados_brutos)
                                )
                            ]
                        )

                    for coluna in [
                        "classe_manual",
                        "confianca",
                        "observacoes",
                        "data_imagem"
                    ]:

                        if coluna not in dados_brutos:
                            dados_brutos[coluna] = ""

                    st.session_state[
                        "dados_auditoria"
                    ] = dados_brutos

                    st.session_state[
                        "chave_auditoria"
                    ] = chave

                df = st.session_state[
                    "dados_auditoria"
                ]

                indice = st.selectbox(
                    "1. Escolha um ponto",
                    range(len(df)),
                    format_func=lambda i: (
                        f'{i+1} — '
                        f'{df.iloc[i]["id_ponto"]}'
                    )
                )

                ponto = df.iloc[indice]

                lat = float(ponto.latitude)
                lon = float(ponto.longitude)

                col_mapa, col_form = st.columns(
                    [3, 2]
                )

                with col_mapa:

                    m = mapa_base(
                        [lat, lon],
                        zoom=16
                    )

                    adicionar_tile(
                        m,
                        "RGB",
                        "Sentinel-2 RGB"
                    )

                    folium.Marker(
                        [lat, lon],
                        tooltip="Ponto em avaliação"
                    ).add_to(m)

                    st_folium(
                        m,
                        height=450,
                        use_container_width=True,
                        returned_objects=[],
                        key=(
                            f"auditoria_{chave}_{indice}"
                        )
                    )

                    st.caption(
                        "Imagem composta do período "
                        "de estudo, não necessariamente "
                        "do dia atual."
                    )

                with col_form:

                    st.markdown(
                        "**2. Registre sua observação**"
                    )

                    rotulos = {
                        "": "Ainda não anotado",
                        "0": "Área construída",
                        "1": "Vegetação",
                        "2": "Água",
                        "3": "Solo exposto",
                        "-1": "Indeterminado"
                    }

                    valor_atual = str(
                        ponto.get(
                            "classe_manual",
                            ""
                        )
                    )

                    conf_atual = str(
                        ponto.get(
                            "confianca",
                            ""
                        )
                    )

                    if valor_atual not in rotulos:
                        valor_atual = ""

                    opcoes_conf = [
                        "",
                        "Alta",
                        "Média",
                        "Baixa"
                    ]

                    if conf_atual not in opcoes_conf:
                        conf_atual = ""

                    with st.form(
                        f"form_auditoria_{chave}_{indice}"
                    ):

                        rotulo = st.selectbox(
                            "Classe observada",
                            list(rotulos),
                            index=list(rotulos).index(
                                valor_atual
                            ),
                            format_func=lambda x: (
                                rotulos[x]
                            )
                        )

                        confianca = st.selectbox(
                            "Confiança",
                            opcoes_conf,
                            index=opcoes_conf.index(
                                conf_atual
                            )
                        )

                        comentario = st.text_area(
                            "Observações",
                            value=str(
                                ponto.get(
                                    "observacoes",
                                    ""
                                )
                            )
                        )

                        data_img = st.text_input(
                            "Data da imagem (se conhecida)",
                            value=str(
                                ponto.get(
                                    "data_imagem",
                                    ""
                                )
                            )
                        )

                        salvar = st.form_submit_button(
                            "Salvar anotação",
                            type="primary"
                        )

                        if salvar:

                            if not rotulo:

                                st.warning(
                                    "Selecione uma classe."
                                )

                            else:

                                df.at[
                                    indice,
                                    "classe_manual"
                                ] = rotulo

                                df.at[
                                    indice,
                                    "confianca"
                                ] = confianca

                                df.at[
                                    indice,
                                    "observacoes"
                                ] = comentario

                                df.at[
                                    indice,
                                    "data_imagem"
                                ] = data_img

                                st.session_state[
                                    "dados_auditoria"
                                ] = df

                                st.success(
                                    "Anotação salva "
                                    "nesta sessão."
                                )

                concluidos = int(
                    df["classe_manual"]
                    .astype(str)
                    .isin([
                        "0",
                        "1",
                        "2",
                        "3",
                        "-1"
                    ])
                    .sum()
                )

                st.progress(
                    concluidos / len(df),
                    text=(
                        f"{concluidos} de "
                        f"{len(df)} pontos anotados"
                    )
                )

                st.download_button(
                    "Baixar minhas anotações",
                    csv_bytes(df),
                    "anotacoes_manuais.csv",
                    mime="text/csv",
                    type="primary"
                )

                st.warning(
                    "As anotações não são armazenadas "
                    "permanentemente pelo site. "
                    "Baixe o CSV antes de sair. "
                    "Os pontos de demonstração "
                    "não são validação científica."
                )

        except Exception as erro:

            st.error(
                "Falha ao carregar os pontos: "
                f"{erro}"
            )


# ============================================================
# PÁGINA 5 - MODELOS E RESULTADOS
# ============================================================

elif pagina == "Modelos e resultados":

    st.subheader(
        "Entenda os modelos MLP-5 e MLP-8"
    )

    st.write(
        "Os modelos foram desenvolvidos para "
        "identificar quatro classes de cobertura "
        "do solo. A versão publicada no mapa "
        "é a MLP-8 V2."
    )

    medidas = pd.DataFrame([
        {
            "Modelo": "MLP-5 V2",
            "Variáveis": 5,
            "Concordância interna (%)": 94.61,
            "F1 macro": 0.9492,
            "Kappa": 0.9275
        },
        {
            "Modelo": "MLP-8 V2",
            "Variáveis": 8,
            "Concordância interna (%)": 95.02,
            "F1 macro": 0.9532,
            "Kappa": 0.9330
        }
    ])

    st.dataframe(
        medidas,
        hide_index=True,
        use_container_width=True
    )


    # Gráfico comparativo MLP-5 V2 x MLP-8 V2
    fig = px.bar(
        medidas,
        x="Concordância interna (%)",
        y="Modelo",
        orientation="h",
        text="Concordância interna (%)",
        color="Modelo",
        color_discrete_map={
            "MLP-5 V2": "#2477BD",
            "MLP-8 V2": "#18A57B"
        }
    )

    fig.update_traces(
        texttemplate="%{x:.2f}%",
        textposition="outside",
        cliponaxis=False
    )

    fig.update_layout(
        title="Comparação dos modelos MLP",
        xaxis_title="Concordância interna (%)",
        yaxis_title="",
        xaxis_range=[0, 105],
        height=300,
        showlegend=False,
        margin=dict(l=10, r=70, t=55, b=30)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.warning(
        "As métricas medem concordância "
        "com pseudo-rótulos. Não representam "
        "validação independente em campo."
    )

    st.markdown(
        "**O que a consulta faz?** "
        "Ela lê a classe de um pixel do mapa "
        "MLP-8 V2 já calculado. "
        "Não treina uma nova rede neural."
    )

    st.markdown(
        "**Por que o visitante não precisa "
        "enviar modelos?** "
        "Os arquivos .keras e .joblib são "
        "artefatos técnicos de treinamento. "
        "Eles não são necessários para "
        "consultar o resultado publicado."
    )

    st.download_button(
        "Baixar resultados em CSV",
        csv_bytes(medidas),
        "metricas_mlp_v2.csv",
        mime="text/csv"
    )


# ============================================================
# PÁGINA 6 - METODOLOGIA
# ============================================================

else:

    st.subheader(
        "Como a pesquisa foi realizada"
    )

    st.markdown(
        """
### Área de estudo

Área-piloto de 10 km de raio, centrada
em Várzea Grande, Mato Grosso,
abrangendo parte de Cuiabá.

### Dados utilizados

- Sentinel-2 SR Harmonized.
- Bandas B2, B3, B4, B8 e B11.
- Índices NDVI, NDWI e NDBI.
- Resolução espacial de referência: 20 m.
- Período: janeiro a junho de 2026.

O SRTM foi usado separadamente para
análise topográfica exploratória e não
como entrada da rede MLP-8.

### Modelos desenvolvidos

**MLP-5:** utiliza cinco bandas espectrais.

**MLP-8:** utiliza cinco bandas espectrais
e três índices derivados.

### Classificação publicada

`projects/erimmt/assets/classificacao_MLP8_V2_2026`

### Resultados espaciais

- 774.600 pixels classificados.
- 785.395 pixels dentro do círculo.
- Cobertura válida de 98,63%.
- Quatro classes de cobertura do solo.

### Limitações científicas

As métricas internas foram obtidas
com pseudo-rótulos derivados de regras
espectrais e dados auxiliares.

Esses resultados não comprovam acurácia
temática independente em campo.

A comparação secundária da V1 com
Dynamic World também não representa
verdade-terreno independente.

Na V2, informações do Dynamic World
participaram da revisão dos rótulos,
portanto não podem ser reutilizadas
como avaliação independente.

A inspeção dos 200 pontos preparados
para auditoria manual ainda não
foi concluída.

O projeto não produz estimativas
validadas de risco de inundação,
temperatura ou vulnerabilidade social.
"""
    )

    st.info(
        "Os mapas são resultados exploratórios "
        "de pesquisa científica. "
        "Não devem fundamentar decisões "
        "operacionais de defesa civil "
        "sem validação apropriada."
    )
