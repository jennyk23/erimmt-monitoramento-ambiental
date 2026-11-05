import io
import json
from datetime import date
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
def ee_tile(layer,start,end,cloud):
    ee,error=ee_instance()
    if ee is None:
        return None,error
    point=ee.Geometry.Point([CENTER[1],CENTER[0]])
    roi=point.buffer(10000)
    images=(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi).filterDate(start,end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",cloud)))
    def prep(im):
        scl=im.select("SCL")
        valid=(scl.neq(0).And(scl.neq(1)).And(scl.neq(3))
               .And(scl.neq(7)).And(scl.neq(8)).And(scl.neq(9))
               .And(scl.neq(10)).And(scl.neq(11)))
        return im.updateMask(valid)
    try:
        n=images.size().getInfo()
        if n==0:return None,"Nenhuma cena encontrada para esse filtro."
        im=images.map(prep).median().clip(roi)
        refl=im.select(["B2","B3","B4","B8","B11"]).divide(10000)
        ndvi=refl.normalizedDifference(["B8","B4"]).rename("NDVI")
        ndwi=refl.normalizedDifference(["B3","B8"]).rename("NDWI")
        ndbi=refl.normalizedDifference(["B11","B8"]).rename("NDBI")
        if layer=="Satélite RGB":
            vis=im.select(["B4","B3","B2"]); params={"min":0,"max":3000,"gamma":1.2}
        else:
            vis={"NDVI":ndvi,"NDWI":ndwi,"NDBI":ndbi}[layer]
            params={"min":-1,"max":1,"palette":["#9b4e24","#eee6be","#23865b"]} if layer=="NDVI" else {"min":-1,"max":1,"palette":["#a97e53","#f1eac3","#287dc0"]}
        m=vis.getMapId(params)
        return m["tile_fetcher"].url_format,f"{n} cenas no intervalo selecionado"
    except Exception as e:
        return None,f"Falha na consulta Earth Engine: {e}"

with st.sidebar:
    st.header("ERIMMT • 2026")
    st.caption("Explorador de sensoriamento remoto e inteligência artificial")
    secao=st.radio("Navegação",["Visão geral","Mapa e índices","Auditoria manual","Modelos e previsões","Dados e metodologia"])
    st.divider()
    st.caption("Área: Várzea Grande, MT • raio 10 km")
    st.caption("Sentinel-2 SR Harmonized • 01/01 a 16/06/2026 • 20 m")
    st.warning("Resultados com pseudo-rótulos não substituem validação de campo.",icon="⚠️")

st.title("Monitoramento ambiental com IA")
st.caption("ERIMMT 2026 | Interface web baseada no notebook de pesquisa • versão demonstrativa funcional")

if secao=="Visão geral":
    st.subheader("Resultados exploratórios — versão V2")
    met=base_metrics()
    best=met.loc[met["Modelo"]=="MLP-8"].iloc[0]
    a,b,c,d=st.columns(4)
    a.metric("Acurácia no teste V2",f'{best["Acurácia"]:.2%}')
    b.metric("F1 macro no teste V2",f'{best["F1 macro"]:.2%}')
    c.metric("Kappa V2",f'{best["Kappa"]:.3f}')
    d.metric("Pontos para anotação manual","200")
    st.markdown('<div class="note"><b>Interpretação:</b> as métricas V2 medem concordância com pseudo-rótulos em teste espacial; não demonstram 95% de acurácia em verdade-terreno. A referência manual independente foi preparada, mas não está avaliada neste notebook.</div>',unsafe_allow_html=True)
    st.markdown("### Comparação de arquiteturas")
    st.dataframe(met,hide_index=True,use_container_width=True)
    chart=met.melt(id_vars=["Modelo"],value_vars=["Acurácia","F1 macro","Acurácia balanceada"],var_name="Métrica",value_name="Valor")
    fig=px.bar(chart,x="Métrica",y="Valor",color="Modelo",barmode="group",range_y=[0,1],text_auto=".1%")
    st.plotly_chart(fig,use_container_width=True)
    st.subheader("Pseudo-rótulos por classe")
    a,b=st.columns(2)
    dist=pd.DataFrame({"Classe":list(COLORS),"V1":[230502,449197,10957,15219],"V2":[226563,449197,10957,19158]})
    dist_m=dist.melt("Classe",var_name="Versão",value_name="Pixels")
    fig2=px.bar(dist_m,x="Classe",y="Pixels",color="Versão",barmode="group")
    a.plotly_chart(fig2,use_container_width=True)
    b.info("Na revisão V2, aproximadamente 3.938 pixels foram redirecionados de área construída para solo exposto. Isso é uma correção de pseudo-rótulos, não uma mudança ambiental temporal.")
    b.dataframe(dist,hide_index=True,use_container_width=True)
    st.markdown("### Comparação externa — Dynamic World (V1)")
    st.write("Em 1.023 amostras, a concordância de MLP-8 com Dynamic World foi **49,07% de acurácia** e **0,5056 de F1 macro**. Dynamic World é uma referência remota, não uma verdade-terreno independente.")

elif secao=="Mapa e índices":
    st.subheader("Mapa interativo")
    left,right=st.columns([1,3])
    with left:
        mode=st.selectbox("Camada",["Satélite RGB","NDVI","NDWI","NDBI"])
        live=st.toggle("Consultar Google Earth Engine",value=False,help="Exige credenciais autorizadas no servidor.")
        start=st.date_input("Data inicial",date(2026,1,1))
        end=st.date_input("Data final (exclusiva)",date(2026,6,16))
        cloud=st.slider("Máximo de nuvens (%)",0,80,10)
        data=st.file_uploader("Pontos em CSV (latitude, longitude)",type="csv",key="map_csv")
        df=None
        if data:
            try:
                df=normalize_df(pd.read_csv(data))
                if not {"latitude","longitude"}.issubset(df.columns):
                    st.error("O arquivo precisa ter latitude e longitude.")
                    df=None
                else:
                    df=df.dropna(subset=["latitude","longitude"])
                    st.caption(f"{len(df)} coordenadas carregadas")
            except Exception as exc:st.error(f"CSV inválido: {exc}")
    m=folium.Map(location=CENTER,zoom_start=11,tiles="OpenStreetMap",control_scale=True)
    folium.Circle(location=CENTER,radius=10000,color="#19a999",fill=False,tooltip="Área de estudo: 10 km").add_to(m)
    if live:
        if end<=start:st.error("A data final precisa ser posterior à inicial.")
        else:
            with st.spinner("Consultando imagens no Earth Engine..."):
                tile,msg=ee_tile(mode,str(start),str(end),cloud)
            if tile:
                folium.TileLayer(tiles=tile,attr="Google Earth Engine / Copernicus",name=mode,overlay=True,control=True,opacity=.78).add_to(m)
                st.success(msg)
            else:st.info("Camada remota indisponível. "+msg)
    if df is not None:
        marker_group=MarkerCluster(name="Pontos enviados")
        for _,r in df.head(2000).iterrows():
            title=str(r.get("id_ponto","Ponto"))
            folium.Marker([float(r.latitude),float(r.longitude)],tooltip=title,popup=title).add_to(marker_group)
        marker_group.add_to(m)
    folium.LayerControl().add_to(m)
    with right:
        st_folium(m,height=590,use_container_width=True,returned_objects=[])
    st.caption("O mapa-base requer internet. As camadas reais do Sentinel-2 só aparecem quando o Google Earth Engine está autenticado; nenhuma classificação foi simulada.")

elif secao=="Auditoria manual":
    st.subheader("Referência manual independente")
    st.write("Importe o arquivo `referencia_manual_para_anotacao.csv` criado no Colab. O aplicativo não revela os pseudo-rótulos ao anotador.")
    upload=st.file_uploader("CSV de anotação",type="csv",key="audit")
    if upload:
        try:
            df=normalize_df(pd.read_csv(upload,dtype={"id_ponto":str}).fillna(""))
            required={"latitude","longitude"}
            if not required.issubset(df):st.error("Faltam as colunas latitude e longitude.")
            else:
                if "id_ponto" not in df: df.insert(0,"id_ponto",[f"REF_{i:03d}" for i in range(1,len(df)+1)])
                for c in ["classe_manual","confianca","data_imagem","observacoes"]:
                    if c not in df:df[c]=""
                if "audit_table" not in st.session_state or st.session_state.get("audit_filename")!=upload.name:
                    st.session_state["audit_table"]=df
                    st.session_state["audit_filename"]=upload.name
                edit=st.data_editor(st.session_state["audit_table"],num_rows="fixed",hide_index=True,use_container_width=True,
                    column_config={"classe_manual":st.column_config.SelectboxColumn("Classe manual",options=["","0","1","2","3","-1"]),
                        "confianca":st.column_config.SelectboxColumn("Confiança",options=["","Alta","Média","Baixa"]),
                        "observacoes":st.column_config.TextColumn("Observações",width="large")},key="annotation_editor")
                valid=edit["classe_manual"].astype(str).isin(["0","1","2","3","-1"]).sum()
                st.progress(valid/max(1,len(edit)),text=f"{valid}/{len(edit)} pontos anotados")
                st.download_button("Baixar anotações em CSV",records_to_csv(edit),"referencia_manual_anotada.csv","text/csv",type="primary")
                st.info("Salve o CSV periodicamente. As edições não são gravadas automaticamente no Google Drive.")
        except Exception as exc:st.error(f"Erro de leitura: {exc}")
    else:
        st.info("O notebook informa que 200 pontos foram preparados (50 por estrato). O CSV real deve ser importado: seus dados não estão embutidos no arquivo .ipynb.")

elif secao=="Modelos e previsões":
    st.subheader("Inferência com modelos treinados")
    st.write("Para classificar dados novos, envie um CSV com bandas espectrais e os dois arquivos exportados pelo Colab (`.keras` e `_scaler.joblib`).")
    model=st.selectbox("Modelo",["MLP-8","MLP-5"])
    csv=st.file_uploader("CSV de bandas e índices",type="csv",key="pred_csv")
    keras_file=st.file_uploader(f"Arquivo {model} (.keras)",type="keras",key="model_upload")
    scaler_file=st.file_uploader(f"Arquivo {model} (_scaler.joblib)",type="joblib",key="scaler_upload")
    st.caption("Importe apenas modelos e scalers produzidos por você ou por fonte confiável. Arquivos joblib podem executar código ao serem carregados.")
    if st.button("Executar classificação",type="primary"):
        if not all([csv,keras_file,scaler_file]):st.error("Envie os três arquivos para executar inferência real.")
        else:
            try:
                import tempfile,os,joblib
                from tensorflow.keras.models import load_model
                features=FEATURES8 if model=="MLP-8" else FEATURES5
                data=pd.read_csv(csv)
                missing=[x for x in features if x not in data]
                if missing:st.error("Faltam colunas: "+", ".join(missing))
                else:
                    with tempfile.TemporaryDirectory() as d:
                        mp=os.path.join(d,"modelo.keras"); sp=os.path.join(d,"escalador.joblib")
                        open(mp,"wb").write(keras_file.getvalue());open(sp,"wb").write(scaler_file.getvalue())
                        neural=load_model(mp,compile=False);scaler=joblib.load(sp)
                        x=data[features].apply(pd.to_numeric,errors="coerce")
                        valid=x.notna().all(axis=1)
                        out=data.copy();out["classe_prevista"]="";out["confianca_modelo"]=np.nan
                        if valid.any():
                            probs=neural.predict(scaler.transform(x.loc[valid]).astype("float32"),verbose=0)
                            out.loc[valid,"classe_prevista"]=[LABELS[int(v)] for v in probs.argmax(axis=1)]
                            out.loc[valid,"confianca_modelo"]=probs.max(axis=1)
                        st.success(f"{valid.sum()} linhas classificadas")
                        st.dataframe(out.head(300),use_container_width=True)
                        st.download_button("Baixar previsões",records_to_csv(out),"previsoes.csv","text/csv")
            except ImportError:st.error("TensorFlow não está instalado. Veja README para habilitar inferência.")
            except Exception as exc:st.error(f"Falha de inferência: {exc}")
    st.markdown("**Importante:** a confiança softmax não é uma probabilidade calibrada de acerto. Esse fluxo classifica linhas de CSV, não imagens inteiras de satélite.")

else:
    st.subheader("Método e transparência")
    st.markdown("""
**Fonte:** notebook `Untitled15.ipynb`, fornecido pela autora via Google Colab.

**Recorte:** centro em Várzea Grande (−15,6461; −56,1326), raio 10 km, imagens Sentinel-2 SR Harmonized de 01/01/2026 a 16/06/2026, nuvens abaixo de 10%, bandas B2, B3, B4, B8 e B11, referência de 20 m.

**Índices:** NDVI (B8−B4)/(B8+B4); NDWI (B3−B8)/(B3+B8); NDBI (B11−B8)/(B11+B8).

**Classes:** área construída, vegetação, água e solo exposto.

**Aprendizado:** pseudo-rotulagem espectral, amostragem estratificada, divisão espacial treino/teste por blocos, modelos MLP-5 e MLP-8, comparação com Dynamic World, auditoria de divergências e revisão dos pseudo-rótulos V2.

**Limitação central:** treino e teste com pseudo-rótulos podem compartilhar o viés de geração desses rótulos. A avaliação manual independente é uma etapa distinta e permanece pendente no notebook disponibilizado.
""")
    st.markdown("**Arquivos do projeto:** notebook original incluído para rastreabilidade; dados CSV, modelos `.keras`, scalers `.joblib` e camadas exportadas devem ser transferidos separadamente do Google Drive.")
