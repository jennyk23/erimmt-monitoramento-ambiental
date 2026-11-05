            else:
                lat = lat_digitada if usar_manual else float(clicado["lat"])
                lon = lon_digitada if usar_manual else float(clicado["lng"])
                try:
                    codigo = classe_em_ponto(lat, lon)
                    if codigo is None:
                        st.warning("Este ponto está fora da cobertura válida ou não foi classificado.")
                    else:
                        st.success(f"Classe indicada pelo mapa MLP-8 V2: **{LABELS[codigo]}**")
                        st.markdown(
                            f'<span style="display:inline-block;width:18px;height:18px;'
                            f'background:{list(COLORS.values())[codigo]};'
                            f'border-radius:3px;margin-right:7px;"></span>'
                            f'<b>{LABELS[codigo]}</b>', unsafe_allow_html=True
                        )
                        st.caption(
                            "Resultado consultado no raster existente. "
                            "Não é uma nova previsão, nem uma classe confirmada em campo."
                        )
                except Exception as exc:
                    st.error(f"Consulta indisponível: {exc}")

    with tab_csv:
        st.write(
            "Você tem vários locais? Envie uma planilha CSV com duas colunas: "
            "**latitude** e **longitude**. A consulta aceita até 80 pontos por vez."
        )
        amostra = pd.DataFrame({
            "id_ponto": ["DEMO_1", "DEMO_2", "DEMO_3"],
            "latitude": [-15.6461, -15.6500, -15.6360],
            "longitude": [-56.1326, -56.1400, -56.1290],
        })
        st.download_button(
            "Baixar CSV de exemplo (coordenadas de demonstração)",
            records_to_csv(amostra), "exemplo_consulta_pontos.csv", "text/csv"
        )
        enviado = st.file_uploader("Escolher CSV de coordenadas", type="csv", key="consulta_csv")
        if enviado:
            try:
                tabela = normalize_df(pd.read_csv(enviado))
                if not {"latitude", "longitude"}.issubset(tabela):
                    st.error("Faltam as colunas latitude e longitude no arquivo.")
                elif len(tabela) == 0 or len(tabela) > 80:
                    st.error("Use uma planilha com 1 a 80 linhas.")
                elif tabela[["latitude", "longitude"]].isna().any().any():
                    st.error("Há coordenadas vazias ou inválidas na planilha.")
                elif not (
                    tabela["latitude"].between(-90, 90).all()
                    and tabela["longitude"].between(-180, 180).all()
                ):
                    st.error("Latitude deve estar entre -90 e 90; longitude, entre -180 e 180.")
                else:
                    st.dataframe(tabela.head(12), hide_index=True, use_container_width=True)
                    if st.button("Consultar pontos do CSV", type="primary"):
                        with st.spinner("Consultando as classes no Earth Engine..."):
                            st.session_state["resultado_lote"] = classes_de_pontos(tabela)
            except Exception as exc:
                st.error(f"Erro ao processar a planilha: {exc}")
        if "resultado_lote" in st.session_state:
            st.dataframe(st.session_state["resultado_lote"], hide_index=True, use_container_width=True)
            st.download_button(
                "Baixar resultados consultados", records_to_csv(st.session_state["resultado_lote"]),
                "resultado_consultas_mlp8_v2.csv", "text/csv"
            )

    with tab_info:
        st.markdown("""
**O que está sendo consultado?** Um mapa georreferenciado produzido
pela rede neural MLP-8 V2 com dados Sentinel-2 e os índices NDVI,
NDWI e NDBI. São quatro classes: construção, vegetação, água e solo exposto.

**O que significa o resultado?** É a classe atribuída pelo modelo ao pixel
selecionado, não uma identificação confirmada por visita ao local.

**Por que não pedimos arquivos `.keras` ou `.joblib`?** O público não precisa
instalar nem enviar modelos. Por segurança, a execução de modelos enviados
por visitantes está desativada; um arquivo `.joblib` desconhecido pode executar código.

**Uma consulta é uma nova inferência?** Não. Esta ferramenta lê o raster
MLP-8 V2 já calculado. Para testar outro modelo com novos dados espectrais
seria necessário disponibilizar previamente um modelo confiável no servidor.
""")


else:
    st.subheader("Como a pesquisa foi realizada")
    st.write(
        "Conheça as etapas do projeto, o significado dos números e o que "
        "a inteligência artificial pode — e não pode — concluir."
    )
    st.markdown(
        "**1. Observamos o território** com imagens Sentinel-2. "
        "**2. Calculamos índices ambientais** para água, vegetação e superfícies. "
        "**3. Treinamos e comparamos MLP-5 e MLP-8.** "
        "**4. Publicamos um mapa exploratório** com quatro classes."
    )
    with st.expander("Dicionário rápido: NDVI, NDWI, NDBI, MLP e pseudo-rótulo", expanded=True):
        st.markdown("""
- **NDVI:** índice que ajuda a caracterizar vegetação.
- **NDWI:** índice que pode ajudar a evidenciar água superficial.
- **NDBI:** índice relacionado a superfícies construídas, mas também sensível a solo exposto.
- **MLP-8:** rede neural que recebe cinco bandas Sentinel-2 e três índices.
- **Pseudo-rótulo:** classe atribuída inicialmente por regras espectrais; não é confirmação em campo.
- **Validação independente:** comparação com observações que não participaram do treinamento ou das regras de rotulagem.
""")
    st.markdown("### Documentação técnica para pesquisa")
    st.markdown("""
**Fonte:** notebook `Untitled15.ipynb`, fornecido pela autora via Google Colab.

**Recorte:** centro em Várzea Grande (−15,6461; −56,1326), raio 10 km, imagens Sentinel-2 SR Harmonized de 01/01/2026 a 16/06/2026, nuvens abaixo de 10%, bandas B2, B3, B4, B8 e B11, referência de 20 m.

**Índices:** NDVI (B8−B4)/(B8+B4); NDWI (B3−B8)/(B3+B8); NDBI (B11−B8)/(B11+B8).

**Classes:** área construída, vegetação, água e solo exposto.

**Aprendizado:** pseudo-rotulagem espectral, amostragem estratificada, divisão espacial treino/teste por blocos, modelos MLP-5 e MLP-8, comparação com Dynamic World, auditoria de divergências e revisão dos pseudo-rótulos V2.

**Mapa MLP-8 V2:** asset `projects/erimmt/assets/classificacao_MLP8_V2_2026`, resolução 20 m, área circular de raio 10 km, 774.600 pixels classificados e cobertura válida de 98,63%. O raster é fixo e não é recalculado pelos filtros de data.

**Limitação central:** treino e teste com pseudo-rótulos podem compartilhar o viés de geração desses rótulos. A avaliação manual independente é uma etapa distinta e permanece pendente no notebook disponibilizado.
""")
    st.markdown("**Arquivos do projeto:** notebook original incluído para rastreabilidade; dados CSV, modelos `.keras`, scalers `.joblib` e camadas exportadas devem ser transferidos separadamente do Google Drive.")
    st.info(
        "A pesquisa ainda precisa de validação manual independente dos pontos. "
        "As classes do mapa não são previsões de risco de inundação."
    )
