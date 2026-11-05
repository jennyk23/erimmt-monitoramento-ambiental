# ERIMMT 2026 — Dashboard ambiental

Aplicação Streamlit baseada no notebook `Untitled15.ipynb` (incluído), sem modificar o notebook original.

## Executar localmente

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Abra o endereço local exibido pelo terminal (normalmente http://localhost:8501).

## Publicar no Streamlit Community Cloud

1. Envie `app.py`, `requirements.txt` e a pasta `.streamlit` para um repositório GitHub (não envie secrets reais).
2. Acesse https://share.streamlit.io e crie um aplicativo apontando para `app.py`.
3. Configure os secrets na plataforma para habilitar Earth Engine; sem credenciais, o dashboard e as demais telas seguem disponíveis.
4. Anexe os CSV e os arquivos de modelos somente onde houver autorização de uso e divulgação.

## Configurar Earth Engine

O notebook usa o projeto `erimmt` do Google Cloud. Para obter camadas Sentinel-2 em ambiente hospedado, configure uma **conta de serviço autorizada no Earth Engine** e com permissão de uso do projeto Cloud. Nos secrets do Streamlit:

```toml
GEE_PROJECT = "erimmt"
GEE_SERVICE_ACCOUNT_JSON = '''{"type":"service_account","project_id":"...","private_key_id":"...","private_key":"...","client_email":"...","client_id":"...","token_uri":"https://oauth2.googleapis.com/token"}'''
```

**Nunca publique** a chave privada no GitHub nem no notebook. Se não houver conta de serviço, execute localmente com credenciais Google Cloud Application Default Credentials apropriadas.

## Inferência com redes neurais

Instale `tensorflow` adicionalmente num ambiente compatível, se desejar usar a página de previsões:

```bash
pip install tensorflow
```

Faça upload dos arquivos `MLP-5.keras`/`MLP-5_scaler.joblib` ou `MLP-8.keras`/`MLP-8_scaler.joblib` (ou versões `*_V2.keras` e `*_V2_scaler.joblib`). O aplicativo pede os arquivos por upload e usa somente os dados enviados. **Nunca carregue arquivos joblib de terceiros desconhecidos.**

## O que é real e o que depende dos dados

- Métricas V2 e comparação Dynamic World: valores **extraídos das saídas do notebook**, não recalculados no site.
- Mapa: base cartográfica e círculo de estudo; a imagem Sentinel-2 só é exibida após autenticação Earth Engine.
- Pontos: são mostrados somente se o visitante importar CSV real.
- Auditoria: a tabela é editável e exportável, mas não fornece rótulos independentes automaticamente.
- Inferência: efetiva sobre CSV com todas as bandas, quando o usuário fornece o modelo, scaler e TensorFlow.
- Publicação: não realizada automaticamente. Esta entrega contém o código pronto para publicar.

## Ética de pesquisa

Resultados baseados em pseudo-rótulos não devem ser descritos como validação independente em campo; documentar fonte, datas, versões e limitações nos trabalhos científicos.
