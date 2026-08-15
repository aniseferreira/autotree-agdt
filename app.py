import gc
import os
import json
import zipfile
import urllib.request
import requests
import streamlit as st
import pandas as pd

# --- 1. INTERCEPTAÇÃO ESPECÍFICA DO SERVIDOR FORA DO AR (UOregon -> Hugging Face) ---
_original_requests_get = requests.get

def patched_requests_get(url, *args, **kwargs):
    # Intercepta SOMENTE requisições destinadas ao servidor da UOregon
    if isinstance(url, str) and "nlp.uoregon.edu" in url:
        if url.endswith(".zip"):
            filename = os.path.basename(url)
            url = f"https://huggingface.co/uonlp/trankit/resolve/main/models/v1.0.0/xlm-roberta-base/{filename}"
        elif "available_langs.json" in url:
            url = "https://huggingface.co/uonlp/trankit/raw/main/available_langs.json"
        elif "version.json" in url:
            url = "https://huggingface.co/uonlp/trankit/raw/main/version.json"
            
    return _original_requests_get(url, *args, **kwargs)

requests.get = patched_requests_get

# Importações do Trankit e Transformers após o patch
from transformers import AutoTokenizer, AutoModel
import trankit
import trankit.utils.tbinfo as tbinfo

# Ajusta constante interna do Trankit
tbinfo.URL = "https://huggingface.co/uonlp/trankit/resolve/main/models/v1.0.0/"

# --- 2. CONFIGURAÇÃO STREAMLIT ---

st.set_page_config(
    page_title="Anotador AGDT (Trankit)",
    page_icon="🏛️",
    layout="wide"
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".trankit_cache")

def prepare_trankit_environment(treebank_model: str, embedding_type: str = "xlm-roberta-base"):
    """
    Garante que o modelo de embeddings (xlm-roberta-base) e o modelo específico
    de árvore de dependências do Trankit estejam pré-carregados e no cache.
    """
    # 1. Pré-carrega explicitamente os pesos do XLM-RoBERTa no cache do Hugging Face
    with st.spinner("Garantindo disponibilidade do XLM-RoBERTa..."):
        try:
            AutoTokenizer.from_pretrained(embedding_type)
            AutoModel.from_pretrained(embedding_type)
        except Exception as e:
            st.warning(f"Aviso ao verificar XLM-RoBERTa: {e}")

    # 2. Prepara diretórios do Trankit
    target_dir = os.path.join(CACHE_DIR, embedding_type)
    os.makedirs(target_dir, exist_ok=True)
    
    extracted_model_dir = os.path.join(target_dir, treebank_model)
    zip_path = os.path.join(target_dir, f"{treebank_model}.zip")
    
    # 3. Download e extração do arquivo .zip do modelo do Trankit
    if not os.path.exists(extracted_model_dir):
        if not os.path.exists(zip_path):
            st.info(f"Baixando modelo {treebank_model} do Hugging Face...")
            hf_url = f"https://huggingface.co/uonlp/trankit/resolve/main/models/v1.0.0/{embedding_type}/{treebank_model}.zip"
            
            with st.spinner("Baixando arquivo do modelo (~350 MB)..."):
                urllib.request.urlretrieve(hf_url, zip_path)
            st.success("Download do modelo concluído!")

        st.info("Descompactando modelo...")
        with st.spinner("Extraindo pesos na pasta de cache..."):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        st.success("Modelo descompactado!")

    # 4. Registra os metadados exigidos pelo Trankit
    version_file = os.path.join(CACHE_DIR, "version.json")
    if not os.path.exists(version_file):
        with open(version_file, "w", encoding="utf-8") as f:
            json.dump({"v1.0.0": "available"}, f)

    save_info_path = os.path.join(CACHE_DIR, "downloaded_langs.json")
    info = {}
    if os.path.exists(save_info_path):
        try:
            with open(save_info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception:
            info = {}
            
    info[treebank_model] = embedding_type
    with open(save_info_path, "w", encoding="utf-8") as f:
        json.dump(info, f)

@st.cache_resource(show_spinner="Carregando modelo linguístico Trankit na memória...")
def load_trankit_pipeline(treebank_model: str) -> trankit.Pipeline:
    gc.collect()
    embedding_type = "xlm-roberta-base"
    
    prepare_trankit_environment(treebank_model, embedding_type=embedding_type)
    
    pipeline = trankit.Pipeline(
        lang=treebank_model,
        embedding=embedding_type,
        gpu=False,
        cache_dir=CACHE_DIR
    )
    return pipeline

def trankit_to_agdt_dataframe(doc: dict) -> pd.DataFrame:
    rows = []
    for sent_idx, sent in enumerate(doc.get("sentences", []), start=1):
        for token in sent.get("tokens", []):
            rows.append({
                "sent_id": sent_idx,
                "id": token.get("id"),
                "form": token.get("text"),
                "lemma": token.get("lemma", "_"),
                "upostag": token.get("upos", "_"),
                "xpostag": token.get("xpos", "_"),
                "feats": token.get("feats", "_"),
                "head": token.get("head", 0),
                "deprel": token.get("deprel", "_")
            })
    return pd.DataFrame(rows)

# --- 3. EXECUÇÃO DA APLICAÇÃO ---

st.title("🏛️ Anotador AGDT - Treebank de Dependências")
st.markdown(
    "Ferramenta de anotação automática para **Grego Antigo** usando o backend "
    "**Trankit** alinhado ao padrão AGDT/Perseids."
)

st.sidebar.header("Configurações")
model_choice = st.sidebar.selectbox(
    "Modelo de Treebank Grego:",
    ["ancient-greek-perseus", "ancient-greek-proiel"],
    help="O modelo 'ancient-greek-perseus' segue o padrão AGDT/Perseids."
)

default_text = "Μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος"
input_text = st.text_area(
    "Texto em Grego Antigo:",
    value=default_text,
    height=150
)

if st.button("Analisar Dependências", type="primary"):
    if input_text.strip():
        try:
            nlp = load_trankit_pipeline(model_choice)
            
            with st.spinner("Processando morfossintaxe e parse de dependências..."):
                doc = nlp(input_text)
                df_agdt = trankit_to_agdt_dataframe(doc)
            
            st.success("Análise concluída com sucesso!")
            
            st.subheader("Tabela de Anotação (Padrão CoNLL-U / AGDT)")
            st.dataframe(df_agdt, use_container_width=True)
            
            tsv_data = df_agdt.to_csv(sep="\t", index=False)
            st.download_button(
                label="📥 Baixar Anotações (TSV / CoNLL-U)",
                data=tsv_data,
                file_name=f"{model_choice}_parsed.tsv",
                mime="text/tab-separated-values"
            )

        except MemoryError:
            st.error("⚠️ Limite de memória RAM atingido. Tente analisar um fragmento menor.")
        except Exception as e:
            st.error(f"Ocorreu um erro durante o processamento: {str(e)}")
            
        finally:
            gc.collect()
    else:
        st.warning("Por favor, insira um texto para analisar.")
