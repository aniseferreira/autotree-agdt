import gc
import os
import json
import zipfile
import urllib.request
import streamlit as st
import pandas as pd
import trankit
import trankit.utils.tbinfo as tbinfo

# --- MONKEY-PATCHING DO TRANKIT PARA APONTAR AO HUGGING FACE ---
# Sobrescreve as URLs padrão do Trankit que apontavam para o Oregon
tbinfo.URL = "https://huggingface.co/uonlp/trankit/resolve/main/models/v1.0.0/"

# Função para engatar as URLs do Hugging Face para metadados
def patched_download_file(url, save_path):
    if "available_langs.json" in url:
        url = "https://huggingface.co/uonlp/trankit/raw/main/available_langs.json"
    elif "version.json" in url:
        url = "https://huggingface.co/uonlp/trankit/raw/main/version.json"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    urllib.request.urlretrieve(url, save_path)

# Aplica o patch na função interna de download do Trankit
trankit.tpipeline.download_file = patched_download_file
trankit.utils.tbinfo.download_file = patched_download_file

# --- CONFIGURAÇÃO DA APLICAÇÃO STREAMLIT ---

st.set_page_config(
    page_title="Anotador AGDT (Trankit)",
    page_icon="🏛️",
    layout="wide"
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".trankit_cache")

def prepare_trankit_offline(treebank_model: str, embedding_type: str = "xlm-roberta-base"):
    """
    Garante o download do zip do modelo a partir do Hugging Face,
    descompacta na estrutura esperada e gera o arquivo de controle local.
    """
    target_dir = os.path.join(CACHE_DIR, embedding_type)
    os.makedirs(target_dir, exist_ok=True)
    
    extracted_model_dir = os.path.join(target_dir, treebank_model)
    zip_path = os.path.join(target_dir, f"{treebank_model}.zip")
    
    # Download do zip se a pasta descompactada não existir
    if not os.path.exists(extracted_model_dir):
        if not os.path.exists(zip_path):
            st.info(f"Baixando modelo {treebank_model} do Hugging Face...")
            hf_url = f"https://huggingface.co/uonlp/trankit/resolve/main/models/v1.0.0/{embedding_type}/{treebank_model}.zip"
            
            with st.spinner("Baixando arquivo do modelo (~350 MB)..."):
                urllib.request.urlretrieve(hf_url, zip_path)
            st.success("Download do Hugging Face concluído!")

        st.info("Descompactando modelo para uso local...")
        with st.spinner("Extraindo arquivos de peso..."):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        st.success("Modelo extraído com sucesso!")

    # Registra a linguagem localmente
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
    
    prepare_trankit_offline(treebank_model, embedding_type=embedding_type)
    
    # Inicializa o Trankit utilizando o monkey-patching aplicado acima
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

# --- INTERFACE GRÁFICA ---

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
