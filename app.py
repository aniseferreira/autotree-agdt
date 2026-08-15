import gc
import os
import urllib.request
import streamlit as st
import pandas as pd
import trankit

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Anotador AGDT (Trankit Large)",
    page_icon="🏛️",
    layout="wide"
)

# Define o caminho do cache local no repositório
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".trankit_cache")

def ensure_large_model_exists(treebank_model: str):
    """
    Verifica se o arquivo zip do modelo 'large' está na pasta local.
    Se não estiver (ex: não enviado ao GitHub), faz o download pelo Hugging Face.
    """
    embedding_type = "xlm-roberta-large"
    target_dir = os.path.join(CACHE_DIR, embedding_type)
    os.makedirs(target_dir, exist_ok=True)
    
    zip_path = os.path.join(target_dir, f"{treebank_model}.zip")
    
    # Se o arquivo zip não existir localmente, baixa do Hugging Face
    if not os.path.exists(zip_path):
        st.info(f"Modelo local não encontrado em {zip_path}. Baixando do Hugging Face...")
        hf_url = f"https://huggingface.co/uonlp/trankit/resolve/main/models/v1.0.0/{embedding_type}/{treebank_model}.zip"
        
        with st.spinner("Baixando modelo xlm-roberta-large (~110 MB)..."):
            urllib.request.urlretrieve(hf_url, zip_path)
        st.success("Download do modelo concluído!")

@st.cache_resource(show_spinner="Carregando modelo linguístico Trankit (XLM-RoBERTa Large)...")
def load_trankit_pipeline(treebank_model: str) -> trankit.Pipeline:
    """
    Carrega o pipeline do Trankit utilizando explicitamente o modelo XLM-RoBERTa Large.
    """
    gc.collect()
    
    # Garante a existência do arquivo na estrutura de pastas
    ensure_large_model_exists(treebank_model)
    
    # Inicializa o Trankit apontando para o embedding 'xlm-roberta-large'
    pipeline = trankit.Pipeline(
        lang=treebank_model,
        embedding='xlm-roberta-large',
        gpu=False,
        cache_dir=CACHE_DIR
    )
    return pipeline

def trankit_to_agdt_dataframe(doc: dict) -> pd.DataFrame:
    """
    Converte os dicionários do Trankit no formato tabular CoNLL-U/AGDT.
    """
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

# --- Interface Gráfica ---

st.title("🏛️ Anotador AGDT - Treebank de Dependências (Trankit Large)")
st.markdown(
    "Ferramenta de anotação automática para **Grego Antigo** usando o backend "
    "**Trankit (XLM-RoBERTa-Large)** alinhado ao padrão AGDT/Perseids."
)

st.sidebar.header("Configurações")
model_choice = st.sidebar.selectbox(
    "Modelo de Treebank Grego:",
    ["ancient-greek-perseus", "ancient-greek-proiel"],
    help="O modelo 'ancient-greek-perseus' segue o padrão AGDT/Perseids."
)

# Entrada de Texto
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
            
            # Exibição dos resultados
            st.subheader("Tabela de Anotação (Padrão CoNLL-U / AGDT)")
            st.dataframe(df_agdt, use_container_width=True)
            
            # Botão de Download
            tsv_data = df_agdt.to_csv(sep="\t", index=False)
            st.download_button(
                label="📥 Baixar Anotações (TSV / CoNLL-U)",
                data=tsv_data,
                file_name=f"{model_choice}_parsed.tsv",
                mime="text/tab-separated-values"
            )

        except MemoryError:
            st.error("⚠️ Limite de memória RAM atingido. O modelo Large exige mais recursos.")
        except Exception as e:
            st.error(f"Ocorreu um erro durante o processamento: {str(e)}")
            
        finally:
            gc.collect()
    else:
        st.warning("Por favor, insira um texto para analisar.")
