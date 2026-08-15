import gc
import os
import urllib.request
import streamlit as st
import pandas as pd
import trankit

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Anotador AGDT (Trankit)",
    page_icon="🏛️",
    layout="wide"
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".trankit_cache")

def ensure_model_exists(treebank_model: str, embedding_type: str = "xlm-roberta-base") -> str:
    """
    Garante a presença do arquivo .zip do modelo no diretório local.
    Baixa do Hugging Face caso não exista (evitando timeout no Oregon).
    """
    target_dir = os.path.join(CACHE_DIR, embedding_type)
    os.makedirs(target_dir, exist_ok=True)
    
    zip_path = os.path.join(target_dir, f"{treebank_model}.zip")
    
    if not os.path.exists(zip_path):
        st.info(f"Baixando modelo {treebank_model} ({embedding_type}) do Hugging Face...")
        hf_url = f"https://huggingface.co/uonlp/trankit/resolve/main/models/v1.0.0/{embedding_type}/{treebank_model}.zip"
        
        with st.spinner("Baixando arquivo do modelo (~350 MB)..."):
            urllib.request.urlretrieve(hf_url, zip_path)
        st.success("Download do modelo concluído!")
        
    return zip_path

@st.cache_resource(show_spinner="Carregando modelo linguístico Trankit...")
def load_trankit_pipeline(treebank_model: str) -> trankit.Pipeline:
    """
    Carrega o pipeline com no_remote=True para bloquear requisições ao nlp.uoregon.edu
    e utiliza 'xlm-roberta-base' para respeitar o limite de 1 GB de RAM do Streamlit Cloud.
    """
    gc.collect()
    
    embedding_type = "xlm-roberta-base"
    
    # 1. Garante que o arquivo zip está presente localmente
    ensure_model_exists(treebank_model, embedding_type=embedding_type)
    
    # 2. Inicializa o pipeline em modo offline/no_remote
    pipeline = trankit.Pipeline(
        lang=treebank_model,
        embedding=embedding_type,
        gpu=False,
        cache_dir=CACHE_DIR,
        no_remote=True  # <- Parâmetro correto para impedir chamadas externas ao servidor da UOregon
    )
    return pipeline

def trankit_to_agdt_dataframe(doc: dict) -> pd.DataFrame:
    """
    Converte a estrutura de dicionários do Trankit no formato tabular CoNLL-U/AGDT.
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
            st.error("⚠️ Limite de memória RAM atingido. Tente analisar um fragmento menor.")
        except Exception as e:
            st.error(f"Ocorreu um erro durante o processamento: {str(e)}")
            
        finally:
            gc.collect()
    else:
        st.warning("Por favor, insira um texto para analisar.")
