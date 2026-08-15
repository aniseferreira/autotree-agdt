import gc
import os
import streamlit as st
import pandas as pd
import trankit

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Anotador AGDT (Trankit)",
    page_icon="🏛️",
    layout="wide"
)

# Define diretório local de cache para evitar poluição e rastrear download no servidor
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".trankit_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

@st.cache_resource(show_spinner="Carregando modelo linguístico (isso pode levar 1-2 minutos na primeira execução)...")
def load_trankit_pipeline(treebank_model: str) -> trankit.Pipeline:
    """
    Carrega o pipeline do Trankit configurado para economizar RAM no Streamlit Cloud.
    """
    # Força a limpeza de memória antes de instanciar novo modelo
    gc.collect()
    
    pipeline = trankit.Pipeline(
        lang=treebank_model,
        gpu=False,                  # Desativa GPU (Streamlit Cloud roda em CPU)
        cache_dir=CACHE_DIR,        # Define local de salvar modelos
        save_disk_space=True        # Remove arquivos compactados temporários após download
    )
    return pipeline

def trankit_to_agdt_dataframe(doc: dict) -> pd.DataFrame:
    """
    Converte o dicionário retornado pelo Trankit no formato tabular CoNLL-U/AGDT.
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
    "Ferramenta de anotação automática para **Grego Antigo** ajustada ao padrão "
    "**Ancient Greek Dependency Treebank (AGDT / Perseids)** usando o backend **Trankit**."
)

# Barra Lateral
st.sidebar.header("Configurações")
model_choice = st.sidebar.selectbox(
    "Modelo de Treebank Grego:",
    ["ancient_greek-perseus", "ancient_greek-proiel"],
    help="O modelo 'perseus' é alinhado com as convenções diretas do AGDT."
)

st.sidebar.markdown("---")
st.sidebar.caption("💡 **Nota para Cloud:** A primeira execução baixa os pesos do modelo (aprox. 500 MB) e pode demorar um pouco mais.")

# Entrada de Texto
default_text = "Μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος"
input_text = st.text_area(
    "Texto em Grego Antigo:",
    value=default_text,
    height=150
)

# Botão de Ação
if st.button("Analisar Dependências", type="primary"):
    if input_text.strip():
        try:
            # Carrega o modelo de forma segura com cache
            nlp = load_trankit_pipeline(model_choice)
            
            with st.spinner("Processando morfossintaxe e parse de dependências..."):
                doc = nlp(input_text)
                df_agdt = trankit_to_agdt_dataframe(doc)
            
            st.success("Análise concluída com sucesso!")
            
            # Exibição dos dados
            st.subheader("Tabela de Anotação (Padrão CoNLL-U / AGDT)")
            st.dataframe(df_agdt, use_container_width=True)
            
            # Botão para Download
            tsv_data = df_agdt.to_csv(sep="\t", index=False)
            st.download_button(
                label="📥 Baixar Anotações (TSV / CoNLL-U)",
                data=tsv_data,
                file_name="agdt_trankit_parsed.tsv",
                mime="text/tab-separated-values"
            )

        except MemoryError:
            st.error("⚠️ Limite de memória RAM do Streamlit Cloud atingido. Tente analisar um texto menor ou reiniciar a aplicação.")
        except Exception as e:
            st.error(f"Ocorreu um erro durante o processamento: {str(e)}")
            
        finally:
            # Coleta de lixo extra após o processamento para liberar RAM
            gc.collect()
    else:
        st.warning("Por favor, insira um texto para analisar.")
