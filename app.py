import gc
import xml.etree.ElementTree as ET
from xml.dom import minidom
import streamlit as st
import pandas as pd
import stanza

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Anotador AGDT (XML / Arethusa)",
    page_icon="🏛️",
    layout="wide"
)

MODEL_MAPPING = {
    "ancient-greek-perseus (Padrão AGDT/Perseids)": "perseus",
    "ancient-greek-proiel": "proiel"
}

@st.cache_resource(show_spinner="Baixando e carregando modelo linguístico na memória...")
def load_stanza_pipeline(package_name: str):
    stanza.download("grc", package=package_name, verbose=False)
    nlp = stanza.Pipeline(
        lang="grc",
        package=package_name,
        processors="tokenize,pos,lemma,depparse",
        verbose=False
    )
    return nlp

def stanza_to_agdt_dataframe(doc) -> pd.DataFrame:
    rows = []
    for sent_idx, sentence in enumerate(doc.sentences, start=1):
        for word in sentence.words:
            rows.append({
                "sent_id": sent_idx,
                "id": word.id,
                "form": word.text,
                "lemma": word.lemma if word.lemma else "_",
                "upostag": word.upos if word.upos else "_",
                "xpostag": word.xpos if word.xpos else "_",  # Tag AGDT de 9 posições
                "feats": word.feats if word.feats else "_",
                "head": word.head if word.head is not None else 0,
                "deprel": word.deprel if word.deprel else "_"
            })
    return pd.DataFrame(rows)

def generate_agdt_xml(df: pd.DataFrame) -> str:
    """
    Converte o DataFrame para a estrutura de XML legível pelo Arethusa (Perseids/AGDT Schema).
    """
    treebank = ET.Element("treebank", version="1.5", lang="grc", **{"xml:lang": "grc"})
    
    grouped = df.groupby("sent_id")
    for sent_id, group in grouped:
        sentence = ET.SubElement(treebank, "sentence", id=str(sent_id), document_id="", subdoc="", span="")
        for _, row in group.iterrows():
            ET.SubElement(
                sentence,
                "word",
                id=str(row["id"]),
                form=str(row["form"]),
                lemma=str(row["lemma"]),
                postag=str(row["xpostag"]),  # Tag morfossintática do AGDT
                head=str(row["head"]),
                relation=str(row["deprel"])
            )
            
    # Formatação com indentação para visualização limpa
    rough_string = ET.tostring(treebank, encoding="utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

# --- 2. INTERFACE STREAMLIT ---

st.title("🏛️ Anotador AGDT - Exportador de XML para o Arethusa")
st.markdown(
    "Gere anotações sintáticas para **Grego Antigo** compatíveis com o **Arethusa (Perseids)** e **Tündra**."
)

st.sidebar.header("Configurações")
selected_label = st.sidebar.selectbox(
    "Modelo de Treebank Grego:",
    list(MODEL_MAPPING.keys()),
    help="O modelo 'perseus' utiliza as tags morfossintáticas originais do AGDT no XPOS."
)
package_choice = MODEL_MAPPING[selected_label]

default_text = "Μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος"
input_text = st.text_area(
    "Texto em Grego Antigo:",
    value=default_text,
    height=150
)

if st.button("Analisar e Gerar XML AGDT", type="primary"):
    if input_text.strip():
        try:
            nlp = load_stanza_pipeline(package_choice)
            
            with st.spinner("Processando morfossintaxe e árvore de dependências..."):
                doc = nlp(input_text)
                df_agdt = stanza_to_agdt_dataframe(doc)
                xml_data = generate_agdt_xml(df_agdt)
            
            st.success("Análise concluída!")
            
            # Exibição de Abas para Tabela e XML
            tab1, tab2 = st.tabs(["📊 Tabela de Anotação", "📄 Código XML (Arethusa)"])
            
            with tab1:
                st.dataframe(df_agdt, use_container_width=True)
                
            with tab2:
                st.code(xml_data, language="xml")
            
            # Botões de Download
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="🏛️ Baixar XML (Para Arethusa / Perseids)",
                    data=xml_data,
                    file_name=f"{package_choice}_arethusa.xml",
                    mime="application/xml"
                )
                
            with col2:
                tsv_data = df_agdt.to_csv(sep="\t", index=False)
                st.download_button(
                    label="📥 Baixar CoNLL-U / TSV (Para Tündra)",
                    data=tsv_data,
                    file_name=f"{package_choice}_parsed.tsv",
                    mime="text/tab-separated-values"
                )

        except Exception as e:
            st.error(f"Ocorreu um erro durante o processamento: {str(e)}")
            
        finally:
            gc.collect()
    else:
        st.warning("Por favor, insira um texto para analisar.")
