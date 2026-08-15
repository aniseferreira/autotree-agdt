import io
import gc
import re
import streamlit as st
import stanza
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="PLN Grego Antigo — UD / AGDT",
    page_icon="ἑ",
    layout="wide"
)


# ============================================================
# STANZA
# ============================================================

@st.cache_resource(show_spinner=False)
def carregar_pipeline():

    stanza.download(
        "grc",
        package="perseus",
        processors="tokenize,lemma,pos,depparse"
    )

    return stanza.Pipeline(
        lang="grc",
        package="perseus",
        processors="tokenize,lemma,pos,depparse",
        tokenize_no_ssplit=True,
        verbose=False
    )


# ============================================================
# CONSTANTES AGDT
# ============================================================

NEGACOES = {
    "οὐ", "οὐκ", "οὐχ", "οὐχι", "μή"
}

CONJUNCOES_SUBORDINATIVAS = {
    "ὅτι", "οτι",
    "διότι", "διοτι",
    "ὡς", "ως",
    "ἐπειδή", "επειδη",
    "ἐπεί", "επει",
    "ἐπειδήπερ", "επειδηπερ",
    "ἐάν", "εαν",
    "ἄν", "αν",
    "εἰ", "ει",
    "ἐάνπερ", "εανπερ"
}

PONTUACAO_AUXG = {
    "ʼ", "῾", "'", "’",
    "“", "”", "‘", "’",
    "\""
}

PONTUACAO_AUXK = {
    ".", "·", "·", ";", ":", ";"
}

PONTUACAO_AUXX = {
    ","
}


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalizar(s):
    if not s:
        return ""
    return s.lower().strip()


def is_conj_subordinativa(word):
    return normalizar(word) in CONJUNCOES_SUBORDINATIVAS


# ============================================================
# CONSTRUÇÃO DA REPRESENTAÇÃO UD
# ============================================================

def construir_words(sent):

    words = []

    for word in sent.words:

        deprel = word.deprel or "_"

        # Mantemos a relação UD original intacta.
        words.append({
            "id": int(word.id),
            "text": word.text,
            "lemma": word.lemma or "_",
            "upos": word.upos or "_",
            "xpos": word.xpos or "_",
            "feats": word.feats or "_",
            "head": int(word.head),
            "deprel": deprel,
            "new_rel": None,
            "new_head": None
        })

    return words


# ============================================================
# MAPEAMENTO UD → AGDT BÁSICO
# ============================================================

def mapear_relacao_basica(w):

    text = normalizar(w["text"])
    rel = w["deprel"]

    # --------------------------------------------------------
    # PONTUAÇÃO
    # --------------------------------------------------------

    if text in PONTUACAO_AUXX:
        return "AuxX"

    if text in PONTUACAO_AUXK:
        return "AuxK"

    if text in PONTUACAO_AUXG:
        return "AuxG"

    # --------------------------------------------------------
    # PARTÍCULAS
    # --------------------------------------------------------

    if text in {"ἂν", "ἄν", "αν"} and rel == "advmod":
        return "AuxY"

    if text == "καί" and rel == "advmod":
        return "AuxZ"

    if text in {"γὰρ", "γαρ"}:
        w["lemma"] = "γάρ"
        w["xpos"] = "d--------"
        return "AuxY"

    if text in {"μὲν", "μεν"}:
        w["lemma"] = "μέν"
        w["xpos"] = "d--------"
        return "AuxY"

    # --------------------------------------------------------
    # NEGADORES
    # --------------------------------------------------------

    if text in NEGACOES:

        if rel in {"mark", "sconj"}:
            return "AuxC"

        return "AuxZ"

    # --------------------------------------------------------
    # CONJUNÇÕES SUBORDINATIVAS
    # --------------------------------------------------------

    if is_conj_subordinativa(w["text"]):
        return "AuxC"
    # --------------------------------------------------------
    # PREPOSIÇÕES
    #
    # O Stanza pode classificar uma preposição como ADV.
    # Para o AGDT, preposições são AuxP.
    # --------------------------------------------------------
    
    if text in PREPOSICOES_GREGAS:
        w["upos"] = "ADP"
        w["xpos"] = "r--------"
        return "AuxP"

    # --------------------------------------------------------
    # VOCATIVO
    # --------------------------------------------------------

    if rel == "vocative":
        return "ExD"

    # --------------------------------------------------------
    # COP
    #
    # NÃO transformamos εἰμί automaticamente em AuxV.
    # --------------------------------------------------------

    if rel == "cop":
        return "cop"

    # --------------------------------------------------------
    # AUX
    #
    # Só aceitamos AuxV quando o parser explicitamente
    # produziu uma relação UD auxiliar.
    #
    # Isto é deliberadamente conservador.
    # --------------------------------------------------------

    if rel == "aux":
        return "AuxV"

    # --------------------------------------------------------
    # PONTUAÇÃO UD
    # --------------------------------------------------------

    if rel == "punct":

        if w["text"] in PONTUACAO_AUXX:
            return "AuxX"

        if w["text"] in PONTUACAO_AUXK:
            return "AuxK"

        if w["text"] in PONTUACAO_AUXG:
            return "AuxG"

        return "AuxG"

    # --------------------------------------------------------
    # MAPA GERAL
    # --------------------------------------------------------

    mapa = {

        "nsubj": "SBJ",
        "nsubj:pass": "SBJ",

        "obj": "OBJ",

        "iobj": "OBJ",

        "xcomp": "OBJ",

        "ccomp": "OBJ",

        "csubj": "SBJ",

        "csubj:pass": "SBJ",

        "root": "PRED",

        "advmod": "ADV",

        "advcl": "ADV",

        "obl": "ADV",

        "obl:arg": "OBJ",

        "amod": "ATR",

        "det": "ATR",

        "nmod": "ATR",

        "acl": "ATR",

        "acl:relcl": "ATR",

        "case": "AuxP",

        "mark": "AuxC",

        "sconj": "AuxC",

        "cc": "COORD"
    }

    return mapa.get(rel, rel)


# ============================================================
# INICIALIZAÇÃO DA ANOTAÇÃO AGDT
# ============================================================

def inicializar_agdt(words):

    for w in words:

        w["new_rel"] = mapear_relacao_basica(w)

        w["new_head"] = w["head"]


# ============================================================
# AUXILIAR: ENCONTRAR TOKEN
# ============================================================

def get_word(words, word_id):

    for w in words:
        if w["id"] == int(word_id):
            return w

    return None


# ============================================================
# AUXP — PREPOSIÇÃO COMO PONTE
# ============================================================

def aplicar_auxp(words):

    for prep in words:

        if prep["new_rel"] != "AuxP":
            continue

        old_head = prep["head"]

        governed = get_word(words, old_head)

        if governed is None:
            continue

        # Na UD:
        #
        # VERBO ── obl ── NOME
        #                    ↑
        #                  case
        #
        # No AGDT:
        #
        # VERBO ── AuxP ── PREP
        #                    │
        #                   ADV/OBJ...
        #
        original_relation = governed["new_rel"]

        # A preposição passa a depender do antigo head
        prep["new_head"] = governed["new_head"]

        # O elemento regido passa a depender da preposição
        governed["new_head"] = prep["id"]

        # Preserva a função que o elemento teria em relação
        # ao head original.
        if original_relation in {
            "OBJ", "ADV", "ATR", "SBJ", "PRED"
        }:
            governed["new_rel"] = original_relation

        # Evita ciclos
        if prep["new_head"] == prep["id"]:
            prep["new_head"] = 0


# ============================================================
# AUXC — CONJUNÇÃO SUBORDINATIVA COMO PONTE
# ============================================================

def aplicar_auxc(words):

    for conj in words:

        if conj["new_rel"] != "AuxC":
            continue

        # Não tratar partículas negativas comuns como ponte
        # quando o parser as usa apenas como negação.
        if normalizar(conj["text"]) in NEGACOES:
            if conj["deprel"] not in {"mark", "sconj"}:
                continue

        subordinate_id = conj["head"]

        subordinate = get_word(
            words,
            subordinate_id
        )

        if subordinate is None:
            continue

        # O verbo/elemento subordinado conserva a relação
        # que tinha em relação ao head superior.
        subordinate_relation = (
            subordinate["new_rel"]
            or "OBJ"
        )

        superior_head = subordinate["new_head"]

        # conjunção passa a depender do head superior
        conj["new_head"] = superior_head

        # subordinado passa a depender da conjunção
        subordinate["new_head"] = conj["id"]

        # A relação do subordinado em relação à conjunção
        # é a função que ele exercia diante do head superior.
        if subordinate_relation in {
            "ADV",
            "OBJ",
            "ATR",
            "SBJ"
        }:
            subordinate["new_rel"] = subordinate_relation


# ============================================================
# COPULAÇÃO
# ============================================================

def aplicar_copula(words):

    for cop in words:

        if cop["new_rel"] != "cop":
            continue

        predicative_id = cop["head"]

        predicative = get_word(
            words,
            predicative_id
        )

        if predicative is None:
            continue

        # Em UD:
        #
        # predicativo ← cop ← εἰμί
        #
        # No AGDT:
        #
        # εἰμί PRED
        #   │
        # PNOM
        #
        cop_old_head = predicative["new_head"]

        cop["new_rel"] = "PRED"
        cop["new_head"] = cop_old_head

        predicative["new_rel"] = "PNOM"
        predicative["new_head"] = cop["id"]

        # Sujeitos que estavam ligados ao predicativo
        # passam para εἰμί.
        for child in words:

            if (
                child["new_head"]
                == predicative["id"]
                and child["new_rel"]
                == "SBJ"
            ):

                child["new_head"] = cop["id"]


# ============================================================
# AUXV — TRATAMENTO CONSERVADOR
# ============================================================

def aplicar_auxv(words):

    """
    IMPORTANTE:

    Não inferimos AuxV simplesmente porque o lema é εἰμί.

    AuxV só é criado aqui quando a anotação UD original
    já identificou a forma como 'aux'.

    As construções perifrásticas específicas do AGDT
    deverão ser adicionadas posteriormente a partir de
    exemplos validados dos treebanks AGDT/TüNDRA/Gorman.
    """

    for w in words:

        if (
            w["lemma"] == "εἰμί"
            and w["deprel"] == "aux"
        ):
            w["new_rel"] = "AuxV"


# ============================================================
# COORDENAÇÃO
# ============================================================

# ============================================================
# COORDENAÇÃO UD → AGDT
# ============================================================

def aplicar_coordenacao(words):
    """
    Converte coordenações da UD para a estrutura AGDT.

    Regras AGDT:
    1. 'conj' é uma relação UD e nunca deve aparecer no AGDT.
    2. O último elemento coordenador é COORD.
    3. Os elementos coordenados dependem de COORD.
    4. Todos os elementos coordenados recebem:
           função sintática + _CO
    5. Conjunções anteriores ao último coordenador recebem AuxY.

    Exemplo simples:

        UD:
            περιέχουσι   root
            λέγονται     conj
            καί           cc

        AGDT:

                         COORD
                           καί
                         /   \
                PRED_CO       PRED_CO
              περιέχουσι     λέγονται


    Coordenação múltipla:

        A καὶ B καὶ C

        AGDT:

                         COORD
                           καὶ
                    /      |      \
                 A_CO     B_CO     C_CO
                           ↑
                         AuxY
                           καὶ

    A função sintática dos elementos coordenados é herdada
    do primeiro elemento.
    """

    # --------------------------------------------------------
    # 1. Localizar todos os elementos que a UD marcou como
    #    "conj".
    # --------------------------------------------------------

    elementos_conj = [
        w for w in words
        if w["deprel"].split(":")[0] == "conj"
    ]

    if not elementos_conj:
        return

    # --------------------------------------------------------
    # 2. Agrupar elementos coordenados pelo primeiro elemento.
    #
    #    Em UD:
    #
    #       A ← head de B
    #       A ← head de C
    #
    #    para:
    #
    #       A e B e C
    #
    #    Portanto todos os "conj" que têm o mesmo head
    #    pertencem à mesma coordenação.
    # --------------------------------------------------------

    grupos = {}

    for segundo in elementos_conj:

        head_id = str(
            segundo["head"]
        )

        grupos.setdefault(
            head_id,
            []
        ).append(segundo)

    # --------------------------------------------------------
    # 3. Processar cada coordenação separadamente.
    # --------------------------------------------------------

    for primeiro_id, segundos in grupos.items():

        primeiro = get_word(
            words,
            primeiro_id
        )

        if primeiro is None:
            continue

        # ----------------------------------------------------
        # Ordenar os elementos pela posição no texto.
        # ----------------------------------------------------

        segundos.sort(
            key=lambda w: int(w["id"])
        )

        elementos = [
            primeiro
        ] + segundos

        elementos.sort(
            key=lambda w: int(w["id"])
        )

        # ----------------------------------------------------
        # Determinar a função sintática do primeiro elemento.
        # ----------------------------------------------------

        funcao_base = primeiro.get(
            "new_rel"
        )

        if not funcao_base:

            funcao_base = mapear_relacao_basica(
                primeiro
            )

        # ----------------------------------------------------
        # NUNCA permitir que uma relação UD sobreviva.
        # ----------------------------------------------------

        if funcao_base in {
            "conj",
            "cc"
        }:
            funcao_base = "PRED"

        # Se já houver _CO, remover para evitar:
        #
        # PRED_CO_CO
        #
        if funcao_base.endswith(
            "_CO"
        ):
            funcao_base = funcao_base[:-3]

        # ----------------------------------------------------
        # 4. Encontrar as conjunções associadas ao grupo.
        # ----------------------------------------------------

        primeiro_pos = int(
            primeiro["id"]
        )

        ultima_pos = int(
            elementos[-1]["id"]
        )

        conjuncoes = []

        for w in words:

            if (
                w["deprel"].split(":")[0]
                != "cc"
            ):
                continue

            wid = int(w["id"])

            # A conjunção deve estar na região da coordenação.
            if not (
                primeiro_pos
                < wid
                <= ultima_pos
            ):
                continue

            # Aceitamos:
            #
            # cc ligado ao primeiro elemento
            # ou cc ligado a algum elemento coordenado.
            head = str(
                w["head"]
            )

            ids_elementos = {
                str(e["id"])
                for e in elementos
            }

            if head in ids_elementos:
                conjuncoes.append(w)

        # ----------------------------------------------------
        # 5. Se não houver cc explícito, ainda corrigimos
        #    "conj" para função_CO.
        # ----------------------------------------------------

        if not conjuncoes:

            for elemento in elementos:

                if elemento is primeiro:
                    continue

                elemento["new_rel"] = (
                    f"{funcao_base}_CO"
                )

            continue

        # ----------------------------------------------------
        # Ordenar conjunções.
        # ----------------------------------------------------

        conjuncoes.sort(
            key=lambda w: int(w["id"])
        )

        # ----------------------------------------------------
        # 6. O ÚLTIMO coordenador é o COORD verdadeiro.
        # ----------------------------------------------------

        coord_real = conjuncoes[-1]

        coord_id = str(
            coord_real["id"]
        )

        antigo_head = str(
            primeiro["new_head"]
        )

        coord_real["new_rel"] = "COORD"

        coord_real["new_head"] = (
            antigo_head
        )

        # ----------------------------------------------------
        # 7. Todos os elementos coordenados recebem
        #    função + _CO e dependem do COORD.
        # ----------------------------------------------------

        for elemento in elementos:

            elemento["new_rel"] = (
                f"{funcao_base}_CO"
            )

            elemento["new_head"] = (
                coord_id
            )

        # ----------------------------------------------------
        # 8. Restaurar COORD depois de alterar os elementos.
        # ----------------------------------------------------

        coord_real["new_rel"] = "COORD"

        coord_real["new_head"] = (
            antigo_head
        )

        # ----------------------------------------------------
        # 9. Todas as conjunções anteriores ao último
        #    são AuxY.
        # ----------------------------------------------------

        for conj_anterior in conjuncoes[:-1]:

            conj_anterior["new_rel"] = "AuxY"

            conj_anterior["new_head"] = (
                coord_id
            )

        # ----------------------------------------------------
        # 10. Segurança:
        # nenhum elemento UD "conj" pode chegar ao AGDT.
        # ----------------------------------------------------

        for elemento in elementos:

            if elemento["new_rel"] == "conj":

                elemento["new_rel"] = (
                    f"{funcao_base}_CO"
                )

                elemento["new_head"] = (
                    coord_id
                )

# ============================================================
# COORDENAÇÃO MÚLTIPLA / CASOS COM VÍRGULAS
# ============================================================

def corrigir_pontuacao_em_coordenacao(words):

    """
    AuxX NÃO é estrutura de ponte.

    Entretanto, em uma estrutura coordenada, uma vírgula pode
    ter função estrutural de separação, mas continua sendo
    pontuação. Portanto, nesta V1 ela permanece AuxX.

    O COORD continua sendo uma palavra coordenativa real.
    """

    for w in words:

        if w["text"] in PONTUACAO_AUXX:
            w["new_rel"] = "AuxX"

        elif w["text"] in PONTUACAO_AUXK:
            w["new_rel"] = "AuxK"

        elif w["text"] in PONTUACAO_AUXG:
            w["new_rel"] = "AuxG"


# ============================================================
# ARTIGOS REPETIDOS / BLOCO NOMINAL
# ============================================================

def aplicar_artigos_repetidos(words):

    for idx, w in enumerate(words):

        if w["upos"] != "DET":
            continue

        if idx < 2:
            continue

        for idx_ant in range(
            idx - 1,
            -1,
            -1
        ):

            anterior = words[idx_ant]

            if anterior["upos"] != "DET":
                continue

            if (
                len(w["xpos"]) >= 9
                and len(anterior["xpos"]) >= 9
                and w["xpos"][6:9]
                == anterior["xpos"][6:9]
                and w["xpos"][6:9] != "---"
            ):

                centro = anterior["head"]

                w["new_head"] = centro
                w["new_rel"] = "ATR"

                break


# ============================================================
# AUXILIARES ESPECÍFICOS
# ============================================================

def aplicar_auxiliares_especiais(words):

    for w in words:

        text = normalizar(w["text"])

        # ἄν como AuxY quando partícula
        if (
            text in {"ἄν", "ἂν", "αν"}
            and w["deprel"] == "advmod"
        ):
            w["new_rel"] = "AuxY"

        # καί adverbial
        if (
            text == "καί"
            and w["deprel"] == "advmod"
        ):
            w["new_rel"] = "AuxZ"


# ============================================================
# PIPELINE UD → AGDT
# ============================================================

def converter_sentenca(sent):

    words = construir_words(sent)

    # 1. mapa básico
    inicializar_agdt(words)

    # 2. auxiliares especiais
    aplicar_auxiliares_especiais(words)

    # 3. AuxV conservador
    aplicar_auxv(words)

    # 4. copula
    aplicar_copula(words)

    # 5. Coordenação
    aplicar_coordenacao(words)

    # 6. AuxP
    aplicar_auxp(words)

    # 7. AuxC
    aplicar_auxc(words)

    # 8. artigos
    aplicar_artigos_repetidos(words)

    # 9. pontuação final
    corrigir_pontuacao_em_coordenacao(words)

    # --------------------------------------------------------
    # Garantia final de head
    # --------------------------------------------------------

    for w in words:

        if w["new_head"] is None:
            w["new_head"] = w["head"]

        if w["new_rel"] is None:
            w["new_rel"] = w["deprel"]

    return {
        "text": sent.text,
        "words": words
    }


# ============================================================
# PROCESSAMENTO DO TEXTO
# ============================================================

def processar_texto(nlp, texto):

    linhas = [
        linha.strip()
        for linha in texto.splitlines()
        if linha.strip()
    ]

    resultados = []

    total = len(linhas)

    progress = st.progress(0)

    for i, linha in enumerate(linhas):

        doc = nlp(linha)

        for sent in doc.sentences:

            resultados.append(
                converter_sentenca(sent)
            )

        if total:
            progress.progress(
                min(
                    (i + 1) / total,
                    1.0
                )
            )

        if i % 50 == 0:
            gc.collect()

    progress.empty()

    return resultados


# ============================================================
# CONLL-U
# ============================================================

def gerar_conllu(sentences):

    output = io.StringIO()

    for i, sent in enumerate(
        sentences,
        start=1
    ):

        output.write(
            f"# sent_id = {i}\n"
        )

        output.write(
            f"# text = {sent['text']}\n"
        )

        for w in sent["words"]:

            row = [
                str(w["id"]),
                w["text"],
                w["lemma"],
                w["upos"],
                w["xpos"],
                w["feats"],
                str(w["new_head"]),
                w["deprel"],
                "_",
                "_"
            ]

            output.write(
                "\t".join(row) + "\n"
            )

        output.write("\n")

    return output.getvalue()


# ============================================================
# AGDT XML
# ============================================================

def gerar_agdt_xml(
    sentences,
    nome_base
):

    root = ET.Element(
        "treebank",
        {
            "xml:lang": "grc",
            "format": "aldt",
            "version": "1.5"
        }
    )

    for i, sent in enumerate(
        sentences,
        start=1
    ):

        sentence = ET.SubElement(
            root,
            "sentence",
            {
                "id": str(i),
                "document_id": nome_base
            }
        )

        for w in sent["words"]:

            ET.SubElement(
                sentence,
                "word",
                {
                    "id": str(w["id"]),
                    "form": w["text"],
                    "lemma": w["lemma"],
                    "postag": w["xpos"],
                    "head": str(w["new_head"]),
                    "relation": w["new_rel"]
                }
            )

    xml_bytes = ET.tostring(
        root,
        encoding="utf-8"
    )

    return minidom.parseString(
        xml_bytes
    ).toprettyxml(
        indent="  ",
        encoding="utf-8"
    )


# ============================================================
# DATAFRAME DE VISUALIZAÇÃO
# ============================================================

def criar_dataframe(sentences):

    rows = []

    for sent_id, sent in enumerate(
        sentences,
        start=1
    ):

        for w in sent["words"]:

            rows.append({
                "Sent.": sent_id,
                "ID": w["id"],
                "Forma": w["text"],
                "Lema": w["lemma"],
                "UPOS": w["upos"],
                "XPOS": w["xpos"],
                "UD Head": w["head"],
                "UD Rel": w["deprel"],
                "AGDT Head": w["new_head"],
                "AGDT Rel": w["new_rel"]
            })

    return pd.DataFrame(rows)


# ============================================================
# INTERFACE
# ============================================================

st.title(
    "PLN Grego Antigo"
)

st.subheader(
    "Anotação automática de dependências — UD / AGDT"
)

st.write(
    """
    O aplicativo recebe um texto em grego antigo, realiza
    a análise morfossintática e sintática com Stanza e produz
    duas representações:

    **UD CoNLL-U** e **AGDT XML**.
    """
)

st.divider()


# ============================================================
# UPLOAD
# ============================================================

arquivo = st.file_uploader(
    "Arquivo de texto grego antigo",
    type=["txt"]
)

if arquivo:

    nome_base = arquivo.name.rsplit(
        ".",
        1
    )[0]

    texto = arquivo.getvalue().decode(
        "utf-8"
    )

    st.success(
        f"Arquivo carregado: {arquivo.name}"
    )

    with st.expander(
        "Ver texto de entrada"
    ):
        st.text_area(
            "Texto",
            texto,
            height=250
        )

    if st.button(
        "🔬 Analisar texto",
        type="primary",
        use_container_width=True
    ):

        try:

            nlp = carregar_pipeline()

            with st.spinner(
                "Analisando o grego antigo..."
            ):

                resultado = processar_texto(
                    nlp,
                    texto
                )

            st.session_state[
                "resultado"
            ] = resultado

            st.session_state[
                "nome_base"
            ] = nome_base

            st.success(
                f"{len(resultado)} sentenças analisadas."
            )

        except Exception as e:

            st.error(
                "Erro durante o processamento."
            )

            st.exception(e)


# ============================================================
# RESULTADOS
# ============================================================

if "resultado" in st.session_state:

    sentences = st.session_state[
        "resultado"
    ]

    nome_base = st.session_state[
        "nome_base"
    ]

    st.divider()

    st.header(
        "Resultado da anotação"
    )

    df = criar_dataframe(
        sentences
    )

    st.dataframe(
        df,
        use_container_width=True,
        height=550
    )

    st.divider()

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # CONLL-U
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "UD — CoNLL-U"
        )

        conllu = gerar_conllu(
            sentences
        )

        st.download_button(
            "⬇️ Baixar CoNLL-U",
            data=conllu.encode(
                "utf-8"
            ),
            file_name=(
                f"{nome_base}.conllu"
            ),
            mime="text/plain",
            use_container_width=True
        )

    # --------------------------------------------------------
    # AGDT
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "AGDT — XML"
        )

        agdt = gerar_agdt_xml(
            sentences,
            nome_base
        )

        st.download_button(
            "⬇️ Baixar AGDT XML",
            data=agdt,
            file_name=(
                f"{nome_base}_agdt.xml"
            ),
            mime="application/xml",
            use_container_width=True
        )

    # --------------------------------------------------------
    # VISUALIZAÇÃO
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Visualização da anotação"
    )

    sent_num = st.number_input(
        "Sentença",
        min_value=1,
        max_value=len(sentences),
        value=1,
        step=1
    )

    sent = sentences[
        sent_num - 1
    ]

    st.write(
        f"**{sent['text']}**"
    )

    sent_df = pd.DataFrame([
        {
            "ID": w["id"],
            "Forma": w["text"],
            "Lema": w["lemma"],
            "UPOS": w["upos"],
            "XPOS": w["xpos"],
            "UD Head": w["head"],
            "UD": w["deprel"],
            "AGDT Head": w["new_head"],
            "AGDT": w["new_rel"]
        }
        for w in sent["words"]
    ])

    st.dataframe(
        sent_df,
        use_container_width=True,
        hide_index=True
    )
