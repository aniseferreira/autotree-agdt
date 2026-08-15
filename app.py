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

# ============================================================
# PREPOSIÇÕES GREGAS
# ============================================================

PREPOSICOES_GREGAS = {
    "ἀμφί", "ἀμφὶ",
    "ἀνά", "ἀνὰ",
    "ἀντί", "ἀντὶ",
    "ἀπό", "ἀπὸ",
    "διά", "διὰ",
    "εἰς",
    "ἐκ", "ἐξ",
    "ἐν",
    "ἐπί", "ἐπὶ",
    "κατά", "κατὰ",
    "μετά", "μετὰ",
    "παρά", "παρὰ",
    "περί", "περὶ",
    "πρό", "πρὸ",
    "πρός", "πρὸς",
    "σύν", "σὺν",
    "ὑπέρ", "ὑπὲρ",
    "ὑπό", "ὑπὸ"
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

    for token in sent.words:

        words.append({
            "id": int(token.id),
            "text": token.text or "_",
            "lemma": token.lemma or "_",
            "upos": token.upos or "_",
            "xpos": token.xpos or "_",
            "feats": token.feats or "_",
            "head": int(token.head),
            "deprel": token.deprel or "_",
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
    # PARTÍCULAS — tratamento contextual
    # --------------------------------------------------------
    #
    # AuxY não deve ser atribuído apenas pelo lema.
    # Se a palavra está funcionando como coordenador UD (cc),
    # ela será tratada pela regra de coordenação.
    #
    # AuxY é usado aqui para partículas discursivas/adverbiais
    # quando o Stanza as analisa como advmod.
    # --------------------------------------------------------

    if text in {"ἂν", "ἄν", "αν"} and rel == "advmod":
        return "AuxY"

    if text in {"γὰρ", "γαρ"} and rel == "advmod":
        w["lemma"] = "γάρ"
        w["xpos"] = "d--------"
        return "AuxY"

    if text in {"μὲν", "μεν"} and rel == "advmod":
        w["lemma"] = "μέν"
        w["xpos"] = "d--------"
        return "AuxY"

    # δέ como partícula discursiva; se for cc, permanece
    # disponível para a regra de coordenação.
    if text in {"δέ", "δε"} and rel == "advmod":
        w["lemma"] = "δέ"
        w["xpos"] = "d--------"
        return "AuxY"

    if text == "καί" and rel == "advmod":
        return "AuxZ"

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
    # --------------------------------------------------------

    if rel == "cop":
        return "cop"

    # --------------------------------------------------------
    # AUX
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
    # ADJETIVO SUBSTANTIVADO
    # --------------------------------------------------------

    if w["upos"] == "ADJ":
        if rel in {
            "obj", "iobj", "obl", "obl:arg",
            "nsubj", "nsubj:pass", "csubj"
        }:
            if rel in {"nsubj", "nsubj:pass", "csubj"}:
                return "SBJ"
            if rel in {"obj", "iobj", "obl:arg"}:
                return "OBJ"
            if rel == "obl":
                return "ADV"

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
        "cc": "COORD",
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
    Converte somente os membros DIRETOS de cada coordenação UD.

    Regras:
    1. Cada grupo é definido pelos elementos com deprel=conj
       que têm o mesmo head UD.
    2. O primeiro elemento é o head sintático da coordenação.
    3. O último cc da coordenação é COORD.
    4. Somente os membros diretos recebem função + _CO.
    5. Conjunções anteriores ao COORD final recebem AuxY.
    6. Dependentes internos dos membros não recebem _CO por
       pertencerem à mesma oração/coordenada.
    """

    elementos_conj = [
        w for w in words
        if w["deprel"].split(":")[0] == "conj"
    ]

    if not elementos_conj:
        return

    grupos = {}
    for segundo in elementos_conj:
        grupos.setdefault(str(segundo["head"]), []).append(segundo)

    grupos_ordenados = sorted(
        grupos.items(),
        key=lambda item: min([int(x["id"]) for x in item[1]] + [int(item[0])])
    )

    for primeiro_id, segundos in grupos_ordenados:

        primeiro = get_word(words, primeiro_id)
        if primeiro is None:
            continue

        segundos = sorted(segundos, key=lambda w: int(w["id"]))
        elementos = sorted(
            [primeiro] + segundos,
            key=lambda w: int(w["id"])
        )

        funcao_base = primeiro.get("new_rel") or mapear_relacao_basica(primeiro)

        if funcao_base in {"conj", "cc", "COORD"}:
            funcao_base = "PRED"

        if funcao_base.endswith("_CO"):
            funcao_base = funcao_base[:-3]

        ids_elementos = {str(e["id"]) for e in elementos}
        primeiro_pos = int(primeiro["id"])
        ultima_pos = int(elementos[-1]["id"])

        conjuncoes = [
            w for w in words
            if (
                w["deprel"].split(":")[0] == "cc"
                and str(w["head"]) in ids_elementos
                and primeiro_pos < int(w["id"]) <= ultima_pos
            )
        ]
        conjuncoes.sort(key=lambda w: int(w["id"]))

        if not conjuncoes:
            for elemento in elementos:
                elemento["new_rel"] = f"{funcao_base}_CO"
                if elemento is not primeiro:
                    elemento["new_head"] = primeiro["new_head"]
            continue

        coord_real = conjuncoes[-1]
        coord_id = str(coord_real["id"])

        antigo_head = primeiro["new_head"]
        coord_real["new_rel"] = "COORD"
        coord_real["new_head"] = antigo_head

        # SOMENTE os membros diretos da coordenação recebem _CO.
        for elemento in elementos:
            elemento["new_rel"] = f"{funcao_base}_CO"
            elemento["new_head"] = coord_id

        coord_real["new_rel"] = "COORD"
        coord_real["new_head"] = antigo_head

        for conj_anterior in conjuncoes[:-1]:
            conj_anterior["new_rel"] = "AuxY"
            conj_anterior["new_head"] = coord_id

        for elemento in elementos:
            if elemento["new_rel"] == "conj":
                elemento["new_rel"] = f"{funcao_base}_CO"
                elemento["new_head"] = coord_id


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

        if (
            text in {"ἄν", "ἂν", "αν"}
            and w["deprel"] == "advmod"
        ):
            w["new_rel"] = "AuxY"

        elif (
            text == "καί"
            and w["deprel"] == "advmod"
        ):
            w["new_rel"] = "AuxZ"

        elif (
            text in {"γὰρ", "γαρ", "μὲν", "μεν", "δέ", "δε"}
            and w["deprel"] == "advmod"
        ):
            w["new_rel"] = "AuxY"


# ============================================================
# PARTICÍPIO SUBSTANTIVADO
# ============================================================

def aplicar_participios_substantivados(words):
    """
    Reconhece construções do tipo ὁ + particípio.

    Quando o artigo é o determinante do particípio, o particípio
    constitui o núcleo de um grupo nominal. Em particular:
      - nominativo → SBJ quando funciona como argumento do verbo;
      - acusativo/dativo → OBJ quando funciona como argumento;
      - demais casos preservam a função já atribuída.

    Dependentes internos do particípio não recebem _CO.
    """

    for participio in words:

        if participio["upos"] != "VERB":
            continue

        feats = participio.get("feats") or ""
        if "VerbForm=Part" not in feats:
            continue

        artigos = [
            w for w in words
            if (
                w["upos"] == "DET"
                and w["head"] == participio["id"]
                and normalizar(w["lemma"]) in {"ὁ", "ο"}
            )
        ]

        if not artigos:
            continue

        caso = None
        m = re.search(r"Case=([^|]+)", feats)
        if m:
            caso = m.group(1)

        if participio["deprel"] in {
            "nsubj", "nsubj:pass", "csubj", "obj", "iobj", "obl:arg"
        }:
            if participio["deprel"] in {"nsubj", "nsubj:pass", "csubj"}:
                participio["new_rel"] = "SBJ"
            else:
                participio["new_rel"] = "OBJ"
            continue

        if caso == "Nom":
            participio["new_rel"] = "SBJ"
        elif caso in {"Acc", "Dat"}:
            participio["new_rel"] = "OBJ"

        for artigo in artigos:
            artigo["new_rel"] = "ATR"
            artigo["new_head"] = participio["id"]


# ============================================================
# INFINITIVO COM FUNÇÃO DE SUJEITO
# ============================================================

def aplicar_infinitivo_sujeito(words):
    """
    Trata a construção em que um infinitivo é o sujeito de um
    predicado nominal/adjetival.

    Exemplo:
        Φοβεῖσθαι οὐδενὶ ἀγαθόν

    AGDT:
        [0] PRED
        ├── Φοβεῖσθαι SBJ
        ├── οὐδενί ADV
        └── ἀγαθόν PNOM

    O token artificial [0] é acrescentado ao final da sentença.
    """

    for infinitivo in list(words):

        if infinitivo["upos"] != "VERB":
            continue

        feats = infinitivo.get("feats") or ""
        if "VerbForm=Inf" not in feats:
            continue

        # Caso já reconhecido pelo parser como sujeito oracional.
        if infinitivo["deprel"] in {"csubj", "csubj:pass"}:
            infinitivo["new_rel"] = "SBJ"
            continue

        # Caso importante: infinitivo já convertido para SBJ e
        # dependente de um adjetivo/nome que estava na raiz.
        if infinitivo["new_rel"] != "SBJ":
            continue

        pred_id = infinitivo["new_head"]
        pred = get_word(words, pred_id)

        if pred is None:
            continue

        if pred["upos"] not in {"ADJ", "NOUN", "PROPN"}:
            continue

        # O predicativo nominal/adjetival não pode ser PRED.
        pred["new_rel"] = "PNOM"

        # O artificial PRED recebe um novo ID após os tokens reais.
        artificial_id = max(int(w["id"]) for w in words) + 1

        artificial = {
            "id": artificial_id,
            "text": "[0]",
            "lemma": "_",
            "upos": "_",
            "xpos": "_",
            "feats": "_",
            "head": 0,
            "deprel": "root",
            "new_rel": "PRED",
            "new_head": 0,
            "artificial": True,
            "insertion_id": "0003e",
        }

        # Reposicionar a raiz sintática.
        infinitivo["new_rel"] = "SBJ"
        infinitivo["new_head"] = artificial_id

        pred["new_head"] = artificial_id

        # Dependentes diretos do predicado original passam para o
        # PRED artificial, conservando a função quando possível.
        for child in words:
            if child["id"] == pred["id"]:
                continue
            if child["new_head"] != pred["id"]:
                continue

            # Neste padrão, o dativo de referência é ADV.
            feats_child = child.get("feats") or ""
            if (
                "Case=Dat" in feats_child
                and child["new_rel"] == "ATR"
            ):
                child["new_rel"] = "ADV"

            child["new_head"] = artificial_id

        words.append(artificial)
        break



# ============================================================
# CORREÇÕES ESTRUTURAIS AGDT — INFINITIVO + COORDENAÇÃO
# ============================================================

def corrigir_coordenacao_de_oracao_subordinada(words):
    """
    Depois das pontes AuxC, garante a arquitetura:

        AuxC
          │
        COORD
        /   \
      X_CO  Y_CO

    quando a coordenação é interna à oração introduzida por AuxC.

    Não propaga _CO para subordinadas internas.
    """

    # Localizar AuxC que introduz uma oração contendo uma coordenação.
    for auxc in words:
        if auxc["new_rel"] != "AuxC":
            continue

        # COORD diretamente associado ao contexto dessa subordinada.
        coords = [
            w for w in words
            if (
                w["new_rel"] == "COORD"
                and (
                    w["new_head"] == auxc["id"]
                    or w["head"] == auxc["id"]
                )
            )
        ]

        for coord in coords:
            coord["new_head"] = auxc["id"]

            membros = [
                w for w in words
                if w["new_head"] == coord["id"]
                and w["new_rel"].endswith("_CO")
            ]

            for membro in membros:
                # _CO somente em membros diretos.
                membro["new_head"] = coord["id"]

    # Caso em que o COORD ainda está ligado ao head anterior do AuxC:
    # se houver dois predicados _CO no mesmo grupo, unificá-los sob
    # o mesmo COORD.
    for coord in [w for w in words if w["new_rel"] == "COORD"]:
        membros = [
            w for w in words
            if w["new_rel"].endswith("_CO")
        ]

        # Não fazemos reagrupamento global: apenas se já houver
        # pelo menos dois membros ligados ao próprio coord.
        ligados = [
            w for w in membros
            if w["new_head"] == coord["id"]
        ]

        if len(ligados) >= 2:
            continue


def garantir_auxk_na_root(words):
    """
    AuxK sempre depende diretamente da ROOT AGDT.
    """
    for w in words:
        if w["new_rel"] == "AuxK":
            w["new_head"] = 0


# ============================================================
# PIPELINE UD → AGDT
# ============================================================

def converter_sentenca(sent):

    words = construir_words(sent)

    # 1. mapa básico
    inicializar_agdt(words)

    # 2. partículas e auxiliares contextuais
    aplicar_auxiliares_especiais(words)

    # 3. infinitivos com função de sujeito
    aplicar_infinitivo_sujeito(words)

    # 4. particípios substantivados
    aplicar_participios_substantivados(words)

    # 5. AuxV conservador
    aplicar_auxv(words)

    # 6. copula
    aplicar_copula(words)

    # 7. AuxC — ponte sintática antes da reconstrução da coordenação
    aplicar_auxc(words)

    # 8. Coordenação — somente membros diretos
    aplicar_coordenacao(words)

    # 9. AuxP
    aplicar_auxp(words)

    # 10. Coordenação de oração subordinada
    corrigir_coordenacao_de_oracao_subordinada(words)

    # 11. artigos
    aplicar_artigos_repetidos(words)

    # 12. pontuação
    corrigir_pontuacao_em_coordenacao(words)

    # 13. AuxK sempre na ROOT
    garantir_auxk_na_root(words)

    # --------------------------------------------------------
    # Garantia final de head
    # --------------------------------------------------------

    for w in words:

        if w["new_head"] is None:
            w["new_head"] = w["head"]

        if w["new_rel"] is None:
            w["new_rel"] = w["deprel"]

        # REGRA AGDT:
        # pontuação de fechamento AuxK pertence à ROOT,
        # nunca ao PRED.
        if w["new_rel"] == "AuxK":
            w["new_head"] = 0

    return {
        "text": sent.text,
        "words": words
    }


# ============================================================
# PROCESSAMENTO DO TEXTO
# ============================================================

def processar_texto(nlp, texto):

    resultados = []

    with st.spinner("Analisando o grego antigo..."):

        doc = nlp(texto)

    sentencas = doc.sentences
    total = len(sentencas)

    progress = st.progress(0)

    for i, sent in enumerate(sentencas):

        resultados.append(
            converter_sentenca(sent)
        )

        if total:
            progress.progress(
                min((i + 1) / total, 1.0)
            )

    progress.empty()
    gc.collect()

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

            # Tokens artificiais AGDT (ex.: [0]) não recebem
            # postag/lemma lexical; usam a marcação específica
            # do formato ALDT/AGDT.
            if w.get("artificial"):
                attrs = {
                    "id": str(w["id"]),
                    "insertion_id": w.get("insertion_id", "0003e"),
                    "artificial": "elliptic",
                    "relation": w["new_rel"],
                    "form": w["text"],
                    "head": str(w["new_head"]),
                }
            else:
                attrs = {
                    "id": str(w["id"]),
                    "form": w["text"],
                    "lemma": w["lemma"],
                    "postag": w["xpos"],
                    "head": str(w["new_head"]),
                    "relation": w["new_rel"]
                }

            ET.SubElement(
                sentence,
                "word",
                attrs
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
# ENTRADA
# ============================================================

st.subheader("Entrada do texto")

modo_entrada = st.radio(
    "Escolha a forma de entrada:",
    [
        "Frase / texto curto",
        "Arquivo .txt"
    ],
    horizontal=True
)

texto = None
nome_base = "teste_agdt"

# ============================================================
# FRASE / TEXTO CURTO
# ============================================================

if modo_entrada == "Frase / texto curto":

    texto = st.text_area(
        "Digite uma ou mais frases em grego antigo:",
        height=150,
        placeholder=(
            "Ex.: καὶ γὰρ οἱ κύβοι ἀριθμὸν περιέχουσι "
            "καὶ ψῆφοι λέγονται."
        )
    )

    nome_base = "teste_agdt"

# ============================================================
# ARQUIVO
# ============================================================

else:

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

# ============================================================
# BOTÃO DE ANÁLISE
# ============================================================
#
# O botão fica sempre visível.
# Quando não há texto, fica desabilitado.
# ============================================================

analisar = st.button(
    "🔬 Analisar texto",
    type="primary",
    use_container_width=True,
    disabled=not bool(texto and texto.strip())
)

if analisar:

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
            f"{len(resultado)} sentença(s) analisada(s)."
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
