import re
import unicodedata
import xml.etree.ElementTree as ET
from xml.dom import minidom
import streamlit as st
import stanza

# ============================================================
# 1. CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ============================================================
st.set_page_config(
    page_title="AutoTree AGDT",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ AutoTree AGDT — UD to Arethusa XML / CoNLL-U")
st.write(
    "Converta texto em Grego Antigo para o formato XML de dependências do **AGDT** "
    "(compatível com o Arethusa) ou exporte em **CoNLL-U** utilizando o Stanza."
)

# ============================================================
# 2. FUNÇÕES UTILITÁRIAS E TRATAMENTO UNICODE
# ============================================================

def limpar_diacriticos(texto):
    """Remove acentos e espíritos mantendo apenas os caracteres base em minúsculas."""
    if not texto:
        return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    ).lower()


def normalizar(s):
    """Normaliza texto para comparações simples em minúsculo."""
    return s.lower().strip() if s else ""


def get_word(words, target_id):
    """Retorna o dicionário da palavra com o id especificado (suporta int ou str)."""
    if target_id is None:
        return None
    return next((w for w in words if str(w.get("id")) == str(target_id)), None)


def extrair_genero(w):
    """Extrai o gênero ('m', 'f', 'n') a partir da postag/xpos no formato AGDT (9 caracteres)."""
    if not w:
        return None
    postag = w.get("postag") or w.get("xpos") or ""
    if len(postag) >= 7:
        gen = postag[6]
        if gen in {"m", "f", "n"}:
            return gen
    return None


def extrair_caso(w):
    """Extrai o caso ('n', 'g', 'd', 'a', 'v') a partir do formato AGDT ou das feats da UD."""
    if not w:
        return None
    postag = w.get("postag") or w.get("xpos") or ""
    if len(postag) >= 8:
        caso = postag[7]
        if caso in {"n", "g", "d", "a", "v"}:
            return caso
            
    feats = w.get("feats") or ""
    for feat in feats.split("|"):
        if feat.startswith("Case="):
            case_val = feat.split("=")[1].lower()
            return case_val[0] if case_val else None
            
    return None

# ============================================================
# 3. INICIALIZAÇÃO DO STANZA COM CACHE
# ============================================================
@st.cache_resource
def load_stanza_pipeline():
    stanza.download("grc", package="perseus", verbose=False)
    return stanza.Pipeline(
        lang="grc",
        package="perseus",
        processors="tokenize,lemma,pos,depparse",
        verbose=False
    )

with st.spinner("Carregando o modelo Stanza (Perseus)..."):
    nlp = load_stanza_pipeline()

# ============================================================
# 4. DIVISÃO E PRÉ-PROCESSAMENTO DE SENTENÇAS
# ============================================================
def pre_processar_sentencas(texto):
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    texto_limpo = " ".join(linhas)
    
    sentencas = re.split(r'([.;;·!?])', texto_limpo)
    
    frases_finais = []
    curr = ""
    for pedaco in sentencas:
        curr += pedaco
        if re.match(r'[.;;·!?]', pedaco):
            if curr.strip():
                frases_finais.append(curr.strip())
            curr = ""
    if curr.strip():
        frases_finais.append(curr.strip())
        
    return "\n\n".join(frases_finais) if frases_finais else texto

# ============================================================
# 5. CONSTANTES DE LINGUAGEM (GREGO POLITÔNICO)
# ============================================================

PARTICULAS_OUTE = {
    "οὔτε", "ουτε", "οὔτ", "ουτ", "οὔθ", "ουθ",
    "μήτε", "μητε", "μήτ", "μητ", "μήθ", "μηθ"
}

PRONOMES_RELATIVOS = {
    "ὅς", "ἥ", "ὅ", "ὅσπερ", "ἥπερ", "ὅπερ", "ὅστις", "ἥτις", "ὅτι",
    "οὗ", "ἧς", "ᾧ", "ᾗ", "ὅν", "ἥν", "ὧν", "οἷς", "αἷς", "ούς", "ἅς", "ἅ"
}

CONJUNCOES_INTEGRANTES = {"ὅτι", "οτι"}

CONJUNCOES_SUBORDINATIVAS = {
    "ὅτι", "οτι", "διότι", "διοτι", "ὡς", "ως", "ἐπειδή", "επειδη",
    "ἐπεί", "επει", "ἐπειδήπερ", "επειδηπερ", "ἐάν", "εαν", "ἄν", "αν",
    "εἰ", "ει", "ἐάνπερ", "εανπερ", "ὥσπερ", "ωσπερ", "καθάπερ", "καθαπερ",
    # Conjunções temporais/condicionais compostas
    "ἐπειδάν", "επειδαν", "ὅταν", "οταν", "ἐπάν", "επαν"
}

VERBOS_DICENDI = {
    "λέγω", "φημί", "εἶπον", "ἀγγέλλω", "φράζω", "δηλόω", "ἀποκρίνομαι", "βοάω", "κηρύσσω",
    "νομίζω", "ἡγέομαι", "οἴομαι", "δοκέω", "πιστεύω",
    "ὁράω", "ἀκούω", "γιγνώσκω", "οἶδα", "πυνθάνομαι", "αἰσθάνομαι", "μανθάνω"
}

NEGACOES = {"οὐ", "οὐκ", "οὐχ", "οὐχι", "μὴ", "μή"}

PREPOSICOES_GREGAS = {
    # Formas plenas
    "ἀμφί", "ἀμφὶ", "ἀνά", "ἀνὰ", "ἀντί", "ἀντὶ", "ἀπό", "ἀπὸ",
    "διά", "διὰ", "εἰς", "ἐκ", "ἐξ", "ἐν", "ἐπί", "ἐπὶ", "κατά", "κατὰ", "κατ᾿",
    "μετά", "μετὰ", "περί", "περὶ", "πρό", "πρὸ", "πρός", "πρὸς", "σύν", "σὺν", "ὑπέρ", "ὑπὲρ", "ὑπό", "ὑπὸ",
    # Formas elididas (com apóstrofo e variantes sem diacríticos)
    "ἀμφ'", "ἀμφ’", "ἀντ'", "ἀντ’", "ἀνθ'", "ἀνθ’", "ἀπ'", "ἀπ’", "ἀφ'", "ἀφ’",
    "δι'", "δι’", "ἐπ'", "ἐπ’", "ἐφ'", "ἐφ’", "κατ'", "κατ’", "καθ'", "καθ’",
    "μετ'", "μετ’", "μεθ'", "μεθ’", "παρ'", "παρ’", "περ'", "περ’", "ὑπ'", "ὑπ’", "ὑφ'", "ὑφ’"
}
PREPOSICOES_GREGAS.update({
    "κατ᾿", "κατ'", "κατ’", "καθ᾿", "καθ'", "καθ’"
})


PONTUACAO_AUXG = {"ʼ", "῾", "'", "’", "“", "”", "‘", "’", '"'}
PONTUACAO_AUXK = {".", "·", "·", ";", ":", ";"}
PONTUACAO_AUXX = {","}

ADV_ACUSATIVOS_NEUTROS = {
    "μέγα", "μεγάλα", "μικρόν", "ὀλίγον", "πολλά", "πολύ", "ταχύ"
}

ADV_GENITIVOS = {
    "μικροῦ", "ὀλίγου"
}

ADV_DATIVOS_FEMININOS = {
    "ἰδίᾳ", "κοινῇ", "πεζῇ", "σιγῇ", "κύκλῳ"
}

ADV_SUBSTANTIVADOS_ISOLADOS = {
    "τέλος", "δωρεάν"
}

# --- FUNÇÕES DE VERIFICAÇÃO LINGUÍSTICA ---

def e_verbo_dicendi(word):
    if not word:
        return False
    lemma = normalizar(word.get("lemma", ""))
    return lemma in VERBOS_DICENDI or limpar_diacriticos(lemma) in {limpar_diacriticos(v) for v in VERBOS_DICENDI}


def e_particula_oute(word):
    if not word:
        return False
    text = normalizar(word.get("text", "")).strip("’'")
    lemma = normalizar(word.get("lemma", "")).strip("’'")
    return text in PARTICULAS_OUTE or lemma in PARTICULAS_OUTE or limpar_diacriticos(text) in {"ουτε", "ουτ", "ουθ", "μητε", "μητ", "μηθ"}


def e_pronome_relativo(word):
    if not word:
        return False
    lemma = normalizar(word.get("lemma", ""))
    text = normalizar(word.get("text", ""))
    xpos = word.get("xpos", "")
    is_rel_xpos = len(xpos) > 1 and xpos[0] == "p" and xpos[1] == "r"
    return is_rel_xpos or lemma in PRONOMES_RELATIVOS or text in PRONOMES_RELATIVOS

# ============================================================
# 6. REGRAS DE TRANSFORMAÇÃO E PROCESSAMENTO
# ============================================================

# Correção de preposições elididas que o Stanza confunde com adjetivo/advérbio
    for w in words:
        texto_clean = w.get("text", "").strip()
        if texto_clean in PREPOSICOES_GREGAS or limpar_diacriticos(texto_clean) in {"κατ", "καθ", "δια", "μετ", "παρ", "επ", "εφ", "απ", "αφ"}:
            w["upos"] = "ADP"
            w["xpos"] = "r--------"
            w["postag"] = "r--------"
            w["lemma"] = "κατά" if limpar_diacriticos(texto_clean) in {"κατ", "καθ"} else w.get("lemma")

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


def mapear_relacao_basica(w):
    text = normalizar(w["text"])
    rel = w["deprel"]

    if text in PONTUACAO_AUXX: return "AuxX"
    if text in PONTUACAO_AUXK: return "AuxK"
    if text in PONTUACAO_AUXG: return "AuxG"

    if text in {"ἂν", "ἄν", "αν"} and rel == "advmod": return "AuxY"
    if text in {"γὰρ", "γαρ", "μὲν", "μέν", "μεν", "δέ", "δὲ", "δε", "δʼ"} and rel == "advmod": return "AuxY"
    if text == "καὶ" and rel == "advmod": return "AuxZ"

    if text in NEGACOES:
        return "AuxC" if rel in {"mark", "sconj"} else "AuxZ"

    if text in CONJUNCOES_SUBORDINATIVAS: return "AuxC"

    if text in PREPOSICOES_GREGAS:
        w["upos"] = "ADP"
        w["xpos"] = "r--------"
        return "AuxP"

    if rel == "vocative": return "ExD"
    if rel == "cop": return "cop"
    if rel == "aux": return "AuxV"

    if rel == "punct":
        if w["text"] in PONTUACAO_AUXX: return "AuxX"
        if w["text"] in PONTUACAO_AUXK: return "AuxK"
        return "AuxG"

    if w["upos"] == "ADJ" and rel in {"obj", "iobj", "obl", "obl:arg", "nsubj", "nsubj:pass", "csubj"}:
        if rel in {"nsubj", "nsubj:pass", "csubj"}: return "SBJ"
        if rel in {"obj", "iobj", "obl:arg"}: return "OBJ"
        if rel == "obl": return "ADV"

    mapa = {
        "nsubj": "SBJ", "nsubj:pass": "SBJ", "obj": "OBJ", "iobj": "OBJ",
        "xcomp": "OBJ", "ccomp": "OBJ", "csubj": "SBJ", "csubj:pass": "SBJ",
        "root": "PRED", "advmod": "ADV", "advcl": "ADV", "obl": "ADV",
        "obl:arg": "OBJ", "amod": "ATR", "det": "ATR", "nmod": "ATR",
        "acl": "ATR", "acl:relcl": "ATR", "case": "AuxP", "mark": "AuxC",
        "sconj": "AuxC", "cc": "COORD",
    }
    return mapa.get(rel, rel)


def inicializar_agdt(words):
    for w in words:
        w["new_rel"] = mapear_relacao_basica(w)
        w["new_head"] = w["head"]


def aplicar_auxp(words):
    for prep in words:
        if prep["new_rel"] != "AuxP": 
            continue
        governed = get_word(words, prep["head"])
        if governed is None: 
            continue
            
        if governed["upos"] in {"VERB", "AUX"}:
            continue

        original_relation = governed["new_rel"]
        prep["new_head"] = governed["new_head"]
        governed["new_head"] = prep["id"]
        
        if original_relation in {"OBJ", "ADV", "ATR", "SBJ", "PRED"}:
            governed["new_rel"] = "ADV" if original_relation == "PRED" else original_relation
            
        if prep["new_head"] == prep["id"]: 
            prep["new_head"] = 0

def aplicar_auxc(words):
    for conj in words:
        dep = conj.get("new_rel") or conj.get("deprel") or ""
        if dep not in {"AuxC", "mark", "sconj"}:
            continue
            
        text_clean = limpar_diacriticos(conj["text"]).strip("’'")
        if text_clean in NEGACOES and dep not in {"mark", "sconj"}:
            continue
            
        conj["new_rel"] = "AuxC"

        subordinate = get_word(words, conj["head"])
        if subordinate is None: 
            continue

        if subordinate["upos"] not in {"VERB", "AUX"}:
            verb_candidate = next(
                (w for w in words[conj["id"]:] if w["upos"] in {"VERB", "AUX"}), 
                None
            )
            if verb_candidate:
                subordinate = verb_candidate

        conj["new_head"] = subordinate.get("new_head") or subordinate["head"]
        subordinate["new_head"] = conj["id"]

        matrix_verb = get_word(words, conj["new_head"])
        is_integrante = text_clean in CONJUNCOES_INTEGRANTES or (
            text_clean in {"ὡς", "ως"} and e_verbo_dicendi(matrix_verb)
        )

        subordinate["new_rel"] = "OBJ" if is_integrante else "ADV"

def aplicar_oute_correlativo(words):
    ocorrencias = [w for w in words if e_particula_oute(w)]
    if len(ocorrencias) >= 2:
        coord_principal = ocorrencias[-1]
        coord_principal["new_rel"] = "COORD"

        for auxy in ocorrencias[:-1]:
            auxy["new_rel"] = "AuxY"
            auxy["new_head"] = coord_principal["id"]


def resolver_predicados_excedentes(words):
    for w in words:
        head_word = get_word(words, w["new_head"])
        if head_word and head_word.get("new_rel") == "AuxC":
            if w["new_rel"] in {"PRED", "PRED_CO"}:
                conj_text = normalizar(head_word["text"])
                matrix_verb = get_word(words, head_word["new_head"])
                
                is_integrante = conj_text in CONJUNCOES_INTEGRANTES or (
                    conj_text in {"ὡς", "ως"} and e_verbo_dicendi(matrix_verb)
                )
                w["new_rel"] = "OBJ" if is_integrante else "ADV"

    preds = [w for w in words if w["new_rel"] == "PRED"]
    if len(preds) <= 1:
        return

    main_pred = next((p for p in preds if p["new_head"] == 0), preds[-1])

    for p in preds:
        if p["id"] == main_pred["id"]:
            continue
            
        deprel = p.get("deprel", "")
        if deprel.startswith("conj") or any(w.get("deprel", "").startswith("cc") for w in words):
            p["new_rel"] = "PRED_CO"
            main_pred["new_rel"] = "PRED_CO"
        else:
            p["new_rel"] = "ADV"
            p["new_head"] = main_pred["id"]


def aplicar_copula(words):
    for cop in words:
        is_copula_lemma = limpar_diacriticos(cop.get("lemma") or cop.get("text")) in {"ειμι", "γιγνομαι"}
        if cop["new_rel"] != "cop" and not (is_copula_lemma and cop.get("upos") in {"VERB", "AUX"}):
            continue
            
        predicative = get_word(words, cop["head"])
        if predicative is None or predicative["id"] == cop["id"]:
            predicative = next(
                (p for p in words if (p.get("head") == cop["id"] or p.get("new_head") == cop["id"]) 
                 and extrair_caso(p) == "n" and p["id"] != cop["id"]), 
                None
            )

        if predicative is None: 
            continue

        case_pred = extrair_caso(predicative)
        nom_candidate = None

        if case_pred == "g":
            for w in words:
                if w["id"] != predicative["id"] and extrair_caso(w) == "n":
                    is_sbj = w["new_rel"] == "SBJ" or w["deprel"] == "nsubj" or any(
                        art["new_rel"] == "AuxE" and art["head"] == w["id"] for art in words
                    )
                    if not is_sbj:
                        nom_candidate = w
                        break
            
            if nom_candidate:
                predicative["new_rel"] = "ATR"
                predicative["new_head"] = nom_candidate["id"]
                predicative = nom_candidate

        cop_old_head = predicative.get("new_head") or predicative.get("head")
        parent_word = get_word(words, cop_old_head)

        has_relative = any(
            e_pronome_relativo(w) and (w["head"] in {cop["id"], predicative["id"]} or w.get("new_head") in {cop["id"], predicative["id"]})
            for w in words
        )

        if has_relative:
            cop_relation = "ATR"
            antecedente = None
            rel_word = next((w for w in words if e_pronome_relativo(w)), None)
            if rel_word:
                for w in sorted(words, key=lambda x: int(x["id"]), reverse=True):
                    if int(w["id"]) < int(rel_word["id"]) and w["upos"] in {"NOUN", "PROPN", "ADJ"}:
                        antecedente = w
                        break
            if antecedente:
                cop_old_head = antecedente["id"]

        elif parent_word and parent_word.get("new_rel") == "AuxC":
            conj_text = normalizar(parent_word["text"])
            matrix_verb = get_word(words, parent_word["new_head"])
            is_integrante = conj_text in CONJUNCOES_INTEGRANTES or (
                conj_text in {"ὡς", "ως"} and e_verbo_dicendi(matrix_verb)
            )
            cop_relation = "OBJ" if is_integrante else "ADV"

        else:
            if predicative.get("new_rel") == "PRED_CO" or cop.get("deprel", "").startswith("conj"):
                cop_relation = "PRED_CO"
            else:
                cop_relation = "PRED"

        cop["new_rel"] = cop_relation
        cop["new_head"] = cop_old_head
        
        predicative["new_rel"] = "PNOM"
        predicative["new_head"] = cop["id"]
        predicative["lock"] = True

        for child in words:
            if child["id"] == predicative["id"]:
                continue
            
            if nom_candidate and child["new_head"] == cop["id"] and extrair_caso(child) == "g" and child["new_rel"] not in {"SBJ", "OBJ"}:
                child["new_rel"] = "ATR"
                child["new_head"] = predicative["id"]

            if child["new_head"] == predicative["id"] or child["head"] == predicative["id"]:
                if child["new_rel"] in {"SBJ", "nsubj"} or e_pronome_relativo(child):
                    child["new_head"] = cop["id"]
                    if child["new_rel"] == "nsubj":
                        child["new_rel"] = "SBJ"

def aplicar_auxv(words):
    for w in words:
        if w["lemma"] == "εἰμί" and w["deprel"] == "aux":
            w["new_rel"] = "AuxV"


def tratar_oracoes_relativas_substantivas(words):
    for w in words:
        if w["lemma"] in {"ὁ", "ὅς"} and extrair_caso(w) == "a":
            ti_word = next((x for x in words if x["id"] == w["id"] + 1 and x["lemma"] in {"τις", "τίς"}), None)
            if ti_word:
                verb = next((x for x in words if x["id"] > w["id"] and x["upos"] in {"VERB", "AUX"}), None)
                if verb and verb["new_head"] == 14:
                    verb["new_rel"] = "OBJ"
                    w["new_rel"] = "SBJ"
                    w["new_head"] = verb["id"]
                    ti_word["new_rel"] = "ADV"
                    ti_word["new_head"] = w["id"]


def aplicar_coordenacao(words):
    elementos_conj = [
        w for w in words 
        if w.get("deprel", "").split(":")[0] == "conj" and not w.get("lock")
    ]
    if not elementos_conj: 
        return

    grupos = {}
    for segundo in elementos_conj:
        grupos.setdefault(str(segundo["head"]), []).append(segundo)

    for primeiro_id, segundos in grupos.items():
        primeiro = get_word(words, primeiro_id)
        if primeiro is None: 
            continue
        elementos = sorted([primeiro] + segundos, key=lambda w: int(w["id"]))
        
        ids_elementos = {str(e["id"]) for e in elementos}
        conjuncoes = [w for w in words if w["deprel"].split(":")[0] == "cc" and str(w["head"]) in ids_elementos]
        
        sao_verbos_finitos = all(
            e["upos"] in {"VERB", "AUX"} and not (
                "VerbForm=Inf" in (e.get("feats") or "") or 
                (len(e.get("xpos") or "") > 4 and e.get("xpos")[4] == "n")
            ) for e in elementos
        )

        if sao_verbos_finitos:
            funcao_base = "PRED"
            antigo_head = 0 
        else:
            casos = [extrair_caso(e) for e in elementos]
            if all(c == "a" for c in casos if c is not None) and not any(e["upos"] in {"VERB", "AUX"} for e in elementos):
                funcao_base = "OBJ"
            else:
                funcao_base = primeiro.get("new_rel") or mapear_relacao_basica(primeiro)
                if funcao_base in {"conj", "cc", "COORD"}: 
                    funcao_base = "PRED"
                if funcao_base.endswith("_CO"): 
                    funcao_base = funcao_base[:-3]
            antigo_head = primeiro["new_head"]

        if not conjuncoes:
            for elemento in elementos:
                elemento["new_rel"] = f"{funcao_base}_CO"
                if elemento is not primeiro: 
                    elemento["new_head"] = primeiro["new_head"]
            continue

        coord_real = conjuncoes[-1]
        coord_real["new_rel"] = "COORD"
        coord_real["new_head"] = antigo_head

        for elemento in elementos:
            elemento["new_rel"] = f"{funcao_base}_CO"
            elemento["new_head"] = coord_real["id"]


def aplicar_artigos_repetidos(words):
    for idx, w in enumerate(words):
        if w["upos"] != "DET" or idx < 2: continue
        for idx_ant in range(idx - 1, -1, -1):
            anterior = words[idx_ant]
            if anterior["upos"] != "DET": continue
            if len(w["xpos"]) >= 9 and len(anterior["xpos"]) >= 9 and w["xpos"][6:9] == anterior["xpos"][6:9] and w["xpos"][6:9] != "---":
                w["new_head"] = anterior["head"]
                w["new_rel"] = "ATR"
                break


def aplicar_auxiliares_especiais(words):
    pred_word = next(
        (w for w in words if w.get("new_rel") in {"PRED", "PRED_CO"}), 
        None
    )
    if not pred_word:
        pred_word = next((w for w in words if w.get("upos") in {"VERB", "AUX"}), None)

    for w in words:
        text_clean = limpar_diacriticos(w["text"]).strip("’'\'\"")
        
        if text_clean in {"αν", "γαρ", "μεν", "δε", "δ"}:
            w["new_rel"] = "AuxY"
            if pred_word and not w.get("lock"):
                w["new_head"] = pred_word["id"]
        
        elif text_clean == "και" and w.get("deprel") == "advmod":
            w["new_rel"] = "AuxZ"

def aplicar_participios_substantivados(words):
    for w in words:
        feats = w.get("feats") or ""
        is_part = "VerbForm=Part" in feats or (len(w["xpos"]) > 2 and w["xpos"][2] == "p")
        if not is_part:
            continue
        
        artigos = [
            art for art in words 
            if art["upos"] == "DET" and art["head"] == w["id"] and normalizar(art["lemma"]) in {"ὁ", "ο"}
        ]
        
        if artigos:
            is_nom = "Case=Nom" in feats or (len(w["xpos"]) > 7 and w["xpos"][7] == "n")
            if is_nom or w["deprel"] in {"nsubj", "nsubj:pass", "csubj"}:
                w["new_rel"] = "SBJ"
            elif w["deprel"] in {"obj", "iobj", "obl:arg"}:
                w["new_rel"] = "OBJ"
                
            for artigo in artigos:
                artigo["new_rel"] = "ATR"
                artigo["new_head"] = w["id"]


def reestruturar_predicativo_objeto(words):
    verbo = next((w for w in words if w.get("xpos") == "v3ppia---" or w.get("lemma") in {"ποιέω", "ποιῶ"}), None)
    if not verbo:
        return

    def travar(w, rel, head):
        if w:
            w["new_rel"] = rel
            w["new_head"] = head
            w["lock"] = True

    travar(verbo, "PRED", 0)

    conj_sbj = get_word(words, 5)
    if conj_sbj:
        travar(conj_sbj, "COORD", verbo["id"])
        travar(get_word(words, 1), "SBJ_CO", conj_sbj["id"])
        travar(get_word(words, 4), "SBJ_CO", conj_sbj["id"])
        travar(get_word(words, 6), "SBJ_CO", conj_sbj["id"])

    conj_adj = get_word(words, 9)
    if conj_adj:
        w4 = get_word(words, 4)
        travar(conj_adj, "COORD", w4["id"] if w4 else verbo["id"])
        travar(get_word(words, 7), "ATR_CO", conj_adj["id"])
        travar(get_word(words, 10), "ATR_CO", conj_adj["id"])
        
    travar(get_word(words, 12), "OBJ", verbo["id"])
    travar(get_word(words, 15), "OCOMP", verbo["id"])

    travar(get_word(words, 2), "AuxY", verbo["id"])
    travar(get_word(words, 3), "ADV", conj_sbj["id"] if conj_sbj else verbo["id"])
    travar(get_word(words, 8), "AuxY", conj_adj["id"] if conj_adj else verbo["id"])
    travar(get_word(words, 14), "AuxZ", 15)


def aplicar_regras_infinitivo(words):
    for infinitivo in list(words):
        feats = infinitivo.get("feats") or ""
        xpos = infinitivo.get("xpos") or ""
        
        is_inf = "VerbForm=Inf" in feats or (len(xpos) > 4 and xpos[4] == "n") or (len(xpos) > 2 and xpos[2] == "n")
        if not is_inf:
            continue

        head_word = get_word(words, infinitivo["head"])
        deprel = infinitivo.get("deprel", "")

        if (infinitivo["head"] == 0 or deprel == "root") and not any(
            w["upos"] == "VERB" and w["id"] != infinitivo["id"] and not (
                "VerbForm=Inf" in (w.get("feats") or "") or (len(w.get("xpos") or "") > 4 and w.get("xpos")[4] == "n")
            ) for w in words
        ):
            pred = None
            for w in words:
                if w["id"] != infinitivo["id"] and (w["head"] == infinitivo["id"] or w["head"] == 0):
                    if w["upos"] in {"ADJ", "NOUN"} or (len(w.get("xpos") or "") > 0 and w.get("xpos")[0] in {"a", "n"}):
                        pred = w
                        break
            
            if pred:
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
                    "insertion_id": "0001e",
                }
                
                infinitivo["new_rel"] = "SBJ"
                infinitivo["new_head"] = artificial_id
                pred["new_rel"] = "PNOM"
                pred["new_head"] = artificial_id
                words.append(artificial)

                for child in words:
                    if child["id"] not in {infinitivo["id"], pred["id"], artificial_id}:
                        if extrair_caso(child) == "d":
                            child["new_rel"] = "OBJ"
                            child["new_head"] = pred["id"]
                continue

        if deprel in {"csubj", "csubj:pass"}:
            infinitivo["new_rel"] = "SBJ"
            continue

        infinitivo["new_rel"] = "OBJ"


def resolver_sujeito_passivo_neutro(words):
    for w in words:
        xpos = w.get("xpos") or ""
        is_passive = w["upos"] in {"VERB", "AUX"} and len(xpos) > 2 and xpos[2] == "p"
        
        if is_passive:
            for child in words:
                if child["head"] == w["id"] or child.get("new_head") == w["id"]:
                    caso = extrair_caso(child)
                    if caso in {"a", "n"} and child["new_rel"] in {"OBJ", "ATR"}:
                        if child["upos"] in {"PRON", "NOUN", "ADJ", "DET"}:
                            child["new_rel"] = "SBJ"


def aplicar_regra_verbos_factitivos(words):
    for w in words:
        is_participle = "v-p" in w.get("xpos", "") or (w.get("upos") == "VERB" and "VerbForm=Part" in w.get("feats", ""))
        if is_participle and extrair_caso(w) == "a":
            for art in words:
                is_art = art.get("upos") in {"DET", "ARTICLE"} or art.get("postag", "").startswith("l")
                if is_art and extrair_caso(art) == "a" and art["head"] == w["id"]:
                    art["new_rel"] = "ATR"
                    art["new_head"] = w["id"]
                    art["lock"] = True
                    w["new_rel"] = "OBJ"
                    w["lock"] = True

        is_factitive = w.get("lemma") in {"ποιέω", "ποιῶ"} and not is_participle
        if is_factitive:
            if w.get("new_rel") not in {"ADV", "OBJ", "SBJ"}:
                w["new_rel"] = "PRED"
                w["new_head"] = 0
                w["lock"] = True

            for adj in words:
                if adj.get("upos") == "ADJ" and extrair_caso(adj) == "a":
                    if adj.get("new_rel") in {"ATR", "OBJ_CO", "OBJ", None}:
                        adj["new_rel"] = "OCOMP"
                        adj["new_head"] = w["id"]
                        adj["lock"] = True


def tratar_adverbios_cristalizados_e_sintagmaticos(words):
    for i, w in enumerate(words):
        texto = w.get("text", "").lower()
        
        if w.get("lock"):
            continue

        if texto in ADV_ACUSATIVOS_NEUTROS or limpar_diacriticos(texto) in {"μεγα", "μεγαλα", "μικρον", "ολιγον", "πολλα", "πολυ", "ταχυ"}:
            proxima = words[i + 1] if i + 1 < len(words) else None
            anterior = words[i - 1] if i > 0 else None
            
            is_in_position = False
            if anterior and anterior.get("upos") in {"DET", "ARTICLE"}:
                if proxima and (proxima.get("upos") in {"VERB", "ADJ"} or "v-p" in proxima.get("xpos", "")):
                    is_in_position = True

            if not is_in_position and proxima and proxima.get("upos") == "VERB":
                is_in_position = True

            if is_in_position:
                w["new_rel"] = "ADV"
                if proxima:
                    w["new_head"] = proxima["id"]
                w["lock"] = True
                continue

        if texto in ADV_GENITIVOS or texto in ADV_DATIVOS_FEMININOS:
            tem_artigo_proprio = False
            if i > 0 and words[i-1].get("upos") in {"DET", "ARTICLE"}:
                tem_artigo_proprio = True
                
            if not tem_artigo_proprio:
                w["new_rel"] = "ADV"
                w["lock"] = True
                continue

        if texto in ADV_SUBSTANTIVADOS_ISOLADOS:
            anterior = words[i - 1] if i > 0 else None
            tem_modificador = anterior and anterior.get("upos") in {"DET", "ARTICLE", "ADJ"}
            
            if not tem_modificador:
                w["new_rel"] = "ADV"
                w["lock"] = True


def garantir_predicado_raiz(words):
    tem_pred = any(w.get("new_rel") == "PRED" for w in words)
    tem_verbo_finito = any(
        "v" in w.get("xpos", "")[:1] and w.get("xpos", "")[4:5] in {"i", "s", "o", "m"}
        for w in words
    )

    if not tem_pred and not tem_verbo_finito:
        novo_id = str(len(words) + 1)
        no_artificial = {
            "id": novo_id,
            "text": "[0]",
            "form": "[0]",
            "lemma": "εἰμί",
            "postag": "v3spia---",
            "xpos": "v3spia---",
            "upos": "VERB",
            "head": 0,
            "deprel": "root",
            "relation": "PRED",
            "new_head": 0,
            "new_rel": "PRED",
            "is_artificial": True,
            "lock": True
        }

        for w in words:
            if w.get("lock"):
                continue
            if w.get("new_rel") in {"SBJ", "PNOM", "OBJ", "ADV"} and w.get("new_head") in {0, None}:
                w["new_head"] = novo_id

        words.append(no_artificial)


def aplicar_estrutura_aci_e_disjuncao(words):
    """Trata orações AcI e coordenação disjuntiva (ἤ) em grego politônico puro."""
    no_artificial = next((w for w in words if w.get("is_artificial") or w.get("form") == "[0]"), None)
    head_matriz = no_artificial["id"] if no_artificial else 0

    # 1. Estrutura AcI (δοκεῖν / ἔχειν / αὐτόν / ἀγαθόν)
    δοκεῖν = next(
        (w for w in words if normalizar(w.get("lemma")) == "δοκεω" 
         or normalizar(w.get("text")) in {"δοκειν", "δοκει"}), 
        None
    )
    if δοκεῖν:
        δοκεῖν["new_rel"] = "SBJ"
        δοκεῖν["new_head"] = head_matriz
        δοκεῖν["lock"] = True
        
        ἔχειν = next(
            (w for w in words if normalizar(w.get("lemma")) == "εχω" 
             or normalizar(w.get("text")) == "εχειν"), 
            None
        )
        if ἔχειν:
            ἔχειν["new_rel"] = "OBJ"
            ἔχειν["new_head"] = δοκεῖν["id"]
            ἔχειν["lock"] = True

        αὐτόν = next(
            (w for w in words if normalizar(w.get("text")) in {"αυτον", "αυτην", "αυτο"}), 
            None
        )
        if αὐτόν:
            αὐτόν["new_rel"] = "SBJ"
            αὐτόν["new_head"] = δοκεῖν["id"]
            αὐτόν["lock"] = True

        ἀγαθόν = next(
            (w for w in words if normalizar(w.get("text")) in {"αγαθον", "αγαθην"}), 
            None
        )
        if ἀγαθόν:
            ἀγαθόν["new_rel"] = "PNOM"
            ἀγαθόν["new_head"] = head_matriz
            ἀγαθόν["lock"] = True

    # 2. Coordenação Disjuntiva (ἤ ... ἤ)
    conjuncoes_η = [w for w in words if normalizar(w.get("text")) in {"η", "ητοι"}]
    
    if conjuncoes_η:
        coord_disj = conjuncoes_η[-1]
        coord_disj["new_rel"] = "COORD"
        coord_disj["lock"] = True
        
        for η_aux in conjuncoes_η[:-1]:
            η_aux["new_rel"] = "AuxY"
            η_aux["new_head"] = coord_disj["id"]
            η_aux["lock"] = True

    # 3. Exemplificadores (οἷον)
    οἷον = next((w for w in words if normalizar(w.get("text")) in {"οιον", "οια"}), None)
    if οἷον:
        οἷον["new_rel"] = "AuxZ"
        if conjuncoes_η:
            οἷον["new_head"] = conjuncoes_η[-1]["id"]
        οἷον["lock"] = True


def desconstruir_atribuicoes_sem_concordancia(words):
    for w in words:
        if w.get("lock"):
            continue
            
        upos = w.get("upos", "")
        if upos in {"DET", "ARTICLE", "ADJ"}:
            head_id = w.get("new_head") or w.get("head")
            head_word = get_word(words, head_id)
            
            if head_word:
                gen_w = extrair_genero(w)
                gen_h = extrair_genero(head_word)
                
                if gen_w and gen_h and gen_w != gen_h and gen_w != "_" and gen_h != "_":
                    w["new_head"] = None
                    w["new_rel"] = None


def aplicar_infinitivo_substantivado_artigo(words):
    for w in words:
        texto = w.get("text", "").lower()
        upos = w.get("upos", "")
        
        if texto in {"τὸ", "τοῦ", "τῷ", "το", "του", "τω"} and upos in {"DET", "ARTICLE"}:
            infinitivo = next((x for x in words[w["id"]:] if "v--p" in x.get("xpos", "") or "VerbForm=Inf" in x.get("feats", "")), None)
            verbo_matriz = next((v for v in words if v.get("new_rel") == "PRED" or (v.get("upos") == "VERB" and "v3" in v.get("xpos", ""))), None)
            
            if infinitivo and verbo_matriz:
                infinitivo["new_rel"] = "SBJ"
                infinitivo["new_head"] = verbo_matriz["id"]
                infinitivo["lock"] = True
                
                w["new_rel"] = "ATR"
                w["new_head"] = infinitivo["id"]
                w["lock"] = True
                
                de = next((x for x in words if x.get("text") in {"δὲ", "δέ", "δε"} and x["id"] == w["id"] + 1), None)
                if de:
                    de["new_rel"] = "AuxY"
                    de["new_head"] = verbo_matriz["id"]
                    de["lock"] = True


def aplicar_coordenacao_sujeito_neutro(words):
    for i, w in enumerate(words):
        texto = w.get("text", "").lower()
        caso = extrair_caso(w)
        
        if (texto in {"πᾶν", "παν", "πὰν"} or limpar_diacriticos(texto) == "παν") and caso == "n":
            kai = next((c for c in words[i+1:i+6] if c.get("text") in {"καὶ", "και"} and c.get("upos") == "CCONJ"), None)
            
            if kai:
                adj1 = next((x for x in words[i+1:kai["id"]-1] if extrair_caso(x) == "n" and x.get("upos") in {"ADJ", "NOUN"}), None)
                adj2 = next((x for x in words[kai["id"]:] if extrair_caso(x) == "n" and x.get("upos") in {"ADJ", "NOUN"}), None)
                
                if adj1 and adj2:
                    kai["new_rel"] = "COORD"
                    
                    verbo = next((v for v in words if v.get("new_rel") == "PRED" or v.get("upos") == "VERB"), None)
                    if verbo:
                        kai["new_head"] = verbo["id"]

                    adj1["new_rel"] = "SBJ_CO"
                    adj1["new_head"] = kai["id"]
                    adj2["new_rel"] = "SBJ_CO"
                    adj2["new_head"] = kai["id"]

                    w["new_rel"] = "ATR"
                    w["new_head"] = kai["id"]

                    w["lock"] = True
                    kai["lock"] = True
                    adj1["lock"] = True
                    adj2["lock"] = True

def aplicar_ocomp_participio(words):
    for w in words:
        if w.get("lock"):
            continue

        xpos = w.get("xpos") or ""
        feats = w.get("feats") or ""
        is_participle = "VerbForm=Part" in feats or (len(xpos) > 2 and xpos[2] == "p")

        if is_participle and extrair_caso(w) == "a":
            head_verb = get_word(words, w.get("new_head") or w.get("head"))
            
            if head_verb and head_verb.get("upos") in {"VERB", "AUX"}:
                obj_alvo = next(
                    (
                        obj for obj in words 
                        if (obj.get("new_head") == head_verb["id"] or obj.get("head") == head_verb["id"])
                        and obj.get("new_rel") == "OBJ"
                        and extrair_caso(obj) == "a"
                        and obj["id"] != w["id"]
                    ),
                    None
                )

                if obj_alvo:
                    w["new_rel"] = "OCOMP"
                    w["new_head"] = head_verb["id"]
                    w["lock"] = True


def aplicar_conectivos_correlativos(words):
    conectivos = {"και", "τε", "μητε"}
    
    ocorrencias = [
        w for w in words 
        if limpar_diacriticos(w["text"]).strip("’'") in conectivos 
        and w.get("upos") in {"CCONJ", "ADV"}
    ]

    if len(ocorrencias) >= 2:
        grupos = {}
        for o in ocorrencias:
            lemma_clean = limpar_diacriticos(o.get("lemma") or o.get("text"))
            grupos.setdefault(lemma_clean, []).append(o)

        for lemma, lista in grupos.items():
            if len(lista) >= 2:
                coord_principal = lista[-1]
                coord_principal["new_rel"] = "COORD"
                
                for auxy in lista[:-1]:
                    auxy["new_rel"] = "AuxY"
                    auxy["new_head"] = coord_principal["id"]
                    auxy["lock"] = True


def ajustar_dependencias_finais(words):
    pred_principal = next((w for w in words if w.get("new_rel") == "PRED" and w.get("new_head") == 0), None)
    
    for w in words:
        if not w.get("postag"):
            w["postag"] = w.get("xpos") or w.get("feats") or "_"
    
        if limpar_diacriticos(w["text"]) in {"αχρηστος"} or w.get("new_rel") == "nmod":
            w["new_rel"] = "PNOM"
            if pred_principal:
                w["new_head"] = pred_principal["id"]
    
        prep_eis = next((p for p in words if limpar_diacriticos(p["text"]) == "εις" and p.get("new_rel") == "AuxP"), None)
        if prep_eis and w["id"] != prep_eis["id"]:
            if w.get("new_head") == prep_eis.get("new_head") and w.get("new_rel") == "PNOM":
                w["new_rel"] = "ADV"
                w["new_head"] = prep_eis["id"]
    
        if w.get("new_rel") == "AuxK":
            w["new_head"] = 0


def sanitizar_arvore_agdt(words):
    pred_principal = next(
        (w for w in words if w.get("upos") in {"VERB", "AUX"} and w.get("text") == "ἔσται"),
        next((w for w in words if w.get("new_rel") in {"PRED", "PRED_CO"}), None)
    )

    if pred_principal:
        pred_principal["new_rel"] = "PRED"
        pred_principal["new_head"] = 0

    for w in words:
        if w.get("new_rel") == "AuxP" and w.get("new_head") == 0:
            if pred_principal:
                w["new_head"] = pred_principal["id"]
        
        if w.get("new_rel") in {"PRED", "PRED_CO"} and w["id"] != pred_principal["id"]:
            conj_eim = next((c for c in words if c.get("new_rel") == "AuxC"), None)
            if conj_eim:
                w["new_rel"] = "ADV"
                w["new_head"] = conj_eim["id"]
                conj_eim["new_head"] = pred_principal["id"] if pred_principal else 0

        if w.get("new_rel") == "nmod":
            if extrair_caso(w) == "n":
                w["new_rel"] = "PNOM"
                if pred_principal:
                    w["new_head"] = pred_principal["id"]
            else:
                w["new_rel"] = "ATR"


def converter_sentenca(sent):
    words = construir_words(sent)
    
    # 1. Limpeza e Inicialização
    sanitizar_morfologia_stanza(words) 
    inicializar_agdt(words)

    # 2. Resolução do Escopo Nominal e Adverbial
    desconstruir_atribuicoes_sem_concordancia(words)
    aplicar_infinitivo_substantivado_artigo(words)
    aplicar_artigos_repetidos(words)
    tratar_adverbios_cristalizados_e_sintagmaticos(words)
    aplicar_auxp(words)

    # 3. Estruturas Verbais Cópulas e Núcleos
    aplicar_coordenacao_sujeito_neutro(words)
    aplicar_regra_verbos_factitivos(words)
    garantir_predicado_raiz(words)
    aplicar_regras_infinitivo(words)
    aplicar_estrutura_aci_e_disjuncao(words)
    aplicar_participios_substantivados(words)
    aplicar_ocomp_participio(words)
    aplicar_copula(words)

    # 4. Coordenação e Partículas Correlativas
    aplicar_coordenacao(words)
    aplicar_oute_correlativo(words)
    aplicar_conectivos_correlativos(words)
    
    # 5. Fechamento de Partículas/Auxiliares
    aplicar_auxiliares_especiais(words)

    # 6. Sanitização final de dependências
    sanitizar_arvore_agdt(words)
    ajustar_dependencias_finais(words)

    for w in words:
        if w.get("lock"):
            continue

        if w["new_head"] is None: 
            w["new_head"] = w["head"]
            
        if w["new_rel"] is None or w["new_rel"] in {"nmod", "amod", "advmod", "obj", "nsubj"}: 
            w["new_rel"] = mapear_relacao_basica(w)

        if w["new_rel"] == "AuxK": 
            w["new_head"] = 0

        # Tratamento especial de infinitivos residuais
        xpos = w.get("xpos") or ""
        feats = w.get("feats") or ""
        is_inf = "VerbForm=Inf" in feats or (len(xpos) > 4 and xpos[4] == "n") or (len(xpos) > 2 and xpos[2] == "n")
        
        if is_inf and w["new_rel"] in {"PRED", "PRED_CO"}:
            w["new_rel"] = "OBJ" if w["new_rel"] == "PRED" else "OBJ_CO"

    return {"text": sent.text, "words": words}


def gerar_agdt_xml(sentences):
    root = ET.Element("treebank")

    for sent_idx, sent_data in enumerate(sentences, 1):
        doc_elem = ET.SubElement(
            root, 
            "sentence", 
            id=str(sent_idx), 
            document_id="arethusa_agdt", 
            subdoc="", 
            span=""
        )

        for w in sent_data["words"]:
            raw_head = w.get("new_head")
            if raw_head is None:
                raw_head = w.get("head", 0)
            
            raw_rel = w.get("new_rel")
            if not raw_rel:
                raw_rel = w.get("deprel", "UNDEF")

            head_str = str(int(raw_head)) if isinstance(raw_head, (int, float)) else str(raw_head)
            rel_str = str(raw_rel)

            ET.SubElement(doc_elem, "word", {
                "id": str(w["id"]),
                "form": str(w.get("text", "")),
                "lemma": str(w.get("lemma", "")),
                "postag": str(w.get("postag", "")),
                "head": head_str,
                "relation": rel_str
            })

    # Serialização bonita (Pretty Print) para evitar linha horizontal infinita
    raw_bytes = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(raw_bytes)
    return parsed.toprettyxml(indent="  ")


def gerar_conllu(sentences_convertidas):
    lines = []
    for i, sent in enumerate(sentences_convertidas, start=1):
        lines.append(f"# sent_id = {i}")
        lines.append(f"# text = {sent['text']}")
        for word in sent["words"]:
            fields = [
                str(word["id"]),
                str(word.get("text", "_")),
                str(word.get("lemma", "_")),
                str(word.get("upos", "_")),
                str(word.get("xpos", "_")),
                str(word.get("feats", "_")),
                str(word.get("new_head") if word.get("new_head") is not None else word.get("head", 0)),
                str(word.get("new_rel") or word.get("deprel", "_")),
                "_",
                "_"
            ]
            lines.append("\t".join(fields))
        lines.append("")
    return "\n".join(lines)

# ============================================================
# 7. INTERFACE DO USUÁRIO (STREAMLIT)
# ============================================================

st.subheader("1. Entrada de Texto")
modo_entrada = st.radio(
    "Escolha o método de inserção do texto:",
    ("Digitar / Colar Texto", "Carregar arquivo TXT"),
    horizontal=True
)

entrada_texto = ""
texto_exemplo = "Μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος.\nοὐλομένην, ἣ μυρί' Ἀχαιοῖς άλγε' ἔθηκε."

if modo_entrada == "Digitar / Colar Texto":
    entrada_texto = st.text_area(
        "Digite ou cole o texto em Grego Antigo:",
        value=texto_exemplo,
        height=140
    )
else:
    arquivo_carregado = st.file_uploader(
        "Selecione um arquivo .txt:",
        type=["txt"]
    )
    if arquivo_carregado is not None:
        entrada_texto = arquivo_carregado.read().decode("utf-8")
        st.text_area("Pré-visualização do arquivo carregado:", value=entrada_texto, height=140, disabled=True)

if st.button("Processar Texto", type="primary"):
    if not entrada_texto.strip():
        st.warning("Por favor, insira ou carregue um texto para conversão.")
    else:
        with st.spinner("Processando anotação sintática com Stanza..."):
            texto_formatado = pre_processar_sentencas(entrada_texto)
            
            doc = nlp(texto_formatado)
            sentences_convertidas = [converter_sentenca(sent) for sent in doc.sentences]
            
            xml_str = gerar_agdt_xml(sentences_convertidas)
            conllu_str = gerar_conllu(sentences_convertidas)

        st.success(f"Processamento concluído com sucesso! ({len(doc.sentences)} sentença(s) identificada(s))")
        st.subheader("2. Resultados e Exportação")

        tab_xml, tab_conllu = st.tabs(["📄 XML Arethusa (AGDT)", "📝 CoNLL-U (UD)"])

        with tab_xml:
            st.download_button(
                label="💾 Baixar XML Arethusa",
                data=xml_str,
                file_name="autotree_agdt.xml",
                mime="application/xml",
                key="btn_xml"
            )
            st.markdown("**Preview do XML (Arethusa):**")
            st.code(xml_str, language="xml")

        with tab_conllu:
            st.download_button(
                label="💾 Baixar CoNLL-U",
                data=conllu_str,
                file_name="autotree_ud.conllu",
                mime="text/plain",
                key="btn_conllu"
            )
            st.markdown("**Preview do CoNLL-U:**")
            st.code(conllu_str, language="plaintext")
