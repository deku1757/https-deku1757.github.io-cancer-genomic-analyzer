from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from io import BytesIO
import json
import os
from urllib.parse import quote
from urllib.request import urlopen, Request
import hashlib
import random

import pickle

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# In-memory database for demo (replace with actual database later)
users_db = {
    'demo': {
        'password': generate_password_hash('demo123'),
        'email': 'demo@genomics.app'
    }
}

# Lightweight in-memory assistant memory per user session.
# Format: {username: [{"role":"user|assistant","content":"...","ts":"..."}]}
assistant_memory = {}
MAX_MEMORY_MESSAGES = 12

# Cached load of model.pkl (or MODEL_PKL_PATH). Used by the assistant for context-aware predictions.
_ml_cached_obj = None
_ml_cached_path = None


def _default_model_pkl_path():
    base = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get('MODEL_PKL_PATH', os.path.join(base, 'model.pkl'))


# Typical column order for student/demo RandomForest classifiers (see model.feature_names_in_).
_ML_CANCER_RF_GENES = frozenset({
    'BRCA1', 'BRCA2', 'TP53', 'KRAS', 'EGFR', 'PIK3CA',
})


def _unwrap_inner_sklearn_object(model_obj):
    """Inner estimator or Pipeline (not applying scaler)."""
    obj = model_obj
    if isinstance(model_obj, dict):
        inner = (
            model_obj.get('model')
            or model_obj.get('clf')
            or model_obj.get('classifier')
            or model_obj.get('estimator')
            or model_obj.get('pipeline')
        )
        if inner is None and len(model_obj) == 1:
            inner = next(iter(model_obj.values()))
        obj = inner if inner is not None else model_obj
    return obj


def load_pkl_model():
    """
    Load model.pkl from project root or MODEL_PKL_PATH.
    Supports sklearn estimators/Pipelines, joblib dumps, and dict bundles with
    optional keys: model, scaler, clf, classifier.
    """
    global _ml_cached_obj, _ml_cached_path
    path = _default_model_pkl_path()
    if not os.path.isfile(path):
        return None, path
    if _ml_cached_obj is not None and _ml_cached_path == path:
        return _ml_cached_obj, path
    try:
        try:
            import joblib
            _ml_cached_obj = joblib.load(path)
        except Exception:
            with open(path, 'rb') as f:
                _ml_cached_obj = pickle.load(f)
        _ml_cached_path = path
        return _ml_cached_obj, path
    except Exception:
        return None, path


def _gene_label_to_score_map(analysis_data):
    labels = analysis_data.get('gene_labels') or []
    scores = analysis_data.get('gene_scores') or []
    return {str(lab).strip().upper(): float(sc) for lab, sc in zip(labels, scores)}


def _build_row_matching_feature_names(model_obj, analysis_data):
    """
    If the pickle exposes sklearn's feature_names_in_, build one row in that exact order.
    Matches common 15-column clinical+mutation schemas (BRCA1…PIK3CA, counts, age, etc.).
    """
    import numpy as np

    base = _unwrap_inner_sklearn_object(model_obj)
    names = getattr(base, 'feature_names_in_', None)
    if names is None or len(names) == 0:
        return None

    gsm = _gene_label_to_score_map(analysis_data)
    mut = list(analysis_data.get('mutation_distribution') or [])
    scores_list = [float(x) for x in (analysis_data.get('gene_scores') or [])]
    overall = float(analysis_data.get('overall_risk') or 0)

    row = []
    for raw_name in names:
        col = str(raw_name).strip()
        uk = col.upper().replace(' ', '_')

        if uk in _ML_CANCER_RF_GENES:
            val = float(gsm.get(uk, 0.0))
        elif uk == 'MUTATION_COUNT':
            val = float(np.sum(mut)) if mut else float(len(scores_list))
        elif uk == 'GENE_EXPRESSION':
            val = float(np.mean(scores_list)) if scores_list else overall
        elif uk == 'COPY_NUMBER_VARIATION':
            if len(scores_list) > 1:
                val = float(np.std(scores_list))
            elif mut:
                val = float(np.std(mut))
            else:
                val = overall * 0.1
        elif uk == 'AGE':
            v = analysis_data.get('patient_age')
            val = float(v) if v is not None else 55.0
        elif uk == 'GENDER':
            g = analysis_data.get('patient_gender')
            if g is None:
                val = 0.0
            else:
                gs = str(g).strip().lower()
                val = 1.0 if gs in ('1', 'm', 'male', 'man') else 0.0
        elif uk == 'FAMILY_HISTORY':
            v = analysis_data.get('family_history', analysis_data.get('patient_family_history', 0))
            if str(v).lower() in ('1', 'true', 'yes', 'y'):
                val = 1.0
            else:
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    val = 0.0
        elif uk == 'SMOKING':
            v = analysis_data.get('smoking', 0)
            if str(v).lower() in ('1', 'true', 'yes', 'y'):
                val = 1.0
            else:
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    val = 0.0
        elif uk == 'ALCOHOL':
            v = analysis_data.get('alcohol', 0)
            if str(v).lower() in ('1', 'true', 'yes', 'y'):
                val = 1.0
            else:
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    val = 0.0
        elif uk == 'BMI':
            v = analysis_data.get('patient_bmi', analysis_data.get('bmi'))
            val = float(v) if v is not None else 26.0
        else:
            val = float(analysis_data.get(col, analysis_data.get(uk.lower(), 0)) or 0)

        row.append(val)

    X = np.asarray(row, dtype=np.float64).reshape(1, -1)
    label = f"feature_names_in_ ({len(names)} cols: {', '.join(str(n) for n in names)})"
    return X, label


def _feature_matrix_variants(analysis_data):
    """Try multiple column layouts; training may match any of these."""
    import numpy as np

    scores = list(analysis_data.get('gene_scores') or [])
    mut = list(analysis_data.get('mutation_distribution') or [])
    overall = float(analysis_data.get('overall_risk') or 0)
    full = np.asarray(scores + mut + [overall], dtype=np.float64).reshape(1, -1)
    return [
        ('gene_scores+mutation_pct+overall_risk', full),
        ('gene_scores_only', np.asarray(scores, dtype=np.float64).reshape(1, -1)),
        ('gene_scores+overall_risk', np.asarray(scores + [overall], dtype=np.float64).reshape(1, -1)),
        ('mutation_pct_only', np.asarray(mut, dtype=np.float64).reshape(1, -1)),
    ]


def _resolve_estimator_and_X(model_obj, X):
    """Unpack dict-style bundles and apply scaler if present."""
    obj = model_obj
    if isinstance(model_obj, dict):
        scaler = model_obj.get('scaler') or model_obj.get('standard_scaler')
        inner = (
            model_obj.get('model')
            or model_obj.get('clf')
            or model_obj.get('classifier')
            or model_obj.get('estimator')
            or model_obj.get('pipeline')
        )
        if inner is None and len(model_obj) == 1:
            inner = next(iter(model_obj.values()))
        if inner is None:
            return None, X
        obj = inner
        if scaler is not None and hasattr(scaler, 'transform'):
            try:
                X = scaler.transform(X)
            except Exception:
                pass
    return obj, X


def _format_prediction(pred, proba):
    """Human-readable lines for chat / LLM context."""
    import numpy as np

    lines = []
    pred_arr = np.asarray(pred).ravel()
    lines.append(f"Raw prediction: {pred_arr.tolist()}")
    if proba is not None:
        p = np.asarray(proba)
        if p.ndim == 2:
            lines.append(f"Class probabilities (rows=sample): {np.round(p, 4).tolist()}")
        else:
            lines.append(f"Probability output: {np.round(p, 4).tolist()}")
    return '\n'.join(lines)


def run_ml_model_prediction(analysis_data):
    """
    Run model.pkl on features derived from analysis_data.
    Returns a dict: loaded, path, summary, detail, error (optional), feature_variant (optional).
    """
    try:
        import numpy as np
    except ImportError:
        return {
            'loaded': False,
            'path': _default_model_pkl_path(),
            'summary': None,
            'detail': None,
            'error': 'numpy is required for model.pkl. Run: pip install numpy scikit-learn',
            'feature_variant': None,
        }

    model_obj, path = load_pkl_model()
    out = {
        'loaded': model_obj is not None,
        'path': path,
        'summary': None,
        'detail': None,
        'error': None,
        'feature_variant': None,
    }
    if model_obj is None:
        out['error'] = 'model.pkl not found or failed to load (place model.pkl next to app.py or set MODEL_PKL_PATH).'
        return out

    named = _build_row_matching_feature_names(model_obj, analysis_data)
    variants = []
    if named:
        variants.append((named[1], named[0]))
    variants.extend(_feature_matrix_variants(analysis_data))

    last_err = None
    for variant_name, X in variants:
        try:
            estimator, Xp = _resolve_estimator_and_X(model_obj, X)
            if estimator is None:
                out['error'] = 'Could not resolve an estimator inside the pickle.'
                return out

            if not hasattr(estimator, 'predict'):
                out['error'] = 'Loaded object has no predict(); expected a sklearn-like model.'
                return out

            pred = estimator.predict(Xp)
            proba = None
            if hasattr(estimator, 'predict_proba'):
                try:
                    proba = estimator.predict_proba(Xp)
                except Exception:
                    proba = None

            detail = _format_prediction(pred, proba)
            out['detail'] = detail
            out['feature_variant'] = variant_name
            pv = np.asarray(pred).ravel()

            class_labels = None
            if hasattr(estimator, 'classes_'):
                try:
                    class_labels = [str(c) for c in np.asarray(estimator.classes_).ravel()]
                except Exception:
                    class_labels = None

            first = pv.flat[0] if pv.size else None
            stringish = (
                first is not None
                and (
                    isinstance(first, str)
                    or (pv.dtype.kind in ('U', 'S'))
                    or (pv.dtype == object and not isinstance(first, (float, int, np.floating, np.integer)))
                )
            )
            if stringish:
                lab = str(first)
                out['predicted_label'] = lab
                out['summary'] = f"predicted risk band (model): {lab}"
            elif pv.size == 1 and np.issubdtype(pv.dtype, np.number):
                val = float(pv[0])
                is_int_like = abs(val - round(val)) < 1e-9
                if class_labels and is_int_like and 0 <= int(round(val)) < len(class_labels):
                    lab = class_labels[int(round(val))]
                    out['predicted_label'] = lab
                    out['summary'] = f"predicted risk band (model): {lab}"
                else:
                    out['summary'] = (
                        f"predicted class/index: {int(round(val))}"
                        if is_int_like
                        else f"model output ≈ {val:.4f}"
                    )
            else:
                out['summary'] = f"prediction: {pv.tolist()}"

            if proba is not None and class_labels:
                p = np.asarray(proba).ravel()
                if p.size == len(class_labels):
                    pairs = sorted(zip(class_labels, p), key=lambda x: -x[1])
                    top = ', '.join(f"{a}: {b:.3f}" for a, b in pairs[:3])
                    out['summary'] += f" (probabilities — {top})"
            elif proba is not None:
                p = np.asarray(proba).ravel()
                if p.size:
                    out['summary'] += f"; max prob ≈ {float(np.max(p)):.3f}"

            return out
        except Exception as ex:
            last_err = ex
            continue

    out['error'] = (
        f'ML inference failed ({type(last_err).__name__}: {last_err}). '
        'Feature count/order must match training; set MODEL_PKL_PATH or adjust feature engineering.'
    )
    return out


def ml_context_for_llm(ml_result):
    """Compact text block for Gemini / prompts."""
    if not ml_result:
        return ''
    if ml_result.get('error') and not ml_result.get('summary'):
        return f"ML model status: {ml_result['error']}"
    parts = []
    if ml_result.get('summary'):
        parts.append(f"Summary: {ml_result['summary']}")
    if ml_result.get('detail'):
        parts.append(ml_result['detail'])
    if ml_result.get('error'):
        parts.append(f"Note: {ml_result['error']}")
    return 'Local ML model (model.pkl):\n' + '\n'.join(parts)


def ml_reply_suffix(ml_result):
    """Short line for non-LLM fallback replies."""
    if not ml_result:
        return ''
    if ml_result.get('summary'):
        name = os.path.basename(ml_result.get('path') or 'model.pkl')
        extra = f" [input layout: {ml_result['feature_variant']}]" if ml_result.get('feature_variant') else ''
        return f"\n\nML model ({name}): {ml_result['summary']}{extra}"
    if ml_result.get('error'):
        return f"\n\n(ML model: {ml_result['error']})"
    return ''


def _rng_for_analysis(analysis_id, session_username=None):
    """Stable pseudo-random generator: same analysis_id (+ user) → same mock numbers."""
    parts = [str(analysis_id or 'analysis_demo')]
    if session_username:
        parts.append(str(session_username))
    digest = hashlib.sha256('\0'.join(parts).encode('utf-8')).digest()
    return random.Random(int.from_bytes(digest[:8], 'big'))


def _mutation_shares_sum_100(rng, n):
    """Return n nonnegative integers that sum to 100."""
    if n < 1:
        return []
    if n == 1:
        return [100]
    cut = sorted(rng.sample(range(1, 100), n - 1))
    edges = [0] + cut + [100]
    return [edges[i + 1] - edges[i] for i in range(n)]


def get_mock_analysis_data(analysis_id='analysis_demo', session_username=None):
    """
    Centralized analysis data used by pages and assistant.
    Values are deterministic from analysis_id. For the default demo id only,
    username is mixed into the seed so the Assistant on the home page is not
    identical for every account. Concrete /results/<id> URLs stay the same for
    all viewers with that id.
    """
    aid = str(analysis_id or 'analysis_demo')
    user_for_seed = session_username if aid == 'analysis_demo' else None
    rng = _rng_for_analysis(aid, user_for_seed)
    labels = ['BRCA1', 'BRCA2', 'TP53', 'PTEN', 'MLH1']
    gene_scores = [rng.randint(48, 98) for _ in labels]
    mutation_distribution = _mutation_shares_sum_100(rng, len(labels))
    overall_risk = int(
        max(
            24,
            min(
                92,
                round(0.55 * max(gene_scores) + 0.15 * sum(gene_scores) / len(gene_scores))
                + rng.randint(-10, 10),
            ),
        )
    )

    base_recs = [
        'Schedule genetic counseling with a certified genetic counselor',
        'Increase cancer screening frequency to every 6-12 months',
        'Consider preventive measures based on your risk profile',
        'Share results with immediate family members for family planning',
        'Maintain healthy lifestyle with regular exercise and healthy diet',
        'Keep detailed medical records and family health history',
    ]
    recommendations = base_recs[:]
    rng.shuffle(recommendations)

    return {
        'analysis_id': aid,
        'analysis_date': datetime.now().strftime('%B %d, %Y'),
        'overall_risk': overall_risk,
        'gene_labels': labels,
        'gene_scores': gene_scores,
        'mutation_distribution': mutation_distribution,
        'patient_age': rng.randint(38, 76),
        'patient_gender': rng.choice(['female', 'male']),
        'patient_bmi': round(20.5 + rng.random() * 16.0, 1),
        'family_history': rng.randint(0, 1),
        'smoking': rng.randint(0, 1),
        'alcohol': rng.randint(0, 1),
        'recommendations': recommendations,
    }


# Educational mapping: high-risk gene signals → cancer types often discussed in genetics (not diagnostic).
_GENE_TO_CANCER_CONTEXT = {
    'BRCA1': (
        'hereditary breast and ovarian cancer (HBOC) risk discussion',
        'breast, ovarian, and (in some families) prostate or pancreatic cancer screening conversations',
    ),
    'BRCA2': (
        'HBOC-related hereditary risk patterns',
        'breast, ovarian, prostate, and sometimes pancreatic cancer in genetics consults',
    ),
    'TP53': (
        'Li-Fraumeni syndrome-related tumor spectrum when pathogenic variants are confirmed',
        'sarcoma, breast cancer, brain tumors, adrenocortical cancer, and others',
    ),
    'PTEN': (
        'PTEN tumor predisposition (e.g. Cowden / PHTS) contexts',
        'breast, thyroid, endometrial, and kidney cancers are commonly reviewed',
    ),
    'MLH1': (
        'Lynch syndrome (mismatch-repair) hereditary colorectal cancer family',
        'colorectal, endometrial, and other Lynch-associated cancers',
    ),
    'KRAS': (
        'MAPK pathway alterations often studied in solid tumors',
        'colorectal, lung, and pancreatic tumor research and targeted therapy discussions',
    ),
    'EGFR': (
        'growth-factor pathway changes common in some carcinomas',
        'non-small cell lung cancer and other EGFR-driven solid tumors (research / treatment context)',
    ),
    'PIK3CA': (
        'PI3K pathway alterations in several epithelial cancers',
        'breast, endometrial, and colorectal cancers among examples in the literature',
    ),
}


def build_cancer_prediction_insight(analysis_data, ml_result=None):
    """
    Human-readable "predicted cancer type" style summary for the results page.
    Based on top gene scores + optional ML risk band. Strictly educational / non-diagnostic.
    """
    labels = analysis_data.get('gene_labels') or []
    scores = analysis_data.get('gene_scores') or []
    pairs = sorted(
        zip(labels, scores),
        key=lambda x: float(x[1]) if x[1] is not None else 0,
        reverse=True,
    )
    top = pairs[:3] if pairs else []

    parts = []
    for gene, score in top:
        g = str(gene).strip().upper()
        ctx = _GENE_TO_CANCER_CONTEXT.get(
            g,
            (
                'broad hereditary or somatic cancer-risk discussions',
                'several organ sites depending on the variant, family history, and clinical work-up',
            ),
        )
        parts.append(
            f"{g} (risk score {int(round(float(score)))}%) - often discussed in the context of "
            f"{ctx[0]}, with emphasis on {ctx[1]}."
        )

    paragraph = ' '.join(parts) if parts else (
        'No gene-level scores were available to summarize. Discuss any concerns with a genetic counselor.'
    )

    ml_note = ''
    if ml_result and ml_result.get('predicted_label'):
        ml_note = (
            f"The risk stratification model placed this profile in the {ml_result['predicted_label']} band "
            f"(e.g. relative High, Medium, or Low in the training labels - informational only, not a diagnosis)."
        )
    elif ml_result and ml_result.get('summary') and not ml_result.get('error'):
        ml_note = f"Model summary: {ml_result['summary']}"

    headline = 'Predicted cancer-type focus from your gene profile'
    sub = (
        f"Overall composite risk in this report is {analysis_data.get('overall_risk', 0)}%. "
        "The patterns below reflect where similar variant profiles are usually discussed in oncology and genetics - "
        "they do not name a cancer you have or will get."
    )

    return {
        'headline': headline,
        'sub': sub,
        'paragraph': paragraph,
        'ml_note': ml_note,
        'top_gene': top[0][0] if top else None,
        'top_gene_score': int(round(float(top[0][1]))) if top else None,
    }


def _strip_leading_article(phrase):
    """Turn 'a gene' / 'the DNA' into 'gene' / 'DNA' for Wikipedia titles."""
    s = (phrase or '').strip()
    low = s.lower()
    for art in ('a ', 'an ', 'the '):
        if low.startswith(art):
            return s[len(art):].strip()
    return s


def _general_query_variants(question):
    """
    Wikipedia summary API expects a page title (e.g. 'DNA'), not 'what is DNA'.
    Produce a short list of candidates: original, stripped question patterns, etc.
    """
    q = (question or '').strip()
    if not q:
        return []
    variants = []
    base = q.rstrip('?').strip()

    def add(v):
        v = (v or '').strip().rstrip('?')
        if not v:
            return
        variants.append(v)

    add(base)
    lower = base.lower()
    for p in (
        'what is ', 'what are ', "what's ", 'who is ', 'who are ', "who's ",
        'where is ', 'where are ', "where's ",
        'when is ', 'why is ', 'how is ',
        'define ', 'tell me about ', 'explain ', 'describe ',
    ):
        if lower.startswith(p):
            rest = _strip_leading_article(base[len(p):].strip())
            add(rest)
            break

    seen = set()
    out = []
    for v in variants:
        k = v.lower()
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def _local_glossary_answer(question):
    """Tiny offline fallback for very common genomics / biology questions."""
    t = (question or '').strip().lower().rstrip('?')
    for p in (
        'what is ', 'what are ', "what's ", 'define ', 'explain ', 'describe ',
        'tell me about ',
    ):
        if t.startswith(p):
            t = t[len(p):].strip()
            break
    t = _strip_leading_article(t).lower().strip()
    snippets = {
        'dna': (
            'DNA (deoxyribonucleic acid) stores genetic information as a sequence of bases '
            '(adenine, thymine, cytosine, guanine). In cells it usually forms a double helix and '
            'is copied when cells divide. This is general biology information, not medical advice.'
        ),
        'rna': (
            'RNA (ribonucleic acid) is related to DNA and helps translate genetic information into '
            'proteins (e.g. mRNA, tRNA, rRNA). This is general biology information, not medical advice.'
        ),
        'gene': (
            'A gene is a segment of DNA that typically codes for a functional product (often a protein). '
            'Variants in genes can affect traits or disease risk. This is general information, not a diagnosis.'
        ),
        'mutation': (
            'A mutation is a change in the DNA sequence compared with a reference. Some mutations are harmless; '
            'others can affect protein function or disease risk. Clinical meaning depends on context and testing.'
        ),
        'chromosome': (
            'A chromosome is a large packaged DNA molecule. Humans usually have 23 pairs. '
            'General biology information only.'
        ),
        'vcf': (
            'VCF (Variant Call Format) is a text file format for describing genetic variants (e.g. SNPs, indels) '
            'often produced by sequencing pipelines. You can upload or paste VCF data on the Analyze page here.'
        ),
        'brca1': (
            'BRCA1 is a human gene involved in DNA repair. Inherited variants can affect cancer risk; '
            'interpretation needs genetic counseling and clinical context—not something this app can diagnose.'
        ),
        'brca2': (
            'BRCA2 is a human gene involved in DNA repair. Like BRCA1, variants may affect cancer risk; '
            'results require professional interpretation.'
        ),
        'tp53': (
            'TP53 encodes the protein p53, important for cell cycle control and tumor suppression. '
            'Variants are interpreted in a clinical/genetic counseling setting.'
        ),
        'genomics': (
            'Genomics is the study of genomes—the full DNA sequence—and how genes and variants work together. '
            'This app focuses on exploring variant-style summaries in a demo/educational way.'
        ),
    }
    return snippets.get(t)


def _fetch_wikipedia_summary(title):
    try:
        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
        req = Request(wiki_url, headers={'User-Agent': 'CancerGenomicsAssistant/1.0'})
        with urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode('utf-8'))
            if payload.get('type') == 'disambiguation':
                return None
            extract = (payload.get('extract') or '').strip()
            return extract or None
    except Exception:
        return None


def _fetch_duckduckgo_instant(q):
    try:
        ddg_url = (
            'https://api.duckduckgo.com/?q='
            f"{quote(q)}&format=json&no_html=1&skip_disambig=1"
        )
        req = Request(ddg_url, headers={'User-Agent': 'CancerGenomicsAssistant/1.0'})
        with urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode('utf-8'))
            abstract = (payload.get('AbstractText') or '').strip()
            if abstract:
                return abstract
            related = payload.get('RelatedTopics') or []
            for item in related:
                text = (item.get('Text') or '').strip() if isinstance(item, dict) else ''
                if text:
                    return text
    except Exception:
        pass
    return None


def fetch_general_answer(question):
    """
    Try to answer broad/general questions using public knowledge endpoints.
    Falls back gracefully if external sources are unavailable.
    """
    q = (question or '').strip()
    if not q:
        return None

    variants = _general_query_variants(q)

    # 1) Wikipedia — try each title candidate (fixes 'what is DNA' -> 'DNA')
    for title in variants:
        if len(title) < 1:
            continue
        extract = _fetch_wikipedia_summary(title)
        if extract:
            return extract

    # 2) DuckDuckGo — full question then shorter variants
    for qq in variants:
        hit = _fetch_duckduckgo_instant(qq)
        if hit:
            return hit

    # 3) Small built-in glossary (works offline / when APIs are empty)
    gloss = _local_glossary_answer(q)
    if gloss:
        return gloss

    return None


def fetch_gemini_answer(user_message, username, current_path, analysis_data, memory, ml_context=''):
    """
    Call Gemini API directly using REST.
    Set environment variable GEMINI_API_KEY to enable.
    """
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not api_key:
        return None

    # Keep memory short to reduce prompt size and latency
    memory_tail = memory[-8:] if memory else []
    memory_lines = []
    for item in memory_tail:
        role = item.get('role', 'user')
        content = item.get('content', '')
        memory_lines.append(f"{role.upper()}: {content}")
    memory_block = "\n".join(memory_lines) if memory_lines else "No previous conversation."

    context_prompt = (
        "You are 'Assistant' for a Cancer Genomics Analyzer web app.\n"
        "Rules:\n"
        "1) Be concise, clear, and friendly.\n"
        "2) For medical topics, include a brief safety note that this is informational.\n"
        "3) If user asks app-specific questions, prioritize app context below.\n"
        "4) If user asks general questions, answer normally.\n"
        "5) Never claim actions you did not perform.\n"
        "6) If a Local ML model output is present, interpret it together with the app's metrics; "
        "state clearly that it is informational model output, not a medical diagnosis.\n\n"
        f"User: {username}\n"
        f"Current page path: {current_path or 'unknown'}\n"
        f"Analysis ID: {analysis_data.get('analysis_id')}\n"
        f"Overall risk: {analysis_data.get('overall_risk')}%\n"
        f"Gene labels: {', '.join(analysis_data.get('gene_labels', []))}\n"
        f"Gene scores: {analysis_data.get('gene_scores', [])}\n"
        f"Mutation distribution: {analysis_data.get('mutation_distribution', [])}\n"
        f"Recommendations: {analysis_data.get('recommendations', [])}\n\n"
    )
    if ml_context:
        context_prompt += f"{ml_context}\n\n"
    context_prompt += (
        f"Recent memory:\n{memory_block}\n\n"
        f"User question: {user_message}"
    )

    try:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-1.5-flash:generateContent?key={api_key}"
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": context_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 400
            }
        }

        req = Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'CancerAssistant/1.0'
            },
            method='POST'
        )
        with urlopen(req, timeout=12) as response:
            data = json.loads(response.read().decode('utf-8'))
            candidates = data.get('candidates') or []
            if not candidates:
                return None
            parts = candidates[0].get('content', {}).get('parts', [])
            text_chunks = [p.get('text', '') for p in parts if p.get('text')]
            answer = "\n".join(text_chunks).strip()
            return answer or None
    except Exception:
        return None

# ============= AUTHENTICATION ROUTES =============

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not username or not email or not password:
            return render_template('register.html', error='All fields required')
        
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')
        
        if username in users_db:
            return render_template('register.html', error='Username already exists')
        
        # Register user
        users_db[username] = {
            'password': generate_password_hash(password),
            'email': email
        }
        
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in users_db and check_password_hash(users_db[username]['password'], password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """User logout"""
    session.pop('username', None)
    return redirect(url_for('home'))

# ============= MAIN ROUTES =============

@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')

@app.route('/analyze')
def analyze():
    """Analysis page"""
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('analyze.html')

# ============= NEW: RESULTS PAGE ROUTE =============

@app.route('/results/<analysis_id>')
def show_results(analysis_id):
    """Display analysis results with interactive charts"""
    data = get_mock_analysis_data(analysis_id)
    ml_snapshot = run_ml_model_prediction(data)
    cancer_insight = build_cancer_prediction_insight(data, ml_snapshot)
    return render_template('results.html', cancer_insight=cancer_insight, **data)

# ============= API ROUTES =============

@app.route('/api/analysis/<analysis_id>')
def get_analysis_json(analysis_id):
    """API endpoint to get analysis data as JSON"""
    analysis_data = get_mock_analysis_data(analysis_id)
    return jsonify(analysis_data)


@app.route('/api/assistant', methods=['POST'])
def assistant_chat():
    """Backend assistant endpoint with page-aware context."""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    payload = request.get_json(silent=True) or {}
    message = (payload.get('message') or '').strip()
    current_path = (payload.get('path') or '').strip()
    analysis_id = (payload.get('analysis_id') or '').strip()

    if not message:
        return jsonify({'reply': 'Please type a question so I can help you.'})

    lower = message.lower()
    username = session.get('username', 'User')
    memory = assistant_memory.get(username, [])
    analysis_data = get_mock_analysis_data(
        analysis_id or 'analysis_demo',
        session_username=username,
    )

    gene_scores = dict(zip(analysis_data['gene_labels'], analysis_data['gene_scores']))
    sorted_genes = sorted(gene_scores.items(), key=lambda x: x[1], reverse=True)
    top_gene, top_gene_score = sorted_genes[0]
    overall_risk = analysis_data['overall_risk']

    ml_result = run_ml_model_prediction(analysis_data)
    ml_block = ml_context_for_llm(ml_result)

    # Prefer Gemini LLM when API key is available.
    llm_reply = fetch_gemini_answer(
        user_message=message,
        username=username,
        current_path=current_path,
        analysis_data=analysis_data,
        memory=memory,
        ml_context=ml_block,
    )
    if llm_reply:
        reply = llm_reply
    else:
        # Fallback deterministic/local logic if Gemini is unavailable.
        if any(word in lower for word in ['hello', 'hi', 'hey']):
            reply = (
                f"Hi {username}! I am your Assistant. I can help with login, VCF analysis steps, "
                "risk interpretation, and export options."
            )
        elif any(phrase in lower for phrase in ['what is your name', "what's your name", 'who are you', 'your name']):
            reply = "My name is Assistant. I am your cancer genomics helper and general Q&A companion."
        elif any(phrase in lower for phrase in ['how are you', 'how are u', 'how r you']):
            reply = "I am doing great, thank you! Ready to help with serious or silly questions."
        elif any(phrase in lower for phrase in ['who made you', 'who created you', 'who built you']):
            reply = "I was configured as your in-app Assistant for this Cancer Genomics Analyzer project."
        elif any(phrase in lower for phrase in ['thank you', 'thanks', 'thx']):
            reply = "You are welcome! I am happy to help."
        elif any(phrase in lower for phrase in ['joke', 'tell me a joke', 'funny']):
            reply = "Why did the cell go to therapy? It had too many unresolved divisions."
        elif any(phrase in lower for phrase in ['who am i', 'my name']):
            reply = f"You are logged in as {username}."
        elif any(word in lower for word in ['help', 'what can you do', 'capabilities']):
            reply = (
                "I can help you with:\n"
                "1) Uploading/pasting VCF data on Analyze page\n"
                "2) Understanding risk score and gene-level risk\n"
                "3) Explaining mutation distribution\n"
                "4) Downloading PDF/CSV reports\n"
                "5) Basic genomics terms (BRCA1, BRCA2, TP53, PTEN, MLH1)"
            )
        elif any(word in lower for word in ['risk', 'overall']):
            level = 'high' if overall_risk > 70 else 'medium' if overall_risk > 40 else 'low'
            reply = (
                f"Current overall risk is {overall_risk}% ({level.upper()}). "
                f"Highest gene risk in the current dataset is {top_gene} at {top_gene_score}%."
            )
            reply += ml_reply_suffix(ml_result)
        elif any(gene.lower() in lower for gene in analysis_data['gene_labels']):
            selected = None
            for gene in analysis_data['gene_labels']:
                if gene.lower() in lower:
                    selected = gene
                    break
            score = gene_scores.get(selected, 0)
            status = 'high' if score >= 80 else 'medium' if score >= 50 else 'low'
            reply = f"{selected} risk score is {score}% ({status.upper()} risk in this dataset)."
            reply += ml_reply_suffix(ml_result)
        elif any(word in lower for word in ['analyze', 'upload', 'vcf']):
            reply = (
                "To run analysis: go to Analyze, upload a .vcf/.txt file or paste VCF text, "
                "then click 'Analyze VCF Data'. I can explain each option before you submit."
            )
        elif any(word in lower for word in ['export', 'download', 'pdf', 'csv']):
            aid = analysis_id or 'analysis_demo'
            reply = (
                f"You can download reports from Results page using:\n"
                f"- /export/{aid}/pdf\n"
                f"- /export/{aid}/csv"
            )
        elif any(word in lower for word in ['recommendation', 'next step', 'what should i do']):
            recs = analysis_data.get('recommendations', [])[:3]
            reply = "Top recommendations:\n- " + "\n- ".join(recs)
            reply += ml_reply_suffix(ml_result)
        else:
            # If the user asks a short follow-up ("and this?", "why?"), enrich query with previous turn.
            recent_user_messages = [m['content'] for m in memory if m.get('role') == 'user']
            previous_user_message = recent_user_messages[-1] if recent_user_messages else ''
            is_short_followup = len(message.split()) <= 5
            general_query = (
                f"{previous_user_message}. Follow-up: {message}"
                if previous_user_message and is_short_followup
                else message
            )

            general_answer = fetch_general_answer(general_query)
            if general_answer:
                reply = general_answer
            else:
                context_hint = f" (page: {current_path})" if current_path else ""
                reply = (
                    "I can answer both app questions and general questions, but I could not fetch a reliable "
                    f"general answer right now{context_hint}. Please try rephrasing your question."
                )

    # Persist conversational memory per user
    timestamp = datetime.now().isoformat()
    memory.append({'role': 'user', 'content': message, 'ts': timestamp})
    memory.append({'role': 'assistant', 'content': reply, 'ts': timestamp})
    assistant_memory[username] = memory[-MAX_MEMORY_MESSAGES:]

    return jsonify({
        'reply': reply,
        'ml': {
            'loaded': ml_result.get('loaded'),
            'summary': ml_result.get('summary'),
            'predicted_label': ml_result.get('predicted_label'),
            'error': ml_result.get('error'),
            'feature_variant': ml_result.get('feature_variant'),
            'model_file': os.path.basename(ml_result.get('path') or 'model.pkl'),
        },
        'context': {
            'overall_risk': overall_risk,
            'top_gene': top_gene,
            'top_gene_score': top_gene_score,
            'memory_size': len(assistant_memory.get(username, []))
        }
    })


@app.route('/api/assistant/memory', methods=['GET', 'DELETE'])
def assistant_memory_api():
    """Inspect or clear assistant memory for current logged-in user."""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    username = session.get('username')
    if request.method == 'DELETE':
        assistant_memory[username] = []
        return jsonify({'ok': True, 'message': 'Assistant memory cleared.'})

    memory = assistant_memory.get(username, [])
    return jsonify({
        'count': len(memory),
        'memory': memory
    })

# ============= EXPORT ROUTES =============

def _export_payload_from_analysis(analysis_id):
    """Same numbers as results page / API for this analysis_id."""
    src = get_mock_analysis_data(analysis_id)
    return {
        'patient_name': 'Sample Patient',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'risk_score': src['overall_risk'],
        'gene_risks': dict(zip(src['gene_labels'], src['gene_scores'])),
        'recommendations': list(src['recommendations']),
    }


@app.route('/export/<analysis_id>/<format>')
def export_results(analysis_id, format):
    """Export analysis results in different formats"""
    analysis_data = _export_payload_from_analysis(analysis_id)
    
    try:
        if format == 'pdf':
            return export_pdf(analysis_data, analysis_id)
        elif format == 'csv':
            return export_csv(analysis_data, analysis_id)
        elif format == 'json':
            return jsonify(analysis_data)
        else:
            return jsonify({'error': 'Invalid format. Use: pdf, csv, or json'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def export_pdf(data, analysis_id):
    """Generate PDF report (requires reportlab)"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Custom title style
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#00d969'),
            spaceAfter=30,
            alignment=1,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#0099cc'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        # Title
        title = Paragraph("🦖 Cancer Genomics Analysis Report", title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))
        
        # Patient Information Table
        elements.append(Paragraph("Patient Information", heading_style))
        
        patient_data = [
            ['Field', 'Value'],
            ['Name', data.get('patient_name', 'Not provided')],
            ['Date of Analysis', data.get('date', datetime.now().strftime('%Y-%m-%d'))],
            ['Overall Risk Score', f"{data.get('risk_score', 0)}%"]
        ]
        
        patient_table = Table(patient_data, colWidths=[2*inch, 3*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00d969')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
        ]))
        elements.append(patient_table)
        elements.append(Spacer(1, 0.4*inch))
        
        # Gene Risk Assessment
        elements.append(Paragraph("Gene Risk Assessment", heading_style))
        
        gene_risks = data.get('gene_risks', {})
        gene_data = [['Gene', 'Risk Score', 'Status']]
        
        for gene, risk in gene_risks.items():
            if risk > 80:
                status = 'HIGH RISK'
            elif risk > 50:
                status = 'MEDIUM RISK'
            else:
                status = 'LOW RISK'
            gene_data.append([gene, f"{risk}%", status])
        
        gene_table = Table(gene_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
        gene_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0099cc')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black)
        ]))
        elements.append(gene_table)
        elements.append(Spacer(1, 0.4*inch))
        
        # Recommendations
        elements.append(Paragraph("Clinical Recommendations", heading_style))
        
        recommendations = data.get('recommendations', [])
        rec_text = "<br/>".join([f"<b>•</b> {rec}" for rec in recommendations])
        elements.append(Paragraph(rec_text, styles['Normal']))
        
        elements.append(Spacer(1, 0.5*inch))
        
        # Disclaimer
        disclaimer = Paragraph(
            "<b>DISCLAIMER:</b> This analysis is for informational purposes only and should not replace "
            "professional medical advice. Please consult with a qualified healthcare provider.",
            ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=9, textColor=colors.red)
        )
        elements.append(disclaimer)
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Footer
        footer = Paragraph(
            f"<i>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</i>",
            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, alignment=1)
        )
        elements.append(footer)
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'analysis_{analysis_id}_{datetime.now().strftime("%Y%m%d")}.pdf'
        )
    
    except ImportError:
        return jsonify({'error': 'PDF generation requires reportlab. Install with: pip install reportlab'}), 400

def export_csv(data, analysis_id):
    """Generate CSV report"""
    import csv
    
    output = BytesIO()
    output_string = []
    
    writer = csv.writer(output_string)
    
    # Header section
    writer.writerow(['CANCER GENOMICS ANALYSIS REPORT'])
    writer.writerow([])
    writer.writerow(['PATIENT INFORMATION'])
    writer.writerow(['Date of Analysis', data.get('date', datetime.now().strftime('%Y-%m-%d'))])
    writer.writerow(['Patient Name', data.get('patient_name', 'Not provided')])
    writer.writerow(['Overall Risk Score', f"{data.get('risk_score', 0)}%"])
    writer.writerow([])
    writer.writerow([])
    
    # Gene Risk Data
    writer.writerow(['GENE RISK ASSESSMENT'])
    writer.writerow(['Gene', 'Risk Score (%)', 'Status'])
    
    gene_risks = data.get('gene_risks', {})
    for gene, risk in gene_risks.items():
        if risk > 80:
            status = 'HIGH'
        elif risk > 50:
            status = 'MEDIUM'
        else:
            status = 'LOW'
        writer.writerow([gene, risk, status])
    
    writer.writerow([])
    writer.writerow([])
    
    # Recommendations
    writer.writerow(['CLINICAL RECOMMENDATIONS'])
    recommendations = data.get('recommendations', [])
    for rec in recommendations:
        writer.writerow([rec])
    
    writer.writerow([])
    writer.writerow(['DISCLAIMER:'])
    writer.writerow(['This analysis is for informational purposes only.'])
    writer.writerow(['Please consult with a qualified healthcare provider.'])
    
    # Combine into string
    csv_data = '\n'.join(''.join(row) if isinstance(row, str) else ','.join(map(str, row)) for row in output_string)
    
    return send_file(
        BytesIO(csv_data.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'analysis_{analysis_id}_{datetime.now().strftime("%Y%m%d")}.csv'
    )

# ============= ERROR HANDLERS =============

@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500

# ============= RUN APP =============

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)