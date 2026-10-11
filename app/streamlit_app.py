import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.final_model import load_final, DEFAULT_PATH
from src.data import load_dataset
from src.score import score_claim, explain_claim, zone_for
from src.costs import bayes_threshold  # noqa: F401  (kept for parity with tier 1)

st.set_page_config(page_title="Claim risk triage", layout="wide")


@st.cache_resource
def get_bundle():
    return load_final(DEFAULT_PATH)


@st.cache_data
def get_data():
    X, y, num_cols, cat_cols = load_dataset(verbose=False)
    return X, y


bundle = get_bundle()
try:
    X_all, y_all = get_data()
except Exception as e:
    X_all = y_all = None
    st.warning(f"Raw data not found, so defaults and replay are unavailable: {e}")

st.title("Claim risk triage (demo)")
st.caption("Costs are hypothetical. SHAP reasons are model attributions, not causes of fraud. "
           "Features are pre-binned, so reasons are coarse.")

L = st.sidebar.slider("Missed-fraud loss L (investigation cost c = 1)", 2.0, 100.0, float(bundle["L"]))
c = float(bundle["c"])
t = c / (L + c)
st.sidebar.write(f"Threshold: {t:.3f}. Review at or above this, investigate at 2x it ({2*t:.3f}).")

tab1, tab2 = st.tabs(["Score a claim", "Replay held-out claims"])

with tab1:
    defaults = {}
    if X_all is not None:
        for col in bundle["num_cols"]:
            defaults[col] = float(X_all[col].median())
        for col in bundle["cat_cols"]:
            defaults[col] = X_all[col].mode().iloc[0]
    with st.form("claim"):
        row = {}
        cols = st.columns(3)
        for i, f in enumerate(bundle["feature_list"]):
            with cols[i % 3]:
                if f in bundle["cat_cols"]:
                    opts = [v for v in bundle["category_maps"][f] if v is not None]
                    idx = opts.index(defaults[f]) if f in defaults and defaults[f] in opts else 0
                    row[f] = st.selectbox(f, opts, index=idx)
                else:
                    row[f] = st.number_input(f, value=defaults.get(f, 0.0))
        go = st.form_submit_button("Score claim")
    if go:
        try:
            r = score_claim(bundle, row, L=L, c=c)
            st.metric("Calibrated fraud probability", f"{r['prob']:.1%}")
            st.subheader(f"Decision: {r['zone'].upper()}")
            ex = explain_claim(bundle, row, top=5)
            df = pd.DataFrame(ex)
            st.write("Top reasons (feature = value):")
            for e in ex:
                direction = "raises" if e["shap"] > 0 else "lowers"
                st.write(f"- {e['reason']} ({direction} risk)")
            st.bar_chart(df.set_index("feature")["shap"])
        except ValueError as err:
            st.error(str(err))

with tab2:
    st.write("Replays claims the final model did not train on (the calibration split), "
             "in file order. The isotonic calibrator did see these claims, so results "
             "are slightly optimistic. Illustrative only.")
    if X_all is None:
        st.stop()
    n = st.slider("Claims to replay", 100, 2000, 500, step=100)
    if st.button("Run replay"):
        idx = [i for i in bundle["cal_index"] if i in X_all.index][:n]
        cost_model = cost_none = cost_all = 0.0
        rows, lat = [], []
        for i in idx:
            r = dict(X_all.loc[i])
            t0 = time.perf_counter()
            res = score_claim(bundle, r, L=L, c=c)
            lat.append((time.perf_counter() - t0) * 1000)
            fraud = int(y_all.loc[i])
            flagged = res["zone"] in ("review", "investigate")
            cost_model += c if flagged else (L if fraud else 0.0)
            cost_none += L if fraud else 0.0
            cost_all += c
            rows.append({"model": cost_model, "investigate none": cost_none,
                         "investigate all": cost_all})
        st.line_chart(pd.DataFrame(rows))
        st.write(f"Final cost per claim: model {cost_model/len(idx):.3f}, "
                 f"none {cost_none/len(idx):.3f}, all {cost_all/len(idx):.3f}")
        st.write(f"Latency per claim: median {pd.Series(lat).median():.1f} ms, "
                 f"95th percentile {pd.Series(lat).quantile(0.95):.1f} ms")