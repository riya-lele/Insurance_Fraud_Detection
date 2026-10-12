import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from src.final_model import load_final, DEFAULT_PATH
from src.data import load_dataset
from src.score import score_claim, explain_claim

st.set_page_config(page_title="Claim risk triage", page_icon="🔎", layout="wide")

FRIENDLY = {
    "Age": "Policyholder age", "Sex": "Sex", "MaritalStatus": "Marital status",
    "AgeOfPolicyHolder": "Age band of policyholder", "DriverRating": "Driver rating (1-4)",
    "PastNumberOfClaims": "Past number of claims", "AddressChange_Claim": "Address change vs claim",
    "Make": "Vehicle make", "VehicleCategory": "Vehicle category", "VehiclePrice": "Vehicle price band",
    "AgeOfVehicle": "Vehicle age", "PolicyType": "Policy type", "BasePolicy": "Base policy",
    "Deductible": "Deductible", "NumberOfCars": "Number of cars", "AgentType": "Agent type",
    "RepNumber": "Claims rep number", "AccidentArea": "Accident area", "Fault": "At fault",
    "PoliceReportFiled": "Police report filed", "WitnessPresent": "Witness present",
    "Month": "Accident month", "WeekOfMonth": "Accident week of month", "DayOfWeek": "Accident day of week",
    "MonthClaimed": "Month claimed", "WeekOfMonthClaimed": "Claim week of month",
    "DayOfWeekClaimed": "Day of week claimed", "Days_Policy_Accident": "Days from policy start to accident",
    "Days_Policy_Claim": "Days from policy start to claim", "NumberOfSuppliments": "Number of supplements",
}
HELP = {
    "PastNumberOfClaims": "A range recorded in the dataset, not a claim-by-claim history.",
    "Days_Policy_Accident": "A range recorded in the dataset.",
    "Days_Policy_Claim": "A range recorded in the dataset.",
    "RepNumber": "Anonymised identifier of the claims representative.",
    "DriverRating": "Rating field as recorded in the dataset.",
    "AddressChange_Claim": "Time between an address change and the claim, as recorded.",
}
GROUPS = [
    ("Policyholder", ["Age", "Sex", "MaritalStatus", "AgeOfPolicyHolder", "DriverRating",
                      "PastNumberOfClaims", "AddressChange_Claim"]),
    ("Vehicle and policy", ["Make", "VehicleCategory", "VehiclePrice", "AgeOfVehicle", "PolicyType",
                            "BasePolicy", "Deductible", "NumberOfCars", "AgentType", "RepNumber"]),
    ("Accident", ["AccidentArea", "Fault", "PoliceReportFiled", "WitnessPresent",
                  "Month", "WeekOfMonth", "DayOfWeek"]),
    ("Claim filing", ["MonthClaimed", "WeekOfMonthClaimed", "DayOfWeekClaimed",
                      "Days_Policy_Accident", "Days_Policy_Claim", "NumberOfSuppliments"]),
]
MODE_PREFILL = "Prefill from an example"
MODE_MANUAL = "Fill in manually"


@st.cache_resource
def get_bundle():
    return load_final(DEFAULT_PATH)


@st.cache_data
def get_data():
    X, y, _, _ = load_dataset(verbose=False)
    return X, y


bundle = get_bundle()
try:
    X_all, y_all = get_data()
except Exception as e:
    st.error(f"Could not load data/raw/fraud_oracle.csv: {e}")
    st.stop()

feats = bundle["feature_list"]
cat_cols = bundle["cat_cols"]
base_rate = float(y_all.mean())
is_int = {f: bool(np.all(np.mod(X_all[f].astype(float), 1) == 0)) for f in bundle["num_cols"]}
unseen = [i for i in bundle["cal_index"] if i in X_all.index]
unseen_fraud = [i for i in unseen if int(y_all.loc[i]) == 1]


def key(f):
    return f"in_{f}"


def typical(f):
    return X_all[f].mode().iloc[0] if f in cat_cols else X_all[f].median()


def set_value(f, v):
    if v is None:
        st.session_state[key(f)] = None
    elif f in cat_cols:
        st.session_state[key(f)] = v
    else:
        st.session_state[key(f)] = int(v) if is_int[f] else float(v)


def load_typical():
    for f in feats:
        set_value(f, typical(f))
    st.session_state["loaded_label"] = None


def load_random(fraud_only):
    pool = unseen_fraud if fraud_only else unseen
    i = np.random.default_rng().choice(pool)
    for f in feats:
        set_value(f, X_all.loc[i, f])
    st.session_state["loaded_label"] = int(y_all.loc[i])


def clear_all():
    for f in feats:
        set_value(f, None)
    st.session_state["loaded_label"] = None


def fill_rest():
    for f in feats:
        if st.session_state.get(key(f)) is None:
            set_value(f, typical(f))


def on_mode_change():
    if st.session_state["mode"] == MODE_MANUAL:
        clear_all()
    else:
        load_typical()


if "mode" not in st.session_state:
    st.session_state["mode"] = MODE_PREFILL
if key(feats[0]) not in st.session_state:
    load_typical()

# ---------------- Sidebar ----------------
st.sidebar.header("Cost setting")
L = st.sidebar.slider(
    "How much worse is a missed fraud than one investigation?", 2, 100, int(bundle["L"]),
    help="Cost scale: one investigation costs 1. A missed fraud costs this many times more.")
c = float(bundle["c"])
t = c / (L + c)
st.sidebar.write(f"A missed fraud is treated as **{L}x** as costly as one investigation.")
st.sidebar.write(f"So claims with estimated fraud risk of **{t:.1%}** or more are worth a look, "
                 f"and **{2*t:.1%}** or more are sent to investigators.")
st.sidebar.caption("Costs are hypothetical, for demonstration only.")

# ---------------- Header ----------------
st.title("🔎 Claim risk triage")
st.write("Estimate how likely a car-insurance claim is to be fraudulent, and see why. "
         "This is a research demo, not a decision tool.")

tab1, tab2, tab3 = st.tabs(["Score a claim", "Replay a stream of claims", "About"])

# ---------------- Tab 1 ----------------
with tab1:
    st.radio("How do you want to enter the claim?", [MODE_PREFILL, MODE_MANUAL],
             key="mode", horizontal=True, on_change=on_mode_change)

    if st.session_state["mode"] == MODE_PREFILL:
        st.caption("All fields are filled from an example. You can still edit any of them.")
        b1, b2, b3, _ = st.columns([1, 1, 1, 2])
        b1.button("Random unseen claim", on_click=load_random, args=(False,),
                  help="A real claim from the data that the model was not trained on.")
        b2.button("Known fraud claim", on_click=load_random, args=(True,),
                  help="A real fraud-labelled claim the model was not trained on.")
        b3.button("Typical claim", on_click=load_typical)
    else:
        st.caption("Every field starts empty. Fill them all to see a result, or fill the rest "
                   "with typical values after entering the ones you care about.")
        st.button("Fill empty fields with typical values", on_click=fill_rest)

    left, right = st.columns([3, 2], gap="large")

    row = {}
    with left:
        used = set()
        for title, names in GROUPS:
            names = [n for n in names if n in feats]
            used.update(names)
            with st.expander(title, expanded=True):
                cols = st.columns(2)
                for j, f in enumerate(names):
                    with cols[j % 2]:
                        label, hlp = FRIENDLY.get(f, f), HELP.get(f)
                        if f in cat_cols:
                            opts = [v for v in bundle["category_maps"][f] if v is not None]
                            row[f] = st.selectbox(label, opts, index=None, key=key(f),
                                                  placeholder="Choose...", help=hlp)
                        elif is_int[f]:
                            row[f] = st.number_input(label, value=None, step=1, key=key(f), help=hlp)
                        else:
                            row[f] = st.number_input(label, value=None, key=key(f), help=hlp)
        others = [f for f in feats if f not in used]
        if others:
            with st.expander("Other", expanded=True):
                for f in others:
                    if f in cat_cols:
                        opts = [v for v in bundle["category_maps"][f] if v is not None]
                        row[f] = st.selectbox(f, opts, index=None, key=key(f), placeholder="Choose...")
                    else:
                        row[f] = st.number_input(f, value=None, key=key(f))

    filled = sum(1 for f in feats if row.get(f) is not None)

    with right:
        st.subheader("Result")
        st.progress(filled / len(feats), text=f"{filled} of {len(feats)} fields filled")
        if filled < len(feats):
            missing = [FRIENDLY.get(f, f) for f in feats if row.get(f) is None]
            st.info(f"Fill the remaining {len(missing)} field(s) to see a result, "
                    "or use the button above to fill them with typical values.")
            with st.expander("Which fields are empty?"):
                st.write(", ".join(missing))
        else:
            try:
                res = score_claim(bundle, row, L=float(L), c=c)
                p, zone = res["prob"], res["zone"]
                msgs = {
                    "approve": ("✅ Approve", "Low risk at this cost setting. No extra review suggested.", st.success),
                    "review": ("⚠️ Review", "Elevated risk. Worth a quick look by a handler.", st.warning),
                    "investigate": ("🚨 Investigate", "High risk. Suggest referring to investigators.", st.error),
                }
                title, text, box = msgs[zone]
                box(f"**{title}**\n\n{text}")
                st.metric("Estimated fraud probability", f"{p:.1%}",
                          f"{p / base_rate:.1f}x the average claim ({base_rate:.1%})",
                          delta_color="off")
                lab = st.session_state.get("loaded_label")
                if lab is not None:
                    st.caption("Recorded outcome of the loaded claim: "
                               + ("**fraud**" if lab == 1 else "**not fraud**")
                               + ". It no longer applies if you edit fields.")
                st.markdown("**Why? Top factors for this claim**")
                ex = explain_claim(bundle, row, top=5)
                for e in ex:
                    arrow = "🔺 raises" if e["shap"] > 0 else "🔻 lowers"
                    st.write(f"- **{FRIENDLY.get(e['feature'], e['feature'])}** = {e['value']} ({arrow} risk)")
                chart = pd.DataFrame({
                    "factor": [FRIENDLY.get(e["feature"], e["feature"]) for e in ex],
                    "effect on risk": [e["shap"] for e in ex]}).set_index("factor")
                st.bar_chart(chart)
                st.caption("Bars above zero push the risk up, below zero push it down. "
                           "These describe the model's reasoning, not the cause of fraud.")
            except ValueError as err:
                st.error(str(err))

# ---------------- Tab 2 ----------------
with tab2:
    st.write("Simulates claims arriving one by one. The model's policy (flag claims at or above "
             "the threshold) is compared with two simple policies: investigate nothing, or "
             "investigate everything.")
    st.caption("Uses claims the model did not train on, but the calibrator did see them, so "
               "results are slightly optimistic. Illustrative only. Costs are hypothetical.")
    n = st.slider("Number of claims to replay", 100, 2000, 500, step=100)
    if st.button("Run replay"):
        idx = unseen[:n]
        cm = cn = ca = 0.0
        rows, lat = [], []
        bar = st.progress(0.0)
        for k, i in enumerate(idx):
            r = dict(X_all.loc[i])
            t0 = time.perf_counter()
            res = score_claim(bundle, r, L=float(L), c=c)
            lat.append((time.perf_counter() - t0) * 1000)
            fraud = int(y_all.loc[i])
            flagged = res["zone"] in ("review", "investigate")
            cm += c if flagged else (L if fraud else 0.0)
            cn += L if fraud else 0.0
            ca += c
            rows.append({"Model policy": cm, "Investigate nothing": cn, "Investigate everything": ca})
            if k % 25 == 0:
                bar.progress((k + 1) / len(idx))
        bar.empty()
        m = len(idx)
        c1, c2, c3 = st.columns(3)
        c1.metric("Model policy, cost per claim", f"{cm / m:.3f}")
        c2.metric("Investigate nothing", f"{cn / m:.3f}")
        c3.metric("Investigate everything", f"{ca / m:.3f}")
        st.line_chart(pd.DataFrame(rows))
        st.caption("Lines show cumulative cost. Lower is better. "
                   f"Scoring time per claim: median {pd.Series(lat).median():.1f} ms.")

# ---------------- Tab 3 ----------------
with tab3:
    st.markdown(f"""
**What this does.** A gradient-boosting model, trained on a public auto-insurance dataset
(about 15,000 claims, roughly {base_rate:.0%} labelled fraud), estimates the probability that a
claim is fraudulent. A calibration step makes that probability meaningful.

**How the decision works.** One investigation costs 1. A missed fraud costs the number in the
sidebar. A claim is worth reviewing when its fraud probability exceeds 1 / (that number + 1).
Claims at twice that level are suggested for investigation.

**How to read the reasons.** Each reason shows a claim detail and whether it raised or lowered
the score. Many inputs are pre-grouped ranges in the data, so reasons are coarse. They describe
the model, not why fraud happened.

**Limits.**
- Costs are made up. The data has no claim amounts, so this shows how decisions change with
  assumptions, not real savings.
- Labels come from past investigations, so they may miss undetected fraud.
- One dataset of unknown origin. Results may not carry over to other insurers.
- Research demo only, not for real claims decisions.
""")