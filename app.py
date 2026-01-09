import re
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Config
# -----------------------------
APP_TITLE = "Cohort Finder"
DEFAULT_EXCEL_PATH = "databases.xlsx"  # <-- put your repo filename here

TOKEN_SPLIT_RE = re.compile(r"\s*(?:;|/|,|\||\n)\s*")  # robust split for your cells

EXPECTED_COLUMNS = [
    "Full Name",
    "Approx. # Participants",
    "Countries",
    "Questionnaires",
    "Diagnosis of CD or ASPD?",
    "Diagnosis Types",
    "Longitudinal Data?",
    "Hormone Data?",
    "Hormone Types",
    "Imaging Data?",
    "Imaging Types",
    "Environment Data?",
    "Environment Subcategories",
    "Study Types",
    "Family Structure",
    "Databank Info",
    "Free of Charge?",
    "Approx. Costs",
    "PubMed IDs",
    "Genetic Types",
    "Minor Ages",
]

BOOL_COLUMNS = [
    "Diagnosis of CD or ASPD?",
    "Longitudinal Data?",
    "Hormone Data?",
    "Imaging Data?",
    "Environment Data?",
    "Free of Charge?",
]

TOKEN_COLUMNS = [
    "Countries",
    "Questionnaires",
    "Diagnosis Types",
    "Hormone Types",
    "Imaging Types",
    "Environment Subcategories",
    "Study Types",
    "Family Structure",
    "Databank Info",
    "Genetic Types",
    "Minor Ages",
]

DISPLAY_COLUMNS = [
    "Full Name",
    "Approx. # Participants",
    "Countries",
    "Questionnaires",
    "Diagnosis of CD or ASPD?",
    "Longitudinal Data?",
    "Hormone Data?",
    "Imaging Data?",
    "Environment Data?",
    "Study Types",
    "Family Structure",
    "Databank Info",
    "Free of Charge?",
    "Approx. Costs",
    "Genetic Types",
    "Minor Ages",
]

# -----------------------------
# Helpers
# -----------------------------
def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def split_tokens(x) -> List[str]:
    s = _safe_str(x)
    if not s:
        return []
    return [t.strip() for t in TOKEN_SPLIT_RE.split(s) if t.strip()]

def parse_bool(x) -> Optional[bool]:
    s = _safe_str(x).lower()
    if not s:
        return None
    if s.startswith("yes"):
        return True
    if s.startswith("no"):
        return False
    if s in {"n/a", "na", "none"}:
        return None
    return None

def parse_int(x) -> Optional[int]:
    s = _safe_str(x)
    if not s:
        return None
    s = s.replace(",", "")
    m = re.search(r"\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None

def build_search_blob(df: pd.DataFrame) -> pd.Series:
    return df.fillna("").astype(str).agg(" | ".join, axis=1).str.lower()

def unique_tokens(df: pd.DataFrame, token_col: str) -> List[str]:
    all_tokens = set()
    for items in df[token_col]:
        for t in items:
            all_tokens.add(t)
    return sorted(all_tokens, key=lambda x: x.lower())

def row_matches_any(row_tokens: List[str], selected: List[str]) -> bool:
    s = set(row_tokens)
    return any(x in s for x in selected)

def row_matches_all(row_tokens: List[str], selected: List[str]) -> bool:
    s = set(row_tokens)
    return all(x in s for x in selected)

@st.cache_data(show_spinner=False)
def load_excel_from_path(path: str) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")

def prepare_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings = []
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        warnings.append("Missing expected columns (some filters/analytics may hide): " + ", ".join(missing))

    if "Approx. # Participants" in df.columns:
        df["_participants_num"] = df["Approx. # Participants"].apply(parse_int)
    else:
        df["_participants_num"] = None

    for c in BOOL_COLUMNS:
        if c in df.columns:
            df[f"_{c}_bool"] = df[c].apply(parse_bool)
        else:
            df[f"_{c}_bool"] = None

    for c in TOKEN_COLUMNS:
        if c in df.columns:
            df[f"_{c}_tokens"] = df[c].apply(split_tokens)
        else:
            df[f"_{c}_tokens"] = [[] for _ in range(len(df))]

    if "Full Name" not in df.columns:
        df["Full Name"] = [f"Row {i+1}" for i in range(len(df))]
        warnings.append("Column 'Full Name' not found; generated placeholder names.")

    df["_search_blob"] = build_search_blob(df)
    return df, warnings

def render_pubmed_links(pubmed_cell: str):
    ids = split_tokens(pubmed_cell)
    if not ids:
        st.write("—")
        return
    for pmid in ids:
        pmid_clean = re.sub(r"[^\d]", "", pmid)
        if pmid_clean:
            st.markdown(f"- [PMID {pmid_clean}](https://pubmed.ncbi.nlm.nih.gov/{pmid_clean}/)")

def render_details(row: pd.Series):
    st.subheader(_safe_str(row.get("Full Name", "")))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Participants (approx.)", _safe_str(row.get("Approx. # Participants", "—")))
        st.write("**Countries**")
        st.write(_safe_str(row.get("Countries", "—")) or "—")
    with c2:
        st.write("**Questionnaires / Instruments**")
        st.write(_safe_str(row.get("Questionnaires", "—")) or "—")
        st.write("**Minor ages**")
        st.write(_safe_str(row.get("Minor Ages", "—")) or "—")
    with c3:
        st.write("**CD/ASPD diagnosis?**")
        st.write(_safe_str(row.get("Diagnosis of CD or ASPD?", "—")) or "—")
        st.write("**Diagnosis types**")
        st.write(_safe_str(row.get("Diagnosis Types", "—")) or "—")

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.write("**Longitudinal?**")
        st.write(_safe_str(row.get("Longitudinal Data?", "—")) or "—")
    with m2:
        st.write("**Hormone data?**")
        st.write(_safe_str(row.get("Hormone Data?", "—")) or "—")
        st.caption(_safe_str(row.get("Hormone Types", "")))
    with m3:
        st.write("**Imaging data?**")
        st.write(_safe_str(row.get("Imaging Data?", "—")) or "—")
        st.caption(_safe_str(row.get("Imaging Types", "")))
    with m4:
        st.write("**Environment data?**")
        st.write(_safe_str(row.get("Environment Data?", "—")) or "—")
        st.caption(_safe_str(row.get("Environment Subcategories", "")))

    st.divider()

    s1, s2 = st.columns(2)
    with s1:
        st.write("**Study types**")
        st.write(_safe_str(row.get("Study Types", "—")) or "—")
        st.write("**Family structure**")
        st.write(_safe_str(row.get("Family Structure", "—")) or "—")
    with s2:
        st.write("**Genetic types**")
        st.write(_safe_str(row.get("Genetic Types", "—")) or "—")
        st.write("**Databank info**")
        st.write(_safe_str(row.get("Databank Info", "—")) or "—")

    st.divider()

    a1, a2 = st.columns(2)
    with a1:
        st.write("**Free of charge?**")
        st.write(_safe_str(row.get("Free of Charge?", "—")) or "—")
    with a2:
        st.write("**Approx. costs**")
        st.write(_safe_str(row.get("Approx. Costs", "—")) or "—")

    st.write("**PubMed IDs**")
    render_pubmed_links(_safe_str(row.get("PubMed IDs", "")))

# -----------------------------
# Filters (no "required fields")
# -----------------------------
def tri_state_bool_filter(label: str, series_bool: pd.Series):
    choice = st.selectbox(label, ["Any", "Yes", "No", "Unknown"], index=0)
    if choice == "Any":
        return pd.Series(True, index=series_bool.index), None
    if choice == "Yes":
        return series_bool.eq(True), "Yes"
    if choice == "No":
        return series_bool.eq(False), "No"
    return series_bool.isna(), "Unknown"

def apply_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    active = {}
    mask = pd.Series(True, index=df.index)

    q = st.sidebar.text_input(
        "Global search (all columns)",
        value="",
        placeholder="e.g., fMRI, CBCL, Sweden, cortisol, contact…",
    )
    if q.strip():
        qq = q.strip().lower()
        mask &= df["_search_blob"].str.contains(re.escape(qq), na=False)
        active["Global search"] = q.strip()

    with st.sidebar.expander("Yes/No flags", expanded=True):
        m, val = tri_state_bool_filter("CD/ASPD diagnosis?", df["_Diagnosis of CD or ASPD?_bool"])
        mask &= m
        if val: active["CD/ASPD diagnosis"] = val

        m, val = tri_state_bool_filter("Longitudinal data?", df["_Longitudinal Data?_bool"])
        mask &= m
        if val: active["Longitudinal"] = val

        m, val = tri_state_bool_filter("Hormone data?", df["_Hormone Data?_bool"])
        mask &= m
        if val: active["Hormone data"] = val

        m, val = tri_state_bool_filter("Imaging data?", df["_Imaging Data?_bool"])
        mask &= m
        if val: active["Imaging data"] = val

        m, val = tri_state_bool_filter("Environment data?", df["_Environment Data?_bool"])
        mask &= m
        if val: active["Environment data"] = val

        m, val = tri_state_bool_filter("Free of charge?", df["_Free of Charge?_bool"])
        mask &= m
        if val: active["Free of charge"] = val

    st.sidebar.markdown("### Structured filters")

    # Participants range
    if df["_participants_num"].notna().any():
        pmin = int(df["_participants_num"].dropna().min())
        pmax = int(df["_participants_num"].dropna().max())
        lo, hi = st.sidebar.slider(
            "Participants (approx.)",
            min_value=pmin,
            max_value=pmax,
            value=(pmin, pmax),
            step=max(1, (pmax - pmin) // 200) if pmax > pmin else 1,
        )
        mask &= df["_participants_num"].fillna(-1).between(lo, hi)
        active["Participants"] = f"{lo}–{hi}"

    # Geography
    with st.sidebar.expander("Geography", expanded=False):
        countries = unique_tokens(df, "_Countries_tokens")
        sel_countries = st.multiselect("Countries", options=countries, default=[])
        if sel_countries:
            mask &= df["_Countries_tokens"].apply(lambda xs: row_matches_any(xs, sel_countries))
            active["Countries"] = ", ".join(sel_countries)

    # Phenotyping
    with st.sidebar.expander("Phenotyping & diagnosis", expanded=False):
        instruments = unique_tokens(df, "_Questionnaires_tokens")
        sel_instr = st.multiselect("Questionnaires / instruments", options=instruments, default=[])
        match_mode = st.radio("Instrument match", options=["Any selected", "All selected"], horizontal=True, index=0)
        if sel_instr:
            if match_mode == "Any selected":
                mask &= df["_Questionnaires_tokens"].apply(lambda xs: row_matches_any(xs, sel_instr))
            else:
                mask &= df["_Questionnaires_tokens"].apply(lambda xs: row_matches_all(xs, sel_instr))
            active["Questionnaires"] = f"{match_mode}: " + ", ".join(sel_instr)

        diag_types = unique_tokens(df, "_Diagnosis Types_tokens")
        sel_diag_types = st.multiselect("Diagnosis types", options=diag_types, default=[])
        if sel_diag_types:
            mask &= df["_Diagnosis Types_tokens"].apply(lambda xs: row_matches_any(xs, sel_diag_types))
            active["Diagnosis types"] = ", ".join(sel_diag_types)

    # Modalities (types)
    with st.sidebar.expander("Modalities (types)", expanded=False):
        hormone_types = unique_tokens(df, "_Hormone Types_tokens")
        sel_horm = st.multiselect("Hormone types", options=hormone_types, default=[])
        if sel_horm:
            mask &= df["_Hormone Types_tokens"].apply(lambda xs: row_matches_any(xs, sel_horm))
            active["Hormone types"] = ", ".join(sel_horm)

        imaging_types = unique_tokens(df, "_Imaging Types_tokens")
        sel_img = st.multiselect("Imaging types", options=imaging_types, default=[])
        if sel_img:
            mask &= df["_Imaging Types_tokens"].apply(lambda xs: row_matches_any(xs, sel_img))
            active["Imaging types"] = ", ".join(sel_img)

        env_sub = unique_tokens(df, "_Environment Subcategories_tokens")
        sel_env = st.multiselect("Environment subcategories", options=env_sub, default=[])
        if sel_env:
            mask &= df["_Environment Subcategories_tokens"].apply(lambda xs: row_matches_any(xs, sel_env))
            active["Environment subcategories"] = ", ".join(sel_env)

    # Study design
    with st.sidebar.expander("Study design", expanded=False):
        study_types = unique_tokens(df, "_Study Types_tokens")
        sel_study = st.multiselect("Study types", options=study_types, default=[])
        if sel_study:
            mask &= df["_Study Types_tokens"].apply(lambda xs: row_matches_any(xs, sel_study))
            active["Study types"] = ", ".join(sel_study)

        family = unique_tokens(df, "_Family Structure_tokens")
        sel_family = st.multiselect("Family structure", options=family, default=[])
        if sel_family:
            mask &= df["_Family Structure_tokens"].apply(lambda xs: row_matches_any(xs, sel_family))
            active["Family structure"] = ", ".join(sel_family)

    # Access & cost
    with st.sidebar.expander("Access & cost", expanded=False):
        dbinfo = unique_tokens(df, "_Databank Info_tokens")
        sel_dbinfo = st.multiselect("Databank access model", options=dbinfo, default=[])
        if sel_dbinfo:
            mask &= df["_Databank Info_tokens"].apply(lambda xs: row_matches_any(xs, sel_dbinfo))
            active["Databank info"] = ", ".join(sel_dbinfo)

        cost_q = st.text_input("Cost text contains", value="", placeholder="e.g., SEK, USD, thousand…")
        if cost_q.strip() and "Approx. Costs" in df.columns:
            mask &= df["Approx. Costs"].fillna("").astype(str).str.contains(cost_q, case=False, na=False)
            active["Cost contains"] = cost_q.strip()

    # Genetics + ages
    with st.sidebar.expander("Genetics & ages", expanded=False):
        genetics = unique_tokens(df, "_Genetic Types_tokens")
        sel_gen = st.multiselect("Genetic types", options=genetics, default=[])
        if sel_gen:
            mask &= df["_Genetic Types_tokens"].apply(lambda xs: row_matches_any(xs, sel_gen))
            active["Genetic types"] = ", ".join(sel_gen)

        ages = unique_tokens(df, "_Minor Ages_tokens")
        sel_ages = st.multiselect("Minor ages bands", options=ages, default=[])
        if sel_ages:
            mask &= df["_Minor Ages_tokens"].apply(lambda xs: row_matches_any(xs, sel_ages))
            active["Minor ages"] = ", ".join(sel_ages)

    # PubMed search
    pmid_q = st.sidebar.text_input("PubMed ID contains", value="", placeholder="e.g., 3924…")
    if pmid_q.strip() and "PubMed IDs" in df.columns:
        mask &= df["PubMed IDs"].fillna("").astype(str).str.contains(pmid_q.strip(), case=False, na=False)
        active["PubMed contains"] = pmid_q.strip()

    return df[mask].copy(), active

# -----------------------------
# Analytics utilities
# -----------------------------
def token_frequency(df: pd.DataFrame, token_col: str) -> pd.DataFrame:
    """Count occurrences of tokens in a *_tokens column."""
    s = df[token_col].explode()
    s = s.dropna()
    s = s[s.astype(str).str.strip().ne("")]
    counts = s.value_counts().rename_axis("token").reset_index(name="count")
    return counts

def token_weighted_sum(df: pd.DataFrame, token_col: str, weight_col: str, split_weight_across_tokens: bool) -> pd.DataFrame:
    """
    Sum weights (participants) by token.
    If split_weight_across_tokens=True, each row's weight is divided by number of tokens in that row.
    """
    tmp = df[[token_col, weight_col]].copy()
    tmp = tmp.explode(token_col)
    tmp = tmp.dropna(subset=[token_col])
    tmp["token"] = tmp[token_col].astype(str).str.strip()
    tmp = tmp[tmp["token"].ne("")]

    if split_weight_across_tokens:
        denom = df[token_col].apply(lambda xs: max(1, len(xs))).reindex(df.index)
        tmp["_denom"] = denom.loc[tmp.index].values
        tmp["_w"] = tmp[weight_col].fillna(0) / tmp["_denom"]
    else:
        tmp["_w"] = tmp[weight_col].fillna(0)

    out = tmp.groupby("token", as_index=False)["_w"].sum().rename(columns={"_w": "participants_sum"})
    out = out.sort_values("participants_sum", ascending=False)
    return out

def yes_no_unknown_summary(df: pd.DataFrame, bool_helper_col: str, label: str) -> pd.DataFrame:
    s = df[bool_helper_col]
    return pd.DataFrame(
        {
            "feature": [label] * 3,
            "state": ["Yes", "No", "Unknown"],
            "count": [int((s == True).sum()), int((s == False).sum()), int(s.isna().sum())],
        }
    )

# -----------------------------
# Pages
# -----------------------------
def page_browse(df: pd.DataFrame):
    st.subheader("Browse")
    st.caption("Filter in the sidebar, scan results, and open a cohort profile on the right.")

    filtered_df, active_filters = apply_filters(df)

    # KPI strip
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Matching cohorts", len(filtered_df))
    with c2:
        total_participants = filtered_df["_participants_num"].dropna().sum() if len(filtered_df) else 0
        st.metric("Sum participants (approx.)", f"{int(total_participants):,}".replace(",", " "))
    with c3:
        uniq_countries = set()
        for xs in filtered_df["_Countries_tokens"]:
            uniq_countries.update(xs)
        st.metric("Unique countries", len(uniq_countries))
    with c4:
        st.metric("Active filters", len(active_filters))

    if active_filters:
        with st.expander("Show active filters", expanded=False):
            st.json(active_filters)

    st.divider()

    # Sorting controls
    sort_col = st.selectbox(
        "Sort results by",
        options=["Full Name", "Approx. # Participants"],
        index=1,
        help="Sorting only affects how results are displayed.",
    )
    sort_desc = st.checkbox("Descending", value=True if sort_col == "Approx. # Participants" else False)

    view_cols = [c for c in DISPLAY_COLUMNS if c in filtered_df.columns]
    view = filtered_df[view_cols].copy()

    if sort_col == "Approx. # Participants":
        view["_p"] = filtered_df["_participants_num"]
        view = view.sort_values("_p", ascending=not sort_desc).drop(columns=["_p"])
    else:
        view = view.sort_values("Full Name", ascending=not sort_desc)


    st.write("### Results table")

    # Row selection (plus a safe fallback dropdown)
    event = st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=520,
    )

    csv_bytes = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered results (CSV)",
        data=csv_bytes,
        file_name="filtered_cohorts.csv",
        mime="text/csv",
    )

    st.write("### Quick scan (cards)")
    page_size = st.slider("Cards per page", 5, 30, 10)
    page_i = st.number_input("Page", min_value=1, max_value=max(1, (len(view) - 1) // page_size + 1), value=1)
    start = (page_i - 1) * page_size
    end = start + page_size

    for _, r in view.iloc[start:end].iterrows():
        title = _safe_str(r.get("Full Name", ""))
        subtitle = f"{_safe_str(r.get('Countries',''))} • Participants: {_safe_str(r.get('Approx. # Participants','—'))}"
        with st.expander(f"{title} — {subtitle}"):
            render_details(filtered_df.loc[filtered_df["Full Name"] == title].iloc[0])



def page_analytics(df: pd.DataFrame):
    st.subheader("Analytics dashboard")
    st.caption("Quick overview of coverage, instruments, geography, and age bands. Filters do NOT apply here by default.")

    # Optional toggle: apply current filters to analytics (some people like this)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Analytics scope")
    use_filters = st.sidebar.checkbox("Apply current Browse filters to Analytics", value=False)
    if use_filters:
        filtered_df, _ = apply_filters(df)
    else:
        filtered_df = df

    # --- Instruments ---
    st.write("## Instruments / questionnaires")

    instr_counts = token_frequency(filtered_df, "_Questionnaires_tokens")
    unique_instr = int(instr_counts["token"].nunique()) if len(instr_counts) else 0
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Unique instruments", unique_instr)
    with c2:
        st.metric("Total instrument mentions", int(instr_counts["count"].sum()) if len(instr_counts) else 0)

    top_n = st.slider("Top N instruments", 10, 80, 25)
    if len(instr_counts):
        top_instr = instr_counts.head(top_n).set_index("token")
        st.bar_chart(top_instr["count"])
        with st.expander("See instrument table"):
            st.dataframe(instr_counts, use_container_width=True, hide_index=True)
    else:
        st.info("No instrument tokens available.")

    # --- Country ---
    st.write("## Geography")

    split_participants = st.checkbox(
        "Split participants across multiple countries per cohort",
        value=True,
        help="If a cohort lists multiple countries, this divides its participant count equally across them.",
    )

    if filtered_df["_participants_num"].notna().any():
        part_by_country = token_weighted_sum(
            filtered_df,
            token_col="_Countries_tokens",
            weight_col="_participants_num",
            split_weight_across_tokens=split_participants,
        )
        top_c = st.slider("Top N countries", 10, 60, 25)
        st.bar_chart(part_by_country.head(top_c).set_index("token")["participants_sum"])
        with st.expander("Country participants table"):
            st.dataframe(part_by_country, use_container_width=True, hide_index=True)
    else:
        st.info("No numeric participants parsed; country participants chart hidden.")

    country_counts = token_frequency(filtered_df, "_Countries_tokens")
    if len(country_counts):
        top_c2 = st.slider("Top N countries by cohort count", 10, 60, 25)
        st.bar_chart(country_counts.head(top_c2).set_index("token")["count"])
    else:
        st.info("No country tokens available.")

    # --- Ages ---
    st.write("## Age bands (Minor Ages)")

    age_counts = token_frequency(filtered_df, "_Minor Ages_tokens")
    if len(age_counts):
        st.bar_chart(age_counts.set_index("token")["count"])
        with st.expander("Age band table"):
            st.dataframe(age_counts, use_container_width=True, hide_index=True)
    else:
        st.info("No age-band tokens available.")

    # --- Coverage (nice + simple) ---
    st.write("## Dataset feature coverage")

    cov = pd.concat(
        [
            yes_no_unknown_summary(filtered_df, "_Longitudinal Data?_bool", "Longitudinal"),
            yes_no_unknown_summary(filtered_df, "_Hormone Data?_bool", "Hormone data"),
            yes_no_unknown_summary(filtered_df, "_Imaging Data?_bool", "Imaging data"),
            yes_no_unknown_summary(filtered_df, "_Environment Data?_bool", "Environment data"),
            yes_no_unknown_summary(filtered_df, "_Diagnosis of CD or ASPD?_bool", "CD/ASPD diagnosis"),
            yes_no_unknown_summary(filtered_df, "_Free of Charge?_bool", "Free of charge"),
        ],
        ignore_index=True,
    )

    # simple display: pivot table (counts)
    cov_pivot = cov.pivot_table(index="feature", columns="state", values="count", aggfunc="sum").fillna(0).astype(int)
    st.dataframe(cov_pivot, use_container_width=True)

    # --- Modalities types (top) ---
    st.write("## Top imaging / hormone types (mentions)")

    col1, col2 = st.columns(2)
    with col1:
        img_counts = token_frequency(filtered_df, "_Imaging Types_tokens")
        if len(img_counts):
            st.bar_chart(img_counts.head(20).set_index("token")["count"])
        else:
            st.info("No imaging type tokens.")
    with col2:
        horm_counts = token_frequency(filtered_df, "_Hormone Types_tokens")
        if len(horm_counts):
            st.bar_chart(horm_counts.head(20).set_index("token")["count"])
        else:
            st.info("No hormone type tokens.")

    # --- Missingness (quietly useful) ---
    st.write("## Data completeness (missingness)")

    base_cols = [c for c in EXPECTED_COLUMNS if c in filtered_df.columns]
    miss = (
        filtered_df[base_cols]
        .isna()
        .mean()
        .mul(100)
        .round(1)
        .sort_values(ascending=False)
        .rename("missing_%")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    st.bar_chart(miss.set_index("column")["missing_%"])
    with st.expander("Missingness table"):
        st.dataframe(miss, use_container_width=True, hide_index=True)

def page_about():
    st.subheader("About / Excel schema")
    st.write(
        "This app reads a single Excel sheet and tokenizes multi-entry cells using delimiters like `;`, `/`, `,`, or `|`."
    )
    st.markdown("**Expected headers:**")
    st.code("\n".join(EXPECTED_COLUMNS), language="text")
    st.markdown(
        """
**Tips**
- Keep multi-value cells consistent: `CBCL / YSR` or `CBCL; YSR` both work.
- Values like `Yes (substudy)` are treated as `Yes` for boolean filters.
"""
    )

# -----------------------------
# App entry
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Browse cohorts like an interactive database, then explore coverage & summary analytics.")

# Load from repo
try:
    raw_df = load_excel_from_path(DEFAULT_EXCEL_PATH)
    source_label = f"Repo file: {DEFAULT_EXCEL_PATH}"
except Exception as e:
    raw_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    source_label = f"Could not load {DEFAULT_EXCEL_PATH}: {e}"

df, prep_warnings = prepare_df(raw_df)
for w in prep_warnings:
    st.warning(w)

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Browse", "Analytics", "About"], index=0)
st.sidebar.caption(f"Loaded rows: **{len(df)}**  •  Source: **{source_label}**")

if page == "Browse":
    page_browse(df)
elif page == "Analytics":
    page_analytics(df)
else:
    page_about()
