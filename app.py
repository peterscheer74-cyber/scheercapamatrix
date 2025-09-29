
import io
import re
import time
from datetime import datetime
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

LEVELS = ["Foundational", "Intermediate", "Advanced", "Leading"]
LEVEL_TO_NUM = {lvl: i + 1 for i, lvl in enumerate(LEVELS)}
NUM_TO_LEVEL = {v: k for k, v in LEVEL_TO_NUM.items()}

st.set_page_config(page_title="Capability Assessment", page_icon="🧭", layout="wide")

def inject_css():
    st.markdown(
        """
        <style>
        .app-header { display:flex; align-items:center; justify-content:space-between; gap:1rem;
                      padding:0.5rem 0 1rem 0; border-bottom:1px solid #eee; }
        .score-card { border:1px solid #eee; border-radius:16px; padding:1rem;
                      box-shadow:0 1px 3px rgba(0,0,0,0.05); }
        .small { font-size:0.85rem; color:#555; }
        </style>
        """,
        unsafe_allow_html=True
    )

def detect_skill_column(df: pd.DataFrame) -> str:
    candidate_cols = [c for c in df.columns if not isinstance(c, tuple)]
    if not candidate_cols:
        # fallback if all columns are MultiIndex
        return df.columns[0] if len(df.columns) else "Skill"
    lowered = {c: str(c).lower() for c in candidate_cols}
    for kw in ["skill", "kompetenz", "capability", "kompetenzen"]:
        for c, lc in lowered.items():
            if kw in lc:
                return c
    return candidate_cols[0]

def to_level_number(val) -> Optional[int]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == "":
        return None
    s_clean = re.sub(r"[^A-Za-z0-9 ]+", "", s).strip()
    for name, num in LEVEL_TO_NUM.items():
        if name.lower() in s_clean.lower():
            return num
    m = re.search(r"(\d+)", s_clean)
    if m:
        try:
            n = int(m.group(1))
            if 1 <= n <= 4:
                return n
        except ValueError:
            pass
    return None

def compute_match_score(gaps: List[int]) -> float:
    if not gaps:
        return 0.0
    pos_gaps = [g for g in gaps if g > 0]
    if not pos_gaps:
        return 100.0
    avg_gap = sum(pos_gaps) / len(gaps)
    score = max(0.0, 100.0 - avg_gap * 25.0)
    return round(score, 1)

def make_gap_chart(df_gaps: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df_gaps["Skill"], df_gaps["Gap"])
    ax.set_xlabel("Skill")
    ax.set_ylabel("Gap (Self - Soll)")
    ax.set_title("Gaps by Skill (positive = below target)")
    ax.tick_params(axis='x', rotation=60)
    st.pyplot(fig)

def load_matrix(file) -> Tuple[pd.DataFrame, List[Tuple[str, str]], str]:
    df = pd.read_excel(file, header=[0,1], engine="openpyxl")
    df = df.loc[:, ~df.columns.duplicated()]
    mi_cols = [c for c in df.columns if isinstance(c, tuple) and (pd.notna(c[0]) or pd.notna(c[1]))]
    combos = []
    for c in mi_cols:
        team = str(c[0]).strip() if pd.notna(c[0]) else ""
        pos = str(c[1]).strip() if pd.notna(c[1]) else ""
        if team or pos:
            combos.append((team, pos))
    skill_col = detect_skill_column(df)
    return df, combos, skill_col

# ---------------- UI ----------------
inject_css()
st.markdown("<div class='app-header'><h2>🧭 Capability Assessment</h2><span class='small'>Selbstbewertung vs. Soll-Profil</span></div>", unsafe_allow_html=True)

with st.expander("1) Excel-Matrix hochladen", expanded=True):
    uploaded = st.file_uploader("Excel-Datei (zweizeiliger Header: Teams in Zeile 1, Positionen in Zeile 2)", type=["xlsx","xls"])
    st.caption("Spalten F–AM enthalten die Team/Positions-Profile. Links stehen Skillnamen/Metadaten.")

if "df" not in st.session_state and uploaded is not None:
    try:
        df, combos, skill_col = load_matrix(uploaded)
        st.session_state.df = df
        st.session_state.combos = combos
        st.session_state.skill_col = skill_col
        st.success(f"Datei geladen. Erkannte Skill-Spalte: **{skill_col}** · {len(combos)} Profile.")
    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")

if "df" in st.session_state:
    df = st.session_state.df
    combos = st.session_state.combos
    skill_col = st.session_state.skill_col

    teams = sorted({t for (t, p) in combos if t})
    col1, col2, col3 = st.columns([1.2, 1.2, 2])
    with col1:
        team = st.selectbox("Team", options=teams, index=0 if teams else None, placeholder="Team wählen...")
    with col2:
        positions = sorted({p for (t, p) in combos if t == team and p})
        pos = st.selectbox("Position", options=positions, index=0 if positions else None, placeholder="Position wählen...")
    with col3:
        employee_name = st.text_input("Dein Name (optional)", placeholder="Max Mustermann")

    if team and pos:
        target_cols = [c for c in df.columns if isinstance(c, tuple) and str(c[0]).strip()==team and str(c[1]).strip()==pos]
        if not target_cols:
            st.warning("Kein Soll-Profil für diese Auswahl gefunden.")
        else:
            base_cols = [c for c in df.columns if not isinstance(c, tuple)]
            base = df[base_cols].copy()

            def combine_row(row):
                vals = [row[c] for c in target_cols]
                for v in vals:
                    if pd.notna(v):
                        return v
                return np.nan

            soll_series = df.apply(combine_row, axis=1)
            work = pd.DataFrame({
                "Skill": base[skill_col].astype(str).str.strip(),
                "Soll (Text)": soll_series,
                "Soll (Stufe)": [to_level_number(v) for v in soll_series],
            })
            work = work[work["Skill"].str.strip() != ""].reset_index(drop=True)

            left, right = st.columns([2,1])
            with left:
                st.subheader("Soll-Profil")
                st.dataframe(work[["Skill", "Soll (Text)"]], use_container_width=True, hide_index=True)
            with right:
                st.subheader("Legende")
                st.write(", ".join([f"**{NUM_TO_LEVEL[i]}** = {i}" for i in range(1,5)]))
                st.caption("Soll-Werte als Text, für Gap-Berechnung 1–4 zugeordnet.")

            st.markdown("---")
            st.subheader("2) Selbstbewertung")
            st.caption("Wähle je Skill deine Stufe.")

            self_levels = []
            for i, row in work.iterrows():
                c1, c2 = st.columns([2.5, 1.5])
                with c1:
                    st.markdown(f"**{row['Skill']}**")
                with c2:
                    sel = st.selectbox(
                        f"Selbstbewertung – {row['Skill']}",
                        options=LEVELS,
                        key=f"self_{i}",
                        index=None,
                        placeholder="Stufe wählen..."
                    )
                self_levels.append(sel)

            work["Selbst (Text)"] = self_levels
            work["Selbst (Stufe)"] = [LEVEL_TO_NUM.get(v) if v else None for v in self_levels]
            work["Gap"] = [
                (soll if soll is not None else np.nan) - (selfv if selfv is not None else np.nan)
                for soll, selfv in zip(work["Soll (Stufe)"], work["Selbst (Stufe)"])
            ]

            valid_gaps = [g for g in work["Gap"] if pd.notna(g)]
            gaps_int = [int(g) for g in valid_gaps]
            score = compute_match_score(gaps_int) if gaps_int else 0.0

            st.markdown("---")
            k1, k2, k3 = st.columns([1,1,2])
            with k1:
                st.markdown("**Match-Score**")
                st.markdown(f"<div class='score-card'><h2 style='margin:0'>{score}</h2><div class='small'>0–100</div></div>", unsafe_allow_html=True)
            with k2:
                met = sum(1 for g in gaps_int if g <= 0)
                total = len(gaps_int) if gaps_int else 0
                st.markdown("**Anforderungen erfüllt**")
                st.markdown(f"<div class='score-card'><h2 style='margin:0'>{met} / {total}</h2><div class='small'>Skills ≥ Soll</div></div>", unsafe_allow_html=True)
            with k3:
                st.markdown("**Gaps pro Skill**")
                df_gaps = work[["Skill", "Gap"]].copy()
                df_gaps = df_gaps[pd.notna(df_gaps["Gap"])]
                if not df_gaps.empty:
                    make_gap_chart(df_gaps)

            st.markdown("---")
            st.subheader("3) Export")
            meta = {
                "Team": team,
                "Position": pos,
                "Name": employee_name or "",
                "Timestamp": datetime.now().isoformat(timespec="seconds"),
                "Score": score,
            }
            export_df = work.copy()
            for k, v in meta.items():
                export_df[k] = v

            csv_buf = io.StringIO()
            export_df.to_csv(csv_buf, index=False)
            st.download_button(
                "Ergebnis als CSV herunterladen",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name=f"assessment_{team}_{pos}_{int(time.time())}.csv",
                mime="text/csv",
            )

else:
    st.info("Lade die Excel-Datei, um zu starten.")
