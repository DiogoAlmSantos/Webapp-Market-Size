import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from fpdf import FPDF
import tempfile
import os

# PERT Definition
def pert(s, m, l):
    return (s + 4*m + l) / 6

# ==========================================
# 1. CONFIG & HELPER FUNCTIONS
# ==========================================
st.set_page_config(page_title="Digital Twin Value Analysis", layout="wide")

# Custom formatter for currency
def currency_fmt(x, pos=None):
    if abs(x) >= 1e9: return f'€{x*1e-9:.1f}B'
    if abs(x) >= 1e6: return f'€{x*1e-6:.1f}M'
    if abs(x) >= 1e3: return f'€{x*1e-3:.0f}K'
    return f'€{x:.0f}'

def run_monte_carlo(opex, savings_min, savings_mode, savings_max, n_sims=50000):
    if opex == 0: return np.zeros(n_sims)
    savings_pct = np.random.triangular(savings_min, savings_mode, savings_max, size=n_sims)
    return opex * savings_pct

# --- PDF GENERATION HELPERS ---
def save_plot_to_temp(fig):
    """Saves a matplotlib figure to a temp file and returns the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        fig.savefig(tmp.name, bbox_inches='tight', dpi=100)
        return tmp.name

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Digital Twin Value Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, label, 0, 1, 'L', 1)
        self.ln(4)

def create_pdf(inputs, plots, table_data):
    # Helper to prevent Unicode errors in FPDF (Latin-1 limitations)
    def fmt(val):
        # Format currency and replace Euro symbol with EUR for PDF compatibility
        return currency_fmt(val).replace('€', 'EUR')

    def clean_str(text):
        # Ensure any other strings don't have the symbol
        return str(text).replace('€', 'EUR')

    pdf = PDF()
    pdf.add_page()
    
    # --- SECTION A: INPUT DATA ---
    pdf.chapter_title("A. Input Data & Assumptions")
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "1. Simulation Parameters", 0, 1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 6, f"Savings Distribution: Min {inputs['dt_min']*100}% | Mode {inputs['dt_mode']*100}% | Max {inputs['dt_max']*100}%", 0, 1)
    pdf.ln(2)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "2. Financial Parameters", 0, 1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 6, f"Analysis Horizon: {inputs['analysis_years']} Years", 0, 1)
    pdf.cell(0, 6, f"Discount Rate: {inputs['discount_rate']*100}%", 0, 1)
    pdf.ln(2)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "3. Pricing Models", 0, 1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 6, f"SaaS: Setup {fmt(inputs['setup_fee'])} | Annual {fmt(inputs['annual_fee'])}", 0, 1)
    pdf.cell(0, 6, f"Performance Based: {inputs['pb_cut']*100}% of Savings | Decay {inputs['decay_factor']*100}%/yr", 0, 1)
    pdf.ln(2)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "4. Company Costs", 0, 1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 6, f"CAPEX: {fmt(inputs['company_capex'])} | Annual OPEX: {fmt(inputs['company_opex'])}", 0, 1)
    pdf.ln(5)

    # Input Data Table
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "5. Data Snapshot (Top Entities)", 0, 1)
    pdf.set_font("Arial", size=9)
    # Header
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(40, 8, "Entity", 1, 0, 'C', 1)
    pdf.cell(50, 8, "Total OPEX", 1, 0, 'C', 1)
    pdf.cell(50, 8, "Est. Annual Savings", 1, 1, 'C', 1)
    # Rows
    for row in table_data:
        # We manually replace characters to be safe
        pdf.cell(40, 8, clean_str(row['Entity']), 1)
        pdf.cell(50, 8, clean_str(row['opex']), 1)
        pdf.cell(50, 8, clean_str(row['savings']), 1, 1)
    pdf.ln(10)

    # --- SECTION B: MARKET SIZING ---
    pdf.add_page()
    pdf.chapter_title("B. Market Sizing Analysis")
    
    if 'risk_Total' in plots:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Risk Profiles (Total Market)", 0, 1)
        pdf.image(plots['risk_Total'], x=10, w=180)
        pdf.ln(5)
    
    if 'potential_Total' in plots:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Market Potential (TAM/SAM/SOM)", 0, 1)
        pdf.image(plots['potential_Total'], x=10, w=180)

    # --- SECTION C: FINANCIAL ANALYSIS ---
    pdf.add_page()
    pdf.chapter_title("C. Financial Analysis Results")

    if 'fin_npv' in plots:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "NPV Distribution Comparison", 0, 1)
        pdf.image(plots['fin_npv'], x=10, w=190)
        pdf.ln(5)

    if 'fin_roi' in plots:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Company ROI Distribution", 0, 1)
        pdf.image(plots['fin_roi'], x=30, w=150)
        pdf.ln(5)

    if 'fin_cf' in plots:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Cumulative Cash Flow Analysis", 0, 1)
        pdf.image(plots['fin_cf'], x=10, w=180)

    return pdf

# ==========================================
# 2. SIDEBAR: INPUTS
# ==========================================
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload 'mc_input.xlsx'", type=["xlsx"])

st.sidebar.header("2. Simulation Assumptions")
st.sidebar.subheader("Digital Twin Savings (%)")
dt_min = st.sidebar.slider("Minimum Savings", 0.0, 1.0, 0.00, 0.005, format="%.2f")
dt_mode = st.sidebar.slider("Most Likely Savings", 0.0, 1.0, 0.03, 0.005, format="%.2f")
dt_max = st.sidebar.slider("Maximum Savings", 0.0, 1.0, 0.10, 0.005, format="%.2f")

st.sidebar.header("3. Financial Params (Carris)")
st.sidebar.info("Overrides Excel defaults.")

col_fin1, col_fin2 = st.sidebar.columns(2)
with col_fin1:
    analysis_years = st.number_input("Analysis Years", value=5, step=1, min_value=1, max_value=20)
with col_fin2:
    discount_rate_input = st.number_input("Discount Rate (%)", value=8.0, step=0.5, format="%.1f") / 100.0

st.sidebar.subheader("SaaS Pricing")
setup_fee = st.sidebar.number_input("Setup Fee (€)", value=50000, step=5000)
annual_fee = st.sidebar.number_input("Annual License (€)", value=150000, step=5000)

st.sidebar.subheader("Perf. Based Model")
pb_cut = st.sidebar.slider("Revenue Cut (%)", 0.0, 0.5, 0.1, 0.01)
decay_factor = st.sidebar.slider("Annual Benefit Decay (%)", 0.0, 0.2, 0.05, 0.01)

st.sidebar.subheader("Company Costs (Your Side)")
company_capex_input = st.sidebar.number_input("Company CAPEX (€)", value=200000, step=10000)
company_opex_input = st.sidebar.number_input("Company Annual OPEX (€)", value=50000, step=5000)

# Container to store plots for PDF
pdf_plots = {}
pdf_table_data = []

# ==========================================
# 3. MAIN APP LOGIC
# ==========================================
st.title("Digital Twin: Market Sizing & Financial Analysis")

if uploaded_file is None:
    st.info("👋 Please upload your Excel file to begin the analysis.")
else:
    # Load Data
    df_data = pd.read_excel(uploaded_file, sheet_name='Data')
    
    # ---------------------------------------------------------
    # PART 0: INPUT DATA OVERVIEW
    # ---------------------------------------------------------
    st.header("Input Data Overview")
    
    # Clean the entity column for matching
    df_data['entity_clean'] = df_data['entity'].astype(str).str.lower().str.strip()
    
    # Define the STRICT list of entities you want to show (in this order)
    target_entities = ['carris', 'portugal', 'espanha', 'iberia', 'eu-27']
    
    # Filter the dataframe to only include these rows
    df_display_base = df_data[df_data['entity_clean'].isin(target_entities)].copy()
    
    # Sort them by the order in 'target_entities' list
    df_display_base['entity_clean'] = pd.Categorical(df_display_base['entity_clean'], categories=target_entities, ordered=True)
    df_display_base = df_display_base.sort_values('entity_clean')

    # Select and rename columns for display
    display_cols = {
        'entity': 'Entity',
        'pkt_road': 'Passenger-km (Road)',
        'pkt_rail': 'Passenger-km (Rail)',
        'pkt_subway': 'Passenger-km (Subway)',
        'opex_pkt_road': 'OPEX/pkm Road [€]',
        'opex_pkt_rail': 'OPEX/pkm Rail [€]',
        'opex_pkt_subway': 'OPEX/pkm Subway [€]',
        'opex_road': 'Total OPEX (Road) [€]',
        'opex_rail': 'Total OPEX (Rail) [€]',
        'opex_subway': 'Total OPEX (Subway) [€]'
    }
    
    # Create final display dataframe
    df_display = df_display_base[list(display_cols.keys())].rename(columns=display_cols).copy()
    
    # Capitalize Entity names for display
    df_display['Entity'] = df_display['Entity'].str.title()
    # Manual fix for EU-27 capitalization
    df_display['Entity'] = df_display['Entity'].replace({'Eu-27': 'EU-27', 'Espanha': 'Spain'})
    
    # Calculate Total Addressable OPEX
    df_display['Total Addressable OPEX [€]'] = (
        df_display['Total OPEX (Road) [€]'].fillna(0) + 
        df_display['Total OPEX (Rail) [€]'].fillna(0) + 
        df_display['Total OPEX (Subway) [€]'].fillna(0)
    )

    # Render Table
    st.dataframe(
        df_display.style.format({
            'Passenger-km (Road)': '{:,.0f}',
            'Passenger-km (Rail)': '{:,.0f}',
            'Passenger-km (Subway)': '{:,.0f}',
            'OPEX/pkm Road [€]': '{:.4f}',
            'OPEX/pkm Rail [€]': '{:.4f}',
            'OPEX/pkm Subway [€]': '{:.4f}',
            'Total OPEX (Road) [€]': '{:,.0f}',
            'Total OPEX (Rail) [€]': '{:,.0f}',
            'Total OPEX (Subway) [€]': '{:,.0f}',
            'Total Addressable OPEX [€]': '{:,.0f}'
        }),
        width='stretch',
        hide_index=True
    )

    # ---------------------------------------------------------
    # REST OF THE APP (Cleaning for logic)
    # ---------------------------------------------------------
    # We use the original df_data for calculations, but clean it similarly
    df_data = df_data.dropna(subset=['entity'])
    
    opex_cols_clean = ['opex_road', 'opex_rail', 'opex_subway']
    for col in opex_cols_clean:
        if col in df_data.columns:
            df_data[col] = pd.to_numeric(df_data[col], errors='coerce').fillna(0.0)
            
    try:
        carris_row = df_data[df_data['entity'].str.lower() == 'carris'].iloc[0]
    except IndexError:
        st.error("Entity 'Carris' not found in uploaded file.")
        st.stop()

    # ---------------------------------------------------------
    # PART A: MARKET SIZING
    # ---------------------------------------------------------
    st.header("A. Market Sizing Analysis (TAM/SAM/SOM)")
    tab_total, tab_road, tab_rail, tab_subway = st.tabs(["Total Market", "Road Only", "Rail Only", "Subway Only"])
    
    def render_market_tab(scope_name, col_list, capture_for_pdf=False):
        results_scope = {}
        stats_scope = []
        
        # Only iterate through the specific entities we care about for the charts to keep them clean
        valid_chart_entities = ['carris', 'portugal', 'espanha', 'iberia', 'eu-27']
        
        for index, row in df_data.iterrows():
            entity_name = str(row['entity']).lower().strip()
            
            # Filter for chart clarity
            if entity_name in valid_chart_entities:
                scope_opex = row[col_list].sum()
                if scope_opex > 0:
                    sim_res = run_monte_carlo(scope_opex, dt_min, dt_mode, dt_max)
                    results_scope[row['entity']] = sim_res
                    mean_val = np.mean(sim_res)
                    stats_scope.append({
                        'Entity': row['entity'],
                        'Mean': mean_val,
                        'P05': np.percentile(sim_res, 5),
                        'P95': np.percentile(sim_res, 95)
                    })
                    
                    # Capture data for PDF Table if this is the 'Total' view
                    if capture_for_pdf:
                        pdf_table_data.append({
                             "Entity": row['entity'].capitalize(),
                             "opex": currency_fmt(scope_opex),
                             "savings": currency_fmt(mean_val)
                         })
        
        if not stats_scope:
            st.warning(f"No data found for {scope_name}.")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Risk Profiles ({scope_name})")
            entities = list(results_scope.keys())
            fig1, axes = plt.subplots(len(entities), 1, figsize=(6, 3 * len(entities)))
            if len(entities) == 1: axes = [axes]
            for i, entity in enumerate(entities):
                ax = axes[i]
                sns.histplot(results_scope[entity], bins=40, kde=True, color=sns.color_palette("viridis", len(entities))[i], ax=ax)
                ax.set_title(f"{entity.capitalize()}")
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(currency_fmt))
            plt.tight_layout()
            st.pyplot(fig1)
            # Capture Plot
            if capture_for_pdf:
                pdf_plots[f'risk_{scope_name}'] = save_plot_to_temp(fig1)

        with col2:
            st.subheader(f"Market Potential ({scope_name})")
            def get_stat(name):
                for s in stats_scope:
                    if str(s['Entity']).lower().strip() == name.lower(): return s
                return None
                
            market_data = []
            hierarchy_entities = [('carris', "SOM (Carris)"), ('portugal', "SAM (Portugal)"), ('iberia', "TAM (Iberia)"), ('eu-27', "PAM (EU-27)")]
            
            for ent_key, label in hierarchy_entities:
                stat = get_stat(ent_key)
                if stat: market_data.append({"Label": label, "Stat": stat})
            
            if market_data:
                labels = [m['Label'] for m in market_data]
                means = [m['Stat']['Mean'] for m in market_data]
                p05s = [m['Stat']['P05'] for m in market_data]
                p95s = [m['Stat']['P95'] for m in market_data]
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                bars = ax2.bar(labels, means, width=0.5, color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'], edgecolor='black')
                ax2.errorbar(labels, means, yerr=[[m-p for m,p in zip(means, p05s)], [p-m for m,p in zip(means, p95s)]], fmt='none', ecolor='black', capsize=5)
                ax2.set_yscale('log')
                ax2.yaxis.set_major_formatter(ticker.FuncFormatter(currency_fmt))
                for i, bar in enumerate(bars):
                    text_x = bar.get_x() + bar.get_width() + 0.05
                    ax2.text(text_x, means[i], f"{currency_fmt(means[i])} (Mean)", fontweight='bold', va='center', fontsize=9)
                    ax2.text(text_x, p95s[i], f"{currency_fmt(p95s[i])} (P95)", va='center', fontsize=8, color='#333333')
                    ax2.text(text_x, p05s[i], f"{currency_fmt(p05s[i])} (P05)", va='center', fontsize=8, color='#333333')
                plt.tight_layout()
                st.pyplot(fig2)
                # Capture Plot
                if capture_for_pdf:
                    pdf_plots[f'potential_{scope_name}'] = save_plot_to_temp(fig2)

    with tab_total: render_market_tab("Total", ['opex_road', 'opex_rail', 'opex_subway'], capture_for_pdf=True)
    with tab_road: render_market_tab("Road", ['opex_road'])
    with tab_rail: render_market_tab("Rail", ['opex_rail'])
    with tab_subway: render_market_tab("Subway", ['opex_subway'])

    # ---------------------------------------------------------
    # PART B: FINANCIAL ANALYSIS
    # ---------------------------------------------------------
    st.markdown("---")
    st.header(f"B. Financial Analysis (Carris - {analysis_years} Year Horizon)")
    
    carris_opex = carris_row['opex_road'] 
    np.random.seed(42)
    sim_savings_pct = np.random.triangular(dt_min, dt_mode, dt_max, size=50000)
    base_annual_savings = carris_opex * sim_savings_pct
    
    company_capex = float(company_capex_input)
    company_opex = float(company_opex_input)
    
    client_saas_npv = np.full(50000, -float(setup_fee), dtype=float)
    company_saas_npv = np.full(50000, float(setup_fee) - company_capex, dtype=float)
    client_pb_npv = np.zeros(50000, dtype=float)
    company_pb_npv = np.full(50000, -company_capex, dtype=float)

    company_saas_profit = np.zeros(50000, dtype=float)
    company_pb_profit = np.zeros(50000, dtype=float)
    
    cf_paths_saas = np.zeros(analysis_years + 1)
    cf_paths_pb = np.zeros((50000, analysis_years + 1))
    cf_paths_saas[0] = float(setup_fee) - company_capex
    cf_paths_pb[:, 0] = -company_capex

    current_savings = base_annual_savings.copy()
    for year in range(1, int(analysis_years) + 1):
        if year > 1:
            current_savings = current_savings * (1 - decay_factor)
        
        cf_c_saas = current_savings - annual_fee
        client_saas_npv += cf_c_saas / ((1 + discount_rate_input) ** year)
        cf_comp_saas = annual_fee - company_opex
        company_saas_npv += cf_comp_saas / ((1 + discount_rate_input) ** year)
        company_saas_profit += cf_comp_saas
        cf_paths_saas[year] = cf_paths_saas[year-1] + cf_comp_saas
        
        cf_c_pb = current_savings * (1 - pb_cut)
        client_pb_npv += cf_c_pb / ((1 + discount_rate_input) ** year)
        cf_comp_pb = (current_savings * pb_cut) - company_opex
        company_pb_npv += cf_comp_pb / ((1 + discount_rate_input) ** year)
        company_pb_profit += cf_comp_pb
        cf_paths_pb[:, year] = cf_paths_pb[:, year-1] + cf_comp_pb

    # ROI Calculations Company Side
    #it is only subtracted the company capex because the company_pb_profit and company_saas_profit already includes the dectuction of opex
    inv_comp = company_capex + (company_opex * analysis_years)
    comp_saas_roi = (company_saas_profit + setup_fee - company_capex) / inv_comp * 100
    comp_pb_roi = (company_pb_profit - company_capex)/ inv_comp * 100

    tab1, tab2, tab3, tab4 = st.tabs(["NPV Comparison", "Company ROI", "Cumulative Cash Flow", "Summary Table"])
    
    with tab1:
        fig_npv, axes_npv = plt.subplots(1, 2, figsize=(14, 6))
        def annotate_plot(ax, data, color):
            m, p05, p95 = np.mean(data), np.percentile(data, 5), np.percentile(data, 95)
            ylim = ax.get_ylim()
            ax.axvline(m, color=color, linestyle='--', linewidth=1.5)
            ax.text(m, ylim[1]*0.9, f"Mean: {currency_fmt(m)}", color=color, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            ax.axvline(p05, color=color, linestyle=':', linewidth=1); ax.text(p05, ylim[1]*0.5, f"P5: {currency_fmt(p05)}", color=color, ha='right', rotation=90)
            ax.axvline(p95, color=color, linestyle=':', linewidth=1); ax.text(p95, ylim[1]*0.5, f"P95: {currency_fmt(p95)}", color=color, ha='left', rotation=90)

        sns.kdeplot(client_saas_npv, fill=True, label='SaaS', color='blue', alpha=0.2, ax=axes_npv[0])
        sns.kdeplot(client_pb_npv, fill=True, label='Perf. Based', color='orange', alpha=0.2, ax=axes_npv[0])
        annotate_plot(axes_npv[0], client_pb_npv, 'darkorange')
        axes_npv[0].set_title("Client (Carris) NPV"); axes_npv[0].xaxis.set_major_formatter(ticker.FuncFormatter(currency_fmt)); axes_npv[0].legend()

        axes_npv[1].axvline(company_saas_npv[0], color='blue', linewidth=3, label=f'SaaS Fixed: {currency_fmt(company_saas_npv[0])}')
        sns.kdeplot(company_pb_npv, fill=True, color='orange', label='Perf. Based', alpha=0.2, ax=axes_npv[1])
        annotate_plot(axes_npv[1], company_pb_npv, 'darkorange')
        axes_npv[1].set_title("Company NPV"); axes_npv[1].xaxis.set_major_formatter(ticker.FuncFormatter(currency_fmt)); axes_npv[1].axvline(0, color='red', linestyle='--'); axes_npv[1].legend()
        st.pyplot(fig_npv)
        
        # Capture for PDF
        pdf_plots['fin_npv'] = save_plot_to_temp(fig_npv)
        
    with tab2:
        fig_roi, ax_roi = plt.subplots(figsize=(10, 5))
        ax_roi.axvline(comp_saas_roi[0], color='blue', linewidth=3, label=f'SaaS Fixed: {comp_saas_roi[0]:.1f}%')
        sns.kdeplot(comp_pb_roi, fill=True, color='orange', label='Perf. Based', alpha=0.3, ax=ax_roi)
        m_r, p05_r, p95_r = np.mean(comp_pb_roi), np.percentile(comp_pb_roi, 5), np.percentile(comp_pb_roi, 95)
        yl = ax_roi.get_ylim()
        ax_roi.axvline(m_r, color='darkorange', linestyle='--', linewidth=1.5); ax_roi.text(m_r, yl[1]*0.9, f"Mean: {m_r:.1f}%", color='darkorange', ha='center', fontweight='bold')
        ax_roi.axvline(p05_r, color='darkorange', linestyle=':', linewidth=1); ax_roi.text(p05_r, yl[1]*0.6, f"P5: {p05_r:.1f}%", color='darkorange', ha='right', rotation=90)
        ax_roi.axvline(p95_r, color='darkorange', linestyle=':', linewidth=1); ax_roi.text(p95_r, yl[1]*0.6, f"P95: {p95_r:.1f}%", color='darkorange', ha='left', rotation=90)
        ax_roi.axvline(0, color='red', linestyle='--'); ax_roi.set_title("Company ROI Distribution"); ax_roi.legend()
        st.pyplot(fig_roi)

        # Capture for PDF
        pdf_plots['fin_roi'] = save_plot_to_temp(fig_roi)

    with tab3:
        st.subheader("Company Cumulative Cash Flow (Undiscounted)")
        fig_cf, ax_cf = plt.subplots(figsize=(10, 6))
        x_years = np.arange(0, analysis_years + 1)
        p05_cf, p50_cf, p95_cf = np.percentile(cf_paths_pb, 5, axis=0), np.percentile(cf_paths_pb, 50, axis=0), np.percentile(cf_paths_pb, 95, axis=0)
        ax_cf.plot(x_years, p50_cf, color='darkorange', linewidth=3, label='PB Median Path')
        ax_cf.fill_between(x_years, p05_cf, p95_cf, color='orange', alpha=0.2, label='PB 90% Confidence Interval')
        ax_cf.plot(x_years, cf_paths_saas, color='blue', linewidth=3, marker='o', label='SaaS Path')
        ax_cf.axhline(0, color='red', linestyle='--', alpha=0.6, label='Break Even')
        ax_cf.set_xlabel("Years"); ax_cf.set_ylabel("Cumulative Cash (€)"); ax_cf.yaxis.set_major_formatter(ticker.FuncFormatter(currency_fmt))
        ax_cf.set_xticks(x_years); ax_cf.legend(); ax_cf.grid(True, alpha=0.3)
        st.pyplot(fig_cf)

        # Capture for PDF
        pdf_plots['fin_cf'] = save_plot_to_temp(fig_cf)

    with tab4:
        st.table(pd.DataFrame({
            "Metric": ["Client NPV (Mean)", "Company NPV (Mean)", "Company ROI (Mean)", "Company Loss Prob"],
            "SaaS Model": [currency_fmt(np.mean(client_saas_npv)), currency_fmt(np.mean(company_saas_npv)), f"{np.mean(comp_saas_roi):.1f}%", "0.0%"],
            "Perf. Based": [currency_fmt(np.mean(client_pb_npv)), currency_fmt(np.mean(company_pb_npv)), f"{np.mean(comp_pb_roi):.1f}%", f"{np.mean(comp_pb_roi < 0)*100:.1f}%"]
        }))

    # ---------------------------------------------------------
    # DOWNLOAD BUTTON
    # ---------------------------------------------------------
    st.sidebar.markdown("---")
    if st.sidebar.button("📄 Generate PDF Report"):
        # 1. Gather all inputs from the current run
        current_inputs = {
            "dt_min": dt_min, "dt_mode": dt_mode, "dt_max": dt_max,
            "analysis_years": int(analysis_years), "discount_rate": discount_rate_input,
            "setup_fee": setup_fee, "annual_fee": annual_fee,
            "pb_cut": pb_cut, "decay_factor": decay_factor,
            "company_capex": company_capex_input, "company_opex": company_opex_input
        }
        
        # 2. Create PDF
        # Note: formatting happens inside create_pdf now
        pdf = create_pdf(current_inputs, pdf_plots, pdf_table_data)
        
        # 3. Save and Download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf.output(tmp_file.name)
            with open(tmp_file.name, "rb") as file:
                st.sidebar.download_button(
                    label="📥 Download PDF",
                    data=file,
                    file_name="Digital_Twin_Analysis_Report.pdf",
                    mime="application/pdf"
                )