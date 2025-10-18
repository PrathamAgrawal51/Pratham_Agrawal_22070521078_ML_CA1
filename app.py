import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer, LabelEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_curve, auc

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="COVID-19 Analysis & Prediction Dashboard",
    page_icon="⚕️",
    layout="wide",
)

# --- 2. STYLING ---
# A stable and professional CSS theme
custom_css = """
<style>
    .stApp { background-color: #F0F2F6; }
    .main .block-container { padding: 1rem 3rem 3rem; }
    [data-testid="stSidebar"] { background-color: #0F172A; }
    [data-testid="stSidebar"] * { color: #FFFFFF; }
    [data-testid="stSidebar"] div[data-baseweb="select"] div { color: black !important; }
    [data-testid="stMetric"] {
        background-image: linear-gradient(140deg, #1E3A8A 0%, #0F172A 100%);
        border: 1px solid #1E3A8A;
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    [data-testid="stMetricLabel"] p { color: #E0E0E0 !important; }
    [data-testid="stMetricValue"] { color: #FFFFFF !important; }
    .st-emotion-cache-1r4qj8v { border: 1px solid #E0E0E0; border-radius: 10px; padding: 20px; }
    .stButton>button { border-radius: 20px; border: 1px solid #2563EB; background-color: #2563EB; color: white; font-weight: bold; }

    /* --- NEW CSS FOR AUTHOR BAR --- */
    .author-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 20px;
        background-color: #0F172A; /* Match sidebar */
        border-radius: 10px;
        margin-top: 1rem;
        margin-bottom: 2rem; /* Space below the bar */
    }
    .author-bar-text {
        color: white;
    }
    .author-bar-text h5, .author-bar-text p {
        margin: 0;
        padding: 0;
        line-height: 1.2;
    }
    .author-bar-icons a {
        text-decoration: none;
        margin-left: 15px; /* space between icons */
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 3. DATA LOADING & PREPARATION ---
@st.cache_data
def load_data(filepath):
    """Loads, preprocesses, and caches the dataset."""
    try:
        df = pd.read_csv(filepath)
        df['year'] = df['year'].astype(str)
        qt = QuantileTransformer(output_distribution='uniform', n_quantiles=4)
        df['risk_level_q'] = qt.fit_transform(df[['excessmean']])
        bin_labels = ['Low', 'Medium', 'High', 'Critical']
        df['risk_level'] = pd.cut(df['risk_level_q'], bins=4, labels=bin_labels, include_lowest=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Please place it in the same directory.")
        return None

# --- 4. MACHINE LEARNING MODEL TRAINING & DATA PREP ---
@st.cache_resource
def get_ml_assets(_df):
    """Prepares data splits and trains the best models."""
    features = ['country', 'sex', 'age_group', 'year']
    X = _df[features]
    
    # --- Regression Assets ---
    y_reg = _df['excessmean']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    reg_preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), features)])
    
    # Best model: Gradient Boosting
    gbr_pipeline = Pipeline(steps=[('preprocessor', reg_preprocessor), ('regressor', GradientBoostingRegressor(random_state=42))])
    gbr_pipeline.fit(X_train_reg, y_train_reg)
    
    # Baseline model: Linear Regression for comparison plot
    lr_pipeline = Pipeline(steps=[('preprocessor', reg_preprocessor), ('regressor', LinearRegression())])
    lr_pipeline.fit(X_train_reg, y_train_reg)

    # --- Classification Assets ---
    y_clf = _df['risk_level']
    le = LabelEncoder()
    y_clf_encoded = le.fit_transform(y_clf)
    X_train_clf, X_test_clf, y_train_clf_encoded, y_test_clf_encoded = train_test_split(X, y_clf_encoded, test_size=0.2, random_state=42, stratify=y_clf_encoded)
    
    # Best model: MLP Classifier
    clf_preprocessor_nn = ColumnTransformer(transformers=[('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore')), ('scaler', StandardScaler(with_mean=False))]), features)])
    mlp_pipeline = Pipeline(steps=[('preprocessor', clf_preprocessor_nn), ('classifier', MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(100, 50), early_stopping=True))])
    mlp_pipeline.fit(X_train_clf, y_train_clf_encoded)
    
    # Baseline model: Logistic Regression for ROC curve comparison
    logreg_pipeline = Pipeline(steps=[('preprocessor', reg_preprocessor), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
    logreg_pipeline.fit(X_train_clf, y_train_clf_encoded)
    
    return {
        "regressor": gbr_pipeline, "classifier": mlp_pipeline, "label_encoder": le,
        "X_test_reg": X_test_reg, "y_test_reg": y_test_reg, "baseline_regressor": lr_pipeline,
        "X_test_clf": X_test_clf, "y_test_clf_encoded": y_test_clf_encoded, "baseline_classifier": logreg_pipeline
    }

# --- 5. UI PAGE DEFINITIONS ---

def show_about_page():
    st.title("📖 About This Project")
    
    # --- NEW AUTHOR BAR ---
    author_bar_html = """
        <div class="author-bar">
            <div class="author-bar-text">
                <h5>Pratham Agrawal</h5>
                <p>PRN: 22070521078</p>
            </div>
            <div class="author-bar-icons">
                <a href="https://www.linkedin.com/in/prathamagrawal51/" target="_blank" title="LinkedIn">
                    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" fill="white" class="bi bi-linkedin" viewBox="0 0 16 16">
                        <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854zm4.943 12.248V6.169H2.542v7.225zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248S2.4 3.226 2.4 3.934c0 .694.521 1.248 1.327 1.248zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016l.016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225z"/>
                    </svg>
                </a>
                <a href="https://github.com/PrathamAgrawal51" target="_blank" title="GitHub">
                    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" fill="white" class="bi bi-github" viewBox="0 0 16 16">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8"/>
                    </svg>
                </a>
            </div>
        </div>
    """
    st.markdown(author_bar_html, unsafe_allow_html=True)
    
    st.markdown("This dashboard is a unified submission for two academic subjects: **Data Science (DS)** and **Machine Learning (ML)**.")

    with st.container(border=True):
        st.subheader("🎯 Project Objectives")
        st.markdown("""
        - **For Data Science (DS):** To analyze the open-source WHO COVID-19 Excess Deaths dataset and generate interactive visual reports to support decision-making.
        - **For Machine Learning (ML):** To showcase the findings of a comprehensive modeling process (documented in a Colab notebook) and provide a predictive tool based on the best-performing models.
        """)

    with st.container(border=True):
        st.subheader("💾 Data Source")
        st.markdown("""
        The analysis is based on the **Global Excess Deaths Associated with COVID-19 (Modelled Estimates)** dataset, provided by the **World Health Organization (WHO)**.
        - **[Link to Dataset](https://www.who.int/data/sets/global-excess-deaths-associated-with-covid-19-modelled-estimates)**
        """)
        
    with st.container(border=True):
        st.subheader("🛠️ Technology Stack")
        st.markdown("""
        - **Programming Language:** Python
        - **Dashboard Framework:** Streamlit
        - **Data Manipulation:** Pandas
        - **Machine Learning:** Scikit-learn
        - **Data Visualization:** Plotly Express, Plotly Go
        """)

    with st.container(border=True):
        st.subheader("🗺️ Dashboard Guide")
        st.markdown("""
        - **Global EDA:** An interactive dashboard for high-level analysis and exploring data with filters.
        - **Comparative Analysis:** A tool to compare two entities (countries or global average) side-by-side.
        - **ML Analysis:** A summary of the machine learning results, including performance tables and key visualizations.
        - **Future Risk Predictor:** An interactive tool that uses our best-trained models to forecast future pandemic vulnerability.
        """)

def show_eda_page(df):
    st.title("📊 Global EDA for Decision-Making")
    st.markdown("This page provides high-level summaries and interactive reports to analyze the pandemic's impact.")
    
    st.header("Global & India-Specific Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Snapshot")
        global_total_deaths = int(df['excessmean'].sum())
        top_country_overall = df.groupby('country')['excessmean'].sum().idxmax()
        st.metric("Global Total Excess Deaths", f"{global_total_deaths:,}")
        st.metric("Country with Highest Deaths", top_country_overall)

    with col2:
        st.subheader("Spotlight: India")
        india_df = df[df['country'] == 'India']
        india_total_deaths = int(india_df['excessmean'].sum())
        india_top_age_group = india_df.groupby('age_group')['excessmean'].sum().idxmax()
        st.metric("Total Excess Deaths in India", f"{india_total_deaths:,}")
        st.metric("Most Affected Age Group in India", india_top_age_group)
    
    st.divider()
    
    st.subheader("Global Context Visualizations")
    col3, col4 = st.columns([3, 2])
    with col3:
        st.markdown("##### Geographic Impact: World Map")
        country_deaths = df.groupby('country')['excessmean'].sum().reset_index()
        fig_map = px.choropleth(country_deaths, locations="country", locationmode='country names', color="excessmean",
                                hover_name="country", color_continuous_scale=px.colors.sequential.Reds,
                                title="Global Distribution of Excess Deaths")
        st.plotly_chart(fig_map, use_container_width=True)
    with col4:
        st.markdown("##### Top 15 Countries by Mortality")
        top_countries = country_deaths.sort_values(by='excessmean', ascending=False).head(15)
        fig_treemap = px.treemap(top_countries, path=['country'], values='excessmean', title="Composition of Top 15 Countries")
        st.plotly_chart(fig_treemap, use_container_width=True)

    st.divider()
    st.header("Interactive Visual Reports")

    with st.sidebar:
        st.header("Dashboard Filters")
        countries = ['All'] + sorted(df['country'].unique().tolist())
        default_country_index = countries.index('India') if 'India' in countries else 0
        selected_country = st.selectbox("Country", countries, index=default_country_index)
        years = ['All'] + sorted(df['year'].unique().tolist())
        selected_year = st.selectbox("Year", years)
        sexes = ['All'] + sorted(df['sex'].unique().tolist())
        selected_sex = st.selectbox("Sex", sexes)

    filtered_df = df.copy()
    if selected_country != 'All': filtered_df = filtered_df[filtered_df['country'] == selected_country]
    if selected_year != 'All': filtered_df = filtered_df[filtered_df['year'] == selected_year]
    if selected_sex != 'All': filtered_df = filtered_df[filtered_df['sex'] == selected_sex]
    
    st.subheader(f"Filtered Results for: {selected_country}")
    filtered_total_deaths = int(filtered_df['excessmean'].sum())
    st.metric("Total Excess Deaths (Filtered Selection)", f"{filtered_total_deaths:,}")

    st.markdown("---")
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Mortality Trend Over Time")
        time_data = filtered_df.groupby('year')['excessmean'].sum().reset_index()
        fig_time = px.line(time_data, x='year', y='excessmean', markers=True, title="Trend for Filtered Selection")
        st.plotly_chart(fig_time, use_container_width=True)
    with col6:
        st.subheader("Demographic Breakdown by Age & Sex")
        demo_data = filtered_df.groupby(['age_group', 'sex'])['excessmean'].sum().reset_index()
        fig_demo = px.bar(demo_data, x='age_group', y='excessmean', color='sex', barmode='group', title="Age & Sex Breakdown")
        st.plotly_chart(fig_demo, use_container_width=True)
        
    col7, col8 = st.columns(2)
    with col7:
        st.subheader("Year-over-Year Comparison")
        yoy_data = filtered_df.groupby('year')['excessmean'].sum().reset_index()
        fig_yoy = px.bar(yoy_data, x='year', y='excessmean', color='year', title="2020 vs 2021 Comparison")
        st.plotly_chart(fig_yoy, use_container_width=True)
    with col8:
        st.subheader("Mortality Distribution by Sex")
        sex_data = filtered_df.groupby('sex')['excessmean'].sum().reset_index()
        fig_donut = px.pie(sex_data, names='sex', values='excessmean', hole=0.4, title="Male vs Female Distribution")
        st.plotly_chart(fig_donut, use_container_width=True)


def show_comparative_analysis_page(df):
    st.title("⚖️ Comparative Analysis")
    st.markdown("Select two entities to compare their pandemic impact side-by-side. You can compare two countries, or a country against the global average.")
    
    col1, col2 = st.columns(2)
    options = ['Global Average'] + sorted(df['country'].unique().tolist())
    selection1 = col1.selectbox("Select Entity 1", options, index=options.index('India'))
    selection2 = col2.selectbox("Select Entity 2", options, index=0)

    def get_stats(selection):
        if selection == 'Global Average':
            avg_deaths = df.groupby(['year', 'sex', 'age_group'])['excessmean'].mean().reset_index()
            return avg_deaths, int(df['excessmean'].sum() / len(df['country'].unique()))
        else:
            country_df = df[df['country'] == selection]
            return country_df, int(country_df['excessmean'].sum())

    df1, total1 = get_stats(selection1)
    df2, total2 = get_stats(selection2)
    
    st.header("Comparative Metrics")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.subheader(selection1)
        st.metric(f"Total Excess Deaths", f"{total1:,}")
    with mcol2:
        st.subheader(selection2)
        st.metric(f"Total Excess Deaths", f"{total2:,}")
    
    st.divider()
    st.header("Comparative Visualizations")
    
    ccol1, ccol2 = st.columns(2)
    with ccol1:
        st.subheader("Trend Comparison")
        trend1 = df1.groupby('year')['excessmean'].sum().reset_index()
        trend2 = df2.groupby('year')['excessmean'].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend1['year'], y=trend1['excessmean'], mode='lines+markers', name=selection1))
        fig.add_trace(go.Scatter(x=trend2['year'], y=trend2['excessmean'], mode='lines+markers', name=selection2))
        st.plotly_chart(fig, use_container_width=True)
    with ccol2:
        st.subheader("Demographic Comparison")
        demo1 = df1.groupby('age_group')['excessmean'].sum().reset_index()
        demo2 = df2.groupby('age_group')['excessmean'].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=demo1['age_group'], y=demo1['excessmean'], name=selection1))
        fig.add_trace(go.Bar(x=demo2['age_group'], y=demo2['excessmean'], name=selection2))
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)

def show_ml_analysis_page(ml_assets):
    st.title("⚙️ ML Model Analysis & Findings")
    st.markdown("A summary of the comprehensive modeling process performed in the Colab notebook.")
    
    reg_tab, clf_tab = st.tabs(["Regression Analysis", "Classification Analysis"])
    with reg_tab:
        st.subheader("Regression Model Performance")
        st.markdown("""
        **Understanding the Metrics:**
        - **R-squared (R²):** Represents the percentage of the variance in the excess deaths that our model can explain. A score of 0.87 means the model explains 87% of the data's behavior. Higher is better.
        - **MAE (Mean Absolute Error):** The average absolute difference between the actual and predicted deaths. It tells us, on average, how many deaths our prediction was off by. Lower is better.
        - **MSE (Mean Squared Error):** Similar to MAE, but it squares the differences before averaging. This penalizes larger errors much more heavily.
        - **RMSE (Root Mean Squared Error):** The square root of the MSE. Its units are the same as the target variable (number of deaths), making it easier to interpret. Lower is better.
        """)
        # --- UPDATED REGRESSION DATA TABLE ---
        reg_data = {
            'Model': ['Gradient Boosting Regressor', 'XGBoost Regressor', 'Decision Tree Regressor', 'Random Forest Regressor', 'Multiple Linear Regression', 'Lasso Regression', 'Ridge Regression', 'MLP Regressor (Neural Network)', 'Simple Linear Regression', 'K-Neighbors Regressor'],
            'Adjusted R-squared (R²)': [0.8718, 0.8696, 0.8598, 0.8199, 0.4614, 0.4612, 0.4512, 0.3490, 0.0055, -0.0596],
            'MAE': [1993.16, 1533.67, 1407.98, 1426.44, 3318.10, 3299.87, 3314.15, 2187.92, 4078.98, 5006.59],
            'MSE': [62019930, 63092820, 67845617, 87187887, 260592167, 260682247, 265507119, 314986860, 481177699, 512686161],
            'RMSE': [7875.27, 7943.10, 8236.84, 9337.44, 16142.87, 16145.66, 16294.39, 17747.87, 21935.76, 22642.57]
        }
        reg_df = pd.DataFrame(reg_data)
        st.dataframe(reg_df.style.format({
            'R-squared (R²)': '{:.4f}',
            'MAE': '{:,.2f}',
            'MSE': '{:,.0f}',
            'RMSE': '{:,.2f}'
        }))
        st.success("**Conclusion:** The **Gradient Boosting Regressor** was the best model, explaining ~87.2% of the variance in the data.", icon="🏆")

        with st.expander("Show Key Visualizations"):
            st.subheader("Actual vs. Predicted: Best vs. Baseline")
            vcol1, vcol2 = st.columns(2)
            y_pred_best = ml_assets['regressor'].predict(ml_assets['X_test_reg'])
            y_pred_base = ml_assets['baseline_regressor'].predict(ml_assets['X_test_reg'])
            
            fig_base = px.scatter(x=ml_assets['y_test_reg'], y=y_pred_base, title="Baseline (Linear Regression)")
            fig_base.add_shape(type='line', x0=0, y0=0, x1=ml_assets['y_test_reg'].max(), y1=ml_assets['y_test_reg'].max(), line=dict(color='red', dash='dash'))
            vcol1.plotly_chart(fig_base, use_container_width=True)

            fig_best = px.scatter(x=ml_assets['y_test_reg'], y=y_pred_best, title="Best Model (Gradient Boosting)")
            fig_best.add_shape(type='line', x0=0, y0=0, x1=ml_assets['y_test_reg'].max(), y1=ml_assets['y_test_reg'].max(), line=dict(color='red', dash='dash'))
            vcol2.plotly_chart(fig_best, use_container_width=True)
            st.info("Notice how the points in the 'Best Model' plot are much tighter around the red line, visually confirming its superior accuracy.")

    with clf_tab:
        st.subheader("Classification Model Performance")
        st.markdown("""
        **Understanding the Metrics:**
        - **Accuracy:** The percentage of total predictions that the model got right.
        - **Precision:** Of all the times the model predicted a certain risk level (e.g., 'Critical'), what percentage were actually correct?
        - **Recall:** Of all the actual 'Critical' risk cases in the data, what percentage did the model correctly identify?
        - **F1-score:** The harmonic mean of Precision and Recall, providing a single score that balances both metrics.
        *(Note: For our multi-class problem, these scores are the weighted average across all four risk levels.)*
        """)
        # --- UPDATED CLASSIFICATION DATA TABLE ---
        clf_data = {
            'Model': ['MLP Classifier (Neural Network)', 'Support Vector Classifier (SVC)', 'Logistic Regression', 'Random Forest Classifier', 'K-Neighbors Classifier'],
            'Accuracy': [0.9098, 0.8776, 0.8768, 0.7480, 0.6312],
            'Precision': [0.9093, 0.8777, 0.8761, 0.7562, 0.6415],
            'Recall': [0.9098, 0.8776, 0.8768, 0.7480, 0.6312],
            'F1-score': [0.9093, 0.8772, 0.8762, 0.7493, 0.6331]
        }
        clf_df = pd.DataFrame(clf_data)
        st.dataframe(clf_df.style.format('{:.4f}', subset=['Accuracy', 'Precision', 'Recall', 'F1-score']))
        st.success("**Conclusion:** The **MLP Classifier** was the best model with ~91% accuracy.", icon="🏆")
        
        with st.expander("Show Key Visualizations"):
            st.subheader("ROC Curve: Best vs. Baseline")
            y_test_bin = label_binarize(ml_assets['y_test_clf_encoded'], classes=range(len(ml_assets['label_encoder'].classes_)))
            
            prob_base = ml_assets['baseline_classifier'].predict_proba(ml_assets['X_test_clf'])
            prob_best = ml_assets['classifier'].predict_proba(ml_assets['X_test_clf'])
            
            fpr_base, tpr_base, _ = roc_curve(y_test_bin.ravel(), prob_base.ravel())
            auc_base = auc(fpr_base, tpr_base)

            fpr_best, tpr_best, _ = roc_curve(y_test_bin.ravel(), prob_best.ravel())
            auc_best = auc(fpr_best, tpr_best)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr_base, y=tpr_base, mode='lines', name=f'Baseline (LogReg) (AUC = {auc_base:.2f})'))
            fig_roc.add_trace(go.Scatter(x=fpr_best, y=tpr_best, mode='lines', name=f'Best Model (MLP) (AUC = {auc_best:.2f})'))
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig_roc.update_layout(title="ROC Curve Comparison", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)
            st.info("The Area Under the Curve (AUC) measures how well a model can distinguish between classes. The MLP Classifier's curve is closer to the top-left, confirming its superior performance.")

def show_predictor_page(df, ml_assets):
    st.title("🔮 Future Pandemic Vulnerability Predictor")
    st.markdown("This tool uses the best-performing models from our analysis to forecast risk.")
    st.info("The predictor uses the **Gradient Boosting Regressor** and the **MLP Classifier**.", icon="🤖")

    reg_model, clf_model, le = ml_assets['regressor'], ml_assets['classifier'], ml_assets['label_encoder']

    st.header("Select a Demographic to Forecast")
    
    countries = sorted(df['country'].unique())
    sexes = sorted(df['sex'].unique())
    age_groups = sorted(df['age_group'].unique())

    country_index = countries.index('India') if 'India' in countries else 0
    sex_index = sexes.index('Male') if 'Male' in sexes else 0
    age_group_index = age_groups.index('0-24') if '0-24' in age_groups else 0

    col1, col2, col3 = st.columns(3)
    with col1: 
        pred_country = st.selectbox("Select Country", countries, index=country_index)
    with col2: 
        pred_sex = st.selectbox("Select Sex", sexes, index=sex_index)
    with col3: 
        pred_age = st.selectbox("Select Age Group", age_groups, index=age_group_index)

    if st.button("Predict Vulnerability", type="primary"):
        input_data = pd.DataFrame({ 'country': [pred_country], 'sex': [pred_sex], 'age_group': [pred_age], 'year': ['2021'] })
        
        predicted_deaths = reg_model.predict(input_data)[0]
        predicted_risk_encoded = clf_model.predict(input_data)[0]
        predicted_risk_label = le.inverse_transform([predicted_risk_encoded])[0]

        st.divider()
        st.subheader("Vulnerability Report Card")
        with st.container(border=True):
            st.markdown(f"#### For **{pred_sex}s** aged **{pred_age}** in **{pred_country}**:")
            result_col1, result_col2 = st.columns(2)
            result_col1.metric("Predicted Excess Deaths", f"{int(predicted_deaths):,}")
            result_col2.metric("Predicted Mortality Risk Level", predicted_risk_label)
            st.success(f"""
            **Finding & Recommendation:** Based on historical patterns, this demographic is forecast to have a **{predicted_risk_label.lower()}** mortality risk. 
            Public health strategies should consider prioritizing resources and protective measures for this group in future public health crises.
            """, icon="📈")

# --- 6. MAIN APP LOGIC ---
def main():
    df = load_data('cleaned_who_excess_deaths.csv')

    if df is not None:
        ml_assets = get_ml_assets(df)
        with st.sidebar:
            st.markdown("## Navigation")
            page = st.radio("Go to", ["About", "Global EDA", "Comparative Analysis", "ML Analysis", "Future Risk Predictor"], label_visibility="collapsed")
        
        if page == "About":
            show_about_page()
        elif page == "Global EDA":
            show_eda_page(df)
        elif page == "Comparative Analysis":
            show_comparative_analysis_page(df)
        elif page == "ML Analysis":
            show_ml_analysis_page(ml_assets)
        elif page == "Future Risk Predictor":
            show_predictor_page(df, ml_assets) 

if __name__ == "__main__":
    main()

