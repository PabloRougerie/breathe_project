# imports
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
import os
from src.params import *



# ===============
# PAGE CONFIG
# ================
st.set_page_config(page_title= "PM2.5 level prediction: model performance monitoring",
                   page_icon= "📈",
                layout= "wide")

with st.sidebar:
    if st.button("Purger le cache BQ et recharger"):
        st.cache_data.clear()
        st.rerun()

#================
# LOAD DATA
#================

@st.cache_data
def load_data_from_bq(table_name):
    client = bigquery.Client()
    full_table_name = f"{GCP_PROJECT}.{BQ_DATASET_MONITORING}.{table_name}"

    #get batch data
    query = f"""
            SELECT *
            FROM `{full_table_name}`

        """

    df = client.query(query).result().to_dataframe()
    return df
    #df_batch["batch_start"] = pd.to_datetime(df_batch["batch_start"])
    #df_batch["batch_end"] = pd.to_datetime(df_batch["batch_end"])
    #get prediction data
    #NOTE: in full scale project we'd filter on request, but here we can getthe whole table and filter later
#     query_pred = f"""
#             SELECT *
#             FROM `{table_name_prediction}`

#         """
#     df_pred = client.query(query_pred).result().to_dataframe()
#     df_pred["date"] = pd.to_datetime(df_pred["date"])
#     return df_batch, df_pred




tab1, tab2 = st.tabs(["Drift Monitoring", "PM2.5 prediction"])

with tab1:
    df_batch = load_data_from_bq("batches")
    df_batch["batch_start"] = pd.to_datetime(df_batch["batch_start"])
    df_batch["batch_end"] = pd.to_datetime(df_batch["batch_end"])
    df_batch = df_batch.sort_values(by=["batch_start"], ascending=False)
    df_batch["drift_detected"] = df_batch["drift_detected"].fillna(False).astype(bool)
    df_batch["promotion_applied"] = df_batch["promotion_applied"].fillna(False).astype(bool)



    for _, batch in df_batch.iterrows():
        drift = bool(batch.get("drift_detected", False))
        promoted = bool(batch.get("promotion_applied", False))
        champion_rmse = batch.get("champion_rmse")
        champion_version = batch.get("champion_version")
        challenger_rmse = batch.get("rmse_challenger")

        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.2, 1.2, 1.5])

            with c1:
                st.markdown("**Batch Start**")
                st.caption(batch["batch_start"].strftime("%Y-%m-%d"))

            with c2:
                st.markdown("**Batch End**")
                st.caption(batch["batch_end"].strftime("%Y-%m-%d"))

            with c3:
                if pd.notna(champion_rmse):
                    champ_delta = f"v{champion_version}" if pd.notna(champion_version) else None
                    st.metric(
                        "Current model RMSE",
                        f"{champion_rmse:.3f}",
                        delta=champ_delta,
                        delta_color="off",
                    )
                else:
                    st.metric("Current model RMSE", "—")

            with c4:
                # Drift YES = alerte rouge ; NO = neutre (pas "succès" vert, ça embrouille avec Promoted).
                # Promoted YES = vert ; NO = neutre (même gris que Drift NO).
                gray = "#7f8c8d"
                drift_html = (
                    '<span style="color:#c0392b;font-weight:600">YES</span>'
                    if drift
                    else f'<span style="color:{gray};font-weight:600">NO</span>'
                )
                prom_html = (
                    '<span style="color:#27ae60;font-weight:600">YES</span>'
                    if promoted
                    else f'<span style="color:{gray};font-weight:600">NO</span>'
                )
                st.markdown("**Drift**", unsafe_allow_html=False)
                st.markdown(drift_html, unsafe_allow_html=True)
                st.markdown("**Promoted**", unsafe_allow_html=False)
                st.markdown(prom_html, unsafe_allow_html=True)

            with c5:
                if pd.notna(challenger_rmse):
                    pct_delta = None
                    if pd.notna(champion_rmse) and float(champion_rmse) != 0:
                        pct = 100 * (1 - float(challenger_rmse) / float(champion_rmse))
                        pct_delta = f"{pct:+.1f}% vs prod"
                    st.metric(
                        "New model RMSE",
                        f"{challenger_rmse:.3f}",
                        delta=pct_delta,
                        delta_color="normal",
                    )
                else:
                    st.metric("New model RMSE", "—")


with tab2:
    df_preds = load_data_from_bq("predictions")

    cities = df_preds["city"].unique().tolist()
    choice = st.selectbox("Choose a city:", cities)

    date_choice= st.slider("date range", min_value= df_preds["date"].min(),
              max_value= df_preds["date"].max(),
              value= (df_preds["date"].min(),df_preds["date"].max()))


    selected_data = df_preds[
        (df_preds["city"] == choice)
        & (df_preds["date"] >= date_choice[0])
        & (df_preds["date"] <= date_choice[1])
    ].sort_values("date")
    st.line_chart(data=selected_data, x="date", y=["y_true", "y_pred"])

















#     # Try multiple possible paths for different deployment scenarios
#     possible_paths = [
#         Path(__file__).parent / 'source_data' / 'predictions.json',  # Relative to app.py
#         Path('UI/source_data/predictions.json'),  # From repo root
#         Path('source_data/predictions.json'),  # From UI directory
#     ]

#     for json_path in possible_paths:
#         if json_path.exists():
#             with open(json_path, 'r') as f:
#                 return json.load(f)

#     # If none found, raise error with helpful message
#     raise FileNotFoundError(
#         f"Could not find predictions.json. Tried: {[str(p) for p in possible_paths]}"
#     )

# predictions = load_predictions()


# #================
# # SIDE BAR
# #================

# st.title("Vessel Trajectory Prediction")
# st.markdown("""
# **Predicting vessel positions**
# *(for instance for Search & Rescue operations)*
# Compare baseline extrapolation vs. machine learning and deep learning models (LightGBM, LSTM).
# """)

# st.sidebar.title("Parameters")

# st.sidebar.subheader("Vessel selection", divider= "red")
# vessel_ID = list(predictions.keys())
# select_vessel = st.sidebar.selectbox("Select a vessel (MMSI identifier):", vessel_ID)

# st.sidebar.subheader("Prediction time selection", divider= "red")
# horizon = st.sidebar.pills("Select the localization forecast time horizon:",
#                            ["1h", "6h", "12h", "24h"],
#                            selection_mode= "single", default= "24h")

# st.sidebar.subheader("Model selection", divider= "red")
# baseline_on = st.sidebar.toggle("Baseline", value= True)
# lgbm_on = st.sidebar.toggle("LightGBM model")
# lstm_on = st.sidebar.toggle("LSTM model")

# st.sidebar.subheader("Additional information", divider= "red")
# true_pos = st.sidebar.toggle("Show true position?")
# error = st.sidebar.toggle("Show prediction error?")




# #================
# # MAIN FIELD
# #================

# #extract data for the right vessel and right time horizon
# vessel_data = predictions[select_vessel]
# horizon_data = vessel_data["predictions"][horizon]


# #MAP
# col1, col2 = st.columns([0.6,0.3])

# with col1:
#     st.header("Trajectory & Prediction")
#     map = folium.Map(location= [24.5278, -84], zoom_start= 5.49)

#     # past trajectory
#     folium.PolyLine(
#         vessel_data["history"]["positions"],
#         color= "black",
#         weight= 3,
#         popup= "past_trajectory"
#     ).add_to(map)

#     #starting point
#     folium.CircleMarker(
#         vessel_data["history"]["positions"][0],
#         radius = 5,
#         color= "grey",
#         fill = True,
#         fill_opacity = 1
#     ).add_to(map)

#     #last_know_point
#     folium.CircleMarker(
#         vessel_data["pred_point"]["pred_position"],
#         radius = 4,
#         color= "blue",
#         fill = True,
#         fill_opacity = 1
#     ).add_to(map)

#     #predictions
#     if baseline_on:
#         folium.CircleMarker(
#         horizon_data["y_pred_baseline"]["position"],
#         radius = 4,
#         color= "purple",
#         fill = True,
#         fill_opacity = 1
#     ).add_to(map)
#         if error:
#             folium.Circle(
#                 location= horizon_data["y_pred_baseline"]["position"],
#                 radius = horizon_data["y_pred_baseline"]["error"]*1000,
#                 color = "purple",
#                 fill = True,
#                 fill_opacity = 0.2
#             ).add_to(map)



#     if lgbm_on:
#         folium.CircleMarker(
#         horizon_data["y_pred_lgbm"]["position"],
#         radius = 4,
#         color= "orange",
#         fill = True,
#         fill_opacity = 1
#     ).add_to(map)
#         if error:
#             folium.Circle(
#                 location= horizon_data["y_pred_lgbm"]["position"],
#                 radius = horizon_data["y_pred_lgbm"]["error"]*1000,
#                 color = "orange",
#                 fill = True,
#                 fill_opacity = 0.2
#             ).add_to(map)

#     if lstm_on:
#         folium.CircleMarker(
#         horizon_data["y_pred_lstm"]["position"],
#         radius = 4,
#         color= "red",
#         fill = True,
#         fill_opacity = 1
#     ).add_to(map)
#         if error:
#             folium.Circle(
#                 location= horizon_data["y_pred_lstm"]["position"],
#                 radius = horizon_data["y_pred_lstm"]["error"]*1000,
#                 color = "red",
#                 fill = True,
#                 fill_opacity = 0.2
#             ).add_to(map)

#     #true position
#     if true_pos:
#         folium.CircleMarker(
#             horizon_data["y_true"],
#             radius = 4,
#             color= "green",
#             fill = True,
#             fill_opacity = 1
#         ).add_to(map)

#     folium_static(map, width=600, height=500)
#     st.caption("⚫ Start | 🔵 Last known | 🟢 True position | 🟣 Baseline | 🔴 LSTM | 🟠 LightGBM")

# with col2:

#     st.header("Performance metrics")
#     st.subheader("Prediction error:")

#     st.metric(label= "🟣 Baseline",
#                value = f"{horizon_data["y_pred_baseline"]["error"]:.0f}km")

#     st.metric(label= "🟠 LightGBM",
#               value = f"{horizon_data["y_pred_lgbm"]["error"]:.0f}km",
#               delta= f"{horizon_data["y_pred_baseline"]["error"] - horizon_data["y_pred_lgbm"]["error"]:.0f}")

#     st.metric(label= "🔴 LSTM",
#               value = f"{horizon_data["y_pred_lstm"]["error"]:.0f}km",
#               delta= f"{horizon_data["y_pred_baseline"]["error"] - horizon_data["y_pred_lstm"]["error"]:.0f}")
