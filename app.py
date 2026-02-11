import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="Analisis & Prediksi Anggaran", layout="wide")

st.title("📊 Analisis & Prediksi Anggaran Damkar Minahasa")

# Sidebar menu
menu = st.sidebar.radio(
    "📂 Pilih Menu",
    ["📑 Data Anggaran", "📈 Pola Belanja", "🔮 Prediksi", "📊 Grafik Tren"],
)

# Upload file
uploaded_file = st.sidebar.file_uploader("Upload file CSV/Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Baca file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Normalisasi nama kolom
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_")

    # Ubah string angka ke integer
    for col in df.columns:
        if col != "Tahun":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Prediksi dengan Regresi Linier
    X = df["Tahun"].values.reshape(-1, 1)

    # Input tahun prediksi dari user
    tahun_awal = int(df["Tahun"].max()) + 1
    tahun_akhir = st.sidebar.number_input(
        "Prediksi sampai tahun:",
        min_value=tahun_awal,
        max_value=2040,
        value=2027,
        step=1,
    )
    tahun_prediksi = np.arange(tahun_awal, tahun_akhir + 1).reshape(-1, 1)

    prediksi_semua = {}
    metrics = {}

    for kolom in [
        "Belanja_Pegawai",
        "Belanja_Barang_Jasa",
        "Belanja_Modal",
        "Total_Belanja",
        "PAD",
    ]:
        y = df[kolom].values
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict(tahun_prediksi)
        prediksi_semua[kolom] = pred.astype(int)

        # Evaluasi model
        y_pred_train = model.predict(X)
        mse = mean_squared_error(y, y_pred_train)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred_train)
        metrics[kolom] = {"MSE": mse, "RMSE": rmse, "R²": r2}

    df_prediksi = pd.DataFrame(
        {
            "Tahun": tahun_prediksi.flatten(),
            "Prediksi_Belanja_Pegawai": prediksi_semua["Belanja_Pegawai"],
            "Prediksi_Belanja_Barang_Jasa": prediksi_semua["Belanja_Barang_Jasa"],
            "Prediksi_Belanja_Modal": prediksi_semua["Belanja_Modal"],
            "Prediksi_Total_Belanja": prediksi_semua["Total_Belanja"],
            "Prediksi_PAD": prediksi_semua["PAD"],
        }
    )

    # === Menu Tampilan ===
    if menu == "📑 Data Anggaran":
        st.subheader("📑 Data Anggaran (2020–2024)")
        with st.expander("Klik untuk melihat data anggaran"):
            st.dataframe(df, use_container_width=True, height=250)

    elif menu == "📈 Pola Belanja":
        st.subheader("📈 Pola Belanja Daerah")
        kategori = st.selectbox(
            "Pilih kategori belanja untuk ditampilkan:",
            [
                "Belanja_Pegawai",
                "Belanja_Barang_Jasa",
                "Belanja_Modal",
                "Total_Belanja",
                "PAD",
            ],
        )

        # Grafik isi kecil
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(
            df["Tahun"],
            df[kategori],
            marker="o",
            markersize=4,
            linewidth=1,
            label=kategori,
        )
        ax.set_title(f"Pola {kategori} (2020–2024)", fontsize=9)
        ax.set_xlabel("Tahun", fontsize=7)
        ax.set_ylabel("Rp", fontsize=7)
        ax.tick_params(axis="both", labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

        st.pyplot(fig, use_container_width=True)

    elif menu == "🔮 Prediksi":
        st.subheader(f"🔮 Hasil Prediksi {tahun_awal}–{tahun_akhir}")
        st.dataframe(df_prediksi, use_container_width=True, height=200)

        csv = df_prediksi.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Prediksi (CSV)",
            data=csv,
            file_name="prediksi_anggaran.csv",
            mime="text/csv",
        )

        # Tampilkan evaluasi model
        st.subheader("📏 Evaluasi Algoritma Regresi Linier")
        with st.expander("Klik untuk melihat hasil evaluasi"):
            df_metrics = pd.DataFrame(metrics).T.reset_index()
            df_metrics = df_metrics.rename(columns={"index": "Kategori"})
            df_metrics["MSE"] = df_metrics["MSE"].apply(lambda x: f"{x:,.0f}")
            df_metrics["RMSE"] = df_metrics["RMSE"].apply(lambda x: f"{x:,.0f}")
            df_metrics["R²"] = df_metrics["R²"].apply(lambda x: f"{x:.3f}")
            st.table(df_metrics)

    elif menu == "📊 Grafik Tren":
        st.subheader(f"📊 Tren Belanja (Aktual & Prediksi sampai {tahun_akhir})")

        kategori = st.multiselect(
            "Pilih kategori yang ingin ditampilkan:",
            [
                "Belanja_Pegawai",
                "Belanja_Barang_Jasa",
                "Belanja_Modal",
                "Total_Belanja",
                "PAD",
            ],
            default=["Total_Belanja"],
        )

        with st.expander("Klik untuk melihat grafik tren"):
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            for kolom in kategori:
                ax2.plot(
                    df["Tahun"],
                    df[kolom],
                    marker="o",
                    markersize=4,
                    linewidth=1,
                    label=f"{kolom} (Aktual)",
                )
                ax2.plot(
                    df_prediksi["Tahun"],
                    df_prediksi[f"Prediksi_{kolom}"],
                    marker="x",
                    markersize=4,
                    linestyle="--",
                    linewidth=1,
                    label=f"{kolom} (Prediksi)",
                )

            ax2.set_title("Tren Belanja Damkar", fontsize=9)
            ax2.set_xlabel("Tahun", fontsize=7)
            ax2.set_ylabel("Rp", fontsize=7)
            ax2.tick_params(axis="both", labelsize=7)
            ax2.legend(fontsize=7)
            ax2.grid(True, linestyle="--", alpha=0.4)

            st.pyplot(fig2, use_container_width=True)

else:
    st.warning("📂 Silakan upload file CSV/Excel terlebih dahulu melalui sidebar.")
