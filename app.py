import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# =========================
# CONFIG HALAMAN
# =========================
st.set_page_config(
    page_title="Analisis & Prediksi Anggaran",
    layout="wide"
)

st.title("📊 Analisis & Prediksi Anggaran Damkar Minahasa")

# =========================
# SIDEBAR MENU
# =========================
menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "📑 Data Anggaran",
        "📈 Pola Belanja",
        "🔮 Prediksi",
        "📊 Grafik Tren"
    ]
)

# =========================
# UPLOAD FILE
# =========================
uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV/Excel",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # =========================
    # BACA FILE
    # =========================
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    # =========================
    # NORMALISASI NAMA KOLOM
    # =========================
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
    )

    st.sidebar.write("Kolom terdeteksi:")
    st.sidebar.write(df.columns.tolist())

    # =========================
    # VALIDASI KOLOM TAHUN
    # =========================
    if "Tahun" not in df.columns:
        st.error("Kolom 'Tahun' tidak ditemukan pada file.")
        st.stop()

    # =========================
    # UBAH DATA KE NUMERIK
    # =========================
    for col in df.columns:
        if col != "Tahun":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", "", regex=False)
            )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    # =========================
    # KOLOM YANG AKAN DIPROSES
    # =========================
    kandidat_kolom = [
        "Belanja_Pegawai",
        "Belanja_Barang_Jasa",
        "Belanja_Modal",
        "Total_Belanja",
        "PAD"
    ]

    kolom_tersedia = [
        k for k in kandidat_kolom if k in df.columns
    ]

    if len(kolom_tersedia) == 0:
        st.error("Tidak ada kolom anggaran yang cocok.")
        st.stop()

    # =========================
    # PREDIKSI REGRESI LINIER
    # =========================
    X = df["Tahun"].values.reshape(-1, 1)

    tahun_awal = int(df["Tahun"].max()) + 1

    tahun_akhir = st.sidebar.number_input(
        "Prediksi sampai tahun:",
        min_value=tahun_awal,
        max_value=2040,
        value=tahun_awal + 2,
        step=1
    )

    tahun_prediksi = np.arange(
        tahun_awal,
        tahun_akhir + 1
    ).reshape(-1, 1)

    prediksi_semua = {}
    metrics = {}

    for kolom in kolom_tersedia:

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

        metrics[kolom] = {
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2
        }

    # =========================
    # DATAFRAME HASIL PREDIKSI
    # =========================
    df_prediksi = pd.DataFrame({
        "Tahun": tahun_prediksi.flatten()
    })

    for kolom in kolom_tersedia:
        df_prediksi[f"Prediksi_{kolom}"] = prediksi_semua[kolom]

    # =========================
    # MENU DATA ANGGARAN
    # =========================
    if menu == "📑 Data Anggaran":

        st.subheader("📑 Data Anggaran")

        with st.expander("Klik untuk melihat data"):

            st.dataframe(
                df,
                use_container_width=True,
                height=300
            )

    # =========================
    # MENU POLA BELANJA
    # =========================
    elif menu == "📈 Pola Belanja":

        st.subheader("📈 Pola Belanja Daerah")

        kategori = st.selectbox(
            "Pilih kategori:",
            kolom_tersedia
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(
            df["Tahun"],
            df[kategori],
            marker="o",
            linewidth=2,
            label=kategori
        )

        ax.set_title(f"Pola {kategori}")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Nilai")
        ax.grid(True, linestyle="--", alpha=0.5)

        ax.legend()

        st.pyplot(fig, use_container_width=True)

    # =========================
    # MENU PREDIKSI
    # =========================
    elif menu == "🔮 Prediksi":

        st.subheader(
            f"🔮 Hasil Prediksi {tahun_awal} - {tahun_akhir}"
        )

        st.dataframe(
            df_prediksi,
            use_container_width=True,
            height=300
        )

        # DOWNLOAD CSV
        csv = df_prediksi.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Prediksi CSV",
            data=csv,
            file_name="prediksi_anggaran.csv",
            mime="text/csv"
        )

        # EVALUASI MODEL
        st.subheader("📏 Evaluasi Model")

        with st.expander("Klik untuk melihat evaluasi"):

            df_metrics = (
                pd.DataFrame(metrics)
                .T
                .reset_index()
            )

            df_metrics = df_metrics.rename(
                columns={"index": "Kategori"}
            )

            df_metrics["MSE"] = df_metrics["MSE"].apply(
                lambda x: f"{x:,.0f}"
            )

            df_metrics["RMSE"] = df_metrics["RMSE"].apply(
                lambda x: f"{x:,.0f}"
            )

            df_metrics["R²"] = df_metrics["R²"].apply(
                lambda x: f"{x:.3f}"
            )

            st.table(df_metrics)

    # =========================
    # MENU GRAFIK TREN
    # =========================
    elif menu == "📊 Grafik Tren":

        st.subheader(
            f"📊 Grafik Tren sampai {tahun_akhir}"
        )

        kategori = st.multiselect(
            "Pilih kategori:",
            kolom_tersedia,
            default=[kolom_tersedia[0]]
        )

        fig2, ax2 = plt.subplots(figsize=(8, 4))

        for kolom in kategori:

            # Data aktual
            ax2.plot(
                df["Tahun"],
                df[kolom],
                marker="o",
                linewidth=2,
                label=f"{kolom} Aktual"
            )

            # Data prediksi
            ax2.plot(
                df_prediksi["Tahun"],
                df_prediksi[f"Prediksi_{kolom}"],
                marker="x",
                linestyle="--",
                linewidth=2,
                label=f"{kolom} Prediksi"
            )

        ax2.set_title("Grafik Tren Anggaran")
        ax2.set_xlabel("Tahun")
        ax2.set_ylabel("Nilai")
        ax2.grid(True, linestyle="--", alpha=0.5)

        ax2.legend()

        st.pyplot(fig2, use_container_width=True)

# =========================
# JIKA BELUM UPLOAD FILE
# =========================
else:
    st.warning(
        "📂 Silakan upload file CSV/Excel terlebih dahulu."
    )