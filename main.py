import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import stft, detrend, butter, filtfilt
from scipy.interpolate import interp1d


st.set_page_config(page_title="Espectrograma por Canal", layout="wide")


def detectar_delimitador(file_bytes: bytes) -> str | None:
    amostra = file_bytes[:5000].decode("utf-8", errors="ignore")
    candidatos = [",", ";", "\t"]
    contagens = {c: amostra.count(c) for c in candidatos}
    melhor = max(contagens, key=contagens.get)
    return melhor if contagens[melhor] > 0 else None


def carregar_dados(uploaded_file) -> pd.DataFrame:
    file_bytes = uploaded_file.getvalue()
    sep = detectar_delimitador(file_bytes)

    tentativas = []
    if sep is not None:
        tentativas.append({"sep": sep})
    tentativas += [
        {"sep": None, "delim_whitespace": True},
        {"sep": ","},
        {"sep": ";"},
        {"sep": "\t"},
    ]

    ultimo_erro = None
    for kwargs in tentativas:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), engine="python", **kwargs)
            if df.shape[1] >= 2:
                df.columns = [str(c).strip() for c in df.columns]
                return df
        except Exception as e:
            ultimo_erro = e

    raise ValueError(f"Não foi possível ler o arquivo. Erro: {ultimo_erro}")


def encontrar_coluna_tempo(df: pd.DataFrame) -> str | None:
    nomes = [c.lower().strip() for c in df.columns]
    prioridades = ["tempo", "time", "timestamp", "t"]
    for p in prioridades:
        for col, nome in zip(df.columns, nomes):
            if nome == p:
                return col
    for col, nome in zip(df.columns, nomes):
        if "tempo" in nome or "time" in nome:
            return col
    return None


def estimar_fs(tempo: np.ndarray, unidade_tempo: str) -> float:
    dt = np.diff(tempo)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        raise ValueError("Não foi possível estimar a frequência de amostragem.")
    dt_mediano = np.median(dt)

    if unidade_tempo == "ms":
        return 1000.0 / dt_mediano
    if unidade_tempo == "us":
        return 1_000_000.0 / dt_mediano
    return 1.0 / dt_mediano


def converter_tempo_para_segundos(tempo: np.ndarray, unidade_tempo: str) -> np.ndarray:
    if unidade_tempo == "ms":
        return tempo / 1000.0
    if unidade_tempo == "us":
        return tempo / 1_000_000.0
    return tempo


def interpolar_para_fs_fixa(t: np.ndarray, y: np.ndarray, fs_novo: float = 100.0):
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]

    if len(t) < 4:
        raise ValueError("Sinal com poucas amostras válidas para interpolação.")

    ordem = np.argsort(t)
    t = t[ordem]
    y = y[ordem]

    t_unico, idx = np.unique(t, return_index=True)
    y_unico = y[idx]

    if len(t_unico) < 4:
        raise ValueError("Tempo com poucas amostras únicas para interpolação.")

    t_uniforme = np.arange(t_unico[0], t_unico[-1], 1.0 / fs_novo)
    if len(t_uniforme) < 4:
        raise ValueError("Janela temporal insuficiente para interpolação.")

    f_interp = interp1d(t_unico, y_unico, kind="linear", bounds_error=False, fill_value="extrapolate")
    y_interp = f_interp(t_uniforme)

    return t_uniforme, y_interp


def aplicar_filtro_pb(y: np.ndarray, fs: float, fc: float = 5.0, ordem: int = 4):
    nyq = fs / 2.0
    if fc >= nyq:
        raise ValueError("A frequência de corte deve ser menor que a frequência de Nyquist.")
    b, a = butter(ordem, fc / nyq, btype="low")
    return filtfilt(b, a, y)


def preprocessar_sinal(t, y, fazer_detrend=True, fazer_interpolacao=True, fs_interp=100.0, fazer_filtro=True, fc=5.0):
    t_proc = np.asarray(t, dtype=float)
    y_proc = np.asarray(y, dtype=float)

    if fazer_interpolacao:
        t_proc, y_proc = interpolar_para_fs_fixa(t_proc, y_proc, fs_novo=fs_interp)
        fs_proc = fs_interp
    else:
        mask = np.isfinite(t_proc) & np.isfinite(y_proc)
        t_proc = t_proc[mask]
        y_proc = y_proc[mask]
        dt = np.diff(t_proc)
        dt = dt[dt > 0]
        if len(dt) == 0:
            raise ValueError("Não foi possível estimar a frequência após remoção de NaNs.")
        fs_proc = 1.0 / np.median(dt)

    if fazer_detrend:
        y_proc = detrend(y_proc, type="linear")

    if fazer_filtro:
        y_proc = aplicar_filtro_pb(y_proc, fs_proc, fc=fc, ordem=4)

    return t_proc, y_proc, fs_proc


def figura_registro(t, y, canal, titulo_extra=""):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, y)
    ax.set_title(f"Registro temporal - canal {canal}{titulo_extra}")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def figura_espectrograma(y, fs, canal, fmax, nperseg, noverlap, titulo_extra=""):
    f, tt, zxx = stft(
        y,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        boundary=None,
        padded=False,
    )

    mag_db = 20 * np.log10(np.abs(zxx) + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    pcm = ax.pcolormesh(tt, f, mag_db, shading="gouraud")
    ax.set_title(f"Espectrograma STFT - canal {canal}{titulo_extra}")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Frequência (Hz)")
    ax.set_ylim(0, fmax)
    fig.colorbar(pcm, ax=ax, label="Magnitude (dB)")
    fig.tight_layout()
    return fig


st.title("Visualização do registro e espectrograma por canal")

uploaded_file = st.file_uploader(
    "Carregue um arquivo TXT ou CSV contendo a coluna de tempo e os canais",
    type=["txt", "csv"],
)

if uploaded_file is not None:
    try:
        df = carregar_dados(uploaded_file)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.subheader("Pré-visualização dos dados")
    st.dataframe(df.head(10), use_container_width=True)

    col_tempo = encontrar_coluna_tempo(df)
    if col_tempo is None:
        st.error("Não encontrei automaticamente a coluna de tempo.")
        st.stop()

    st.success(f"Coluna de tempo detectada: {col_tempo}")

    colunas_numericas = []
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].notna().sum() > 0:
                colunas_numericas.append(col)
        except Exception:
            pass

    canais = [c for c in colunas_numericas if c != col_tempo]
    if len(canais) == 0:
        st.error("Nenhum canal numérico foi encontrado além da coluna de tempo.")
        st.stop()

    with st.sidebar:
        st.header("Configurações")

        unidade_tempo = st.selectbox(
            "Unidade da coluna de tempo",
            ["ms", "s", "us"],
            index=0 if col_tempo.lower().strip() == "tempo" else 1,
        )

        tempo = df[col_tempo].to_numpy(dtype=float)
        tempo_s = converter_tempo_para_segundos(tempo, unidade_tempo)

        try:
            fs_auto = estimar_fs(tempo, unidade_tempo)
        except Exception:
            fs_auto = np.nan

        st.markdown("### Pré-processamento")
        fazer_detrend = st.checkbox("Aplicar detrend linear", value=True)
        fazer_interpolacao = st.checkbox("Interpolar para frequência fixa", value=True)
        fs_interp = st.number_input("Frequência após interpolação (Hz)", min_value=1.0, value=100.0, step=1.0)

        fazer_filtro = st.checkbox("Aplicar filtro passa-baixa", value=True)
        fc = st.number_input("Frequência de corte do filtro (Hz)", min_value=0.1, value=5.0, step=0.1)

        st.markdown("---")
        canais_escolhidos = st.multiselect(
            "Canais para exibir",
            options=canais,
            default=canais,
        )

        fs_exibicao = fs_interp if fazer_interpolacao else (float(fs_auto) if np.isfinite(fs_auto) else 100.0)
        fmax_lim = max(1.0, fs_exibicao / 2.0)
        fmax_default = min(5.0, fmax_lim)

        fmax = st.slider(
            "Frequência máxima do espectrograma (Hz)",
            min_value=1.0,
            max_value=float(fmax_lim),
            value=float(fmax_default),
            step=0.5,
        )

        nperseg = st.slider("Tamanho da janela (nperseg)", min_value=32, max_value=1024, value=256, step=32)
        noverlap = st.slider("Sobreposição (noverlap)", min_value=0, max_value=nperseg - 1, value=min(192, nperseg - 1), step=1)

    st.subheader("Resumo do sinal original")
    c1, c2, c3 = st.columns(3)
    c1.metric("Número de amostras", f"{len(df)}")
    c2.metric("Duração aproximada (s)", f"{tempo_s[-1] - tempo_s[0]:.2f}")
    if np.isfinite(fs_auto):
        c3.metric("Frequência estimada original (Hz)", f"{fs_auto:.2f}")
    else:
        c3.metric("Frequência estimada original (Hz)", "indisponível")

    if not canais_escolhidos:
        st.warning("Selecione ao menos um canal.")
        st.stop()

    for canal in canais_escolhidos:
        st.markdown(f"## Canal {canal}")
        y = df[canal].to_numpy(dtype=float)

        mask = np.isfinite(tempo_s) & np.isfinite(y)
        t_bruto = tempo_s[mask]
        y_bruto = y[mask]

        if len(y_bruto) < 4:
            st.warning(f"O canal {canal} não tem amostras suficientes.")
            continue

        try:
            t_proc, y_proc, fs_proc = preprocessar_sinal(
                t_bruto,
                y_bruto,
                fazer_detrend=fazer_detrend,
                fazer_interpolacao=fazer_interpolacao,
                fs_interp=float(fs_interp),
                fazer_filtro=fazer_filtro,
                fc=float(fc),
            )
        except Exception as e:
            st.error(f"Erro ao processar o canal {canal}: {e}")
            continue

        c4, c5 = st.columns(2)
        c4.metric("Amostras processadas", f"{len(y_proc)}")
        c5.metric("Fs usada na STFT (Hz)", f"{fs_proc:.2f}")

        with st.expander(f"Ver registro bruto e processado - canal {canal}", expanded=True):
            fig1 = figura_registro(t_bruto, y_bruto, canal, titulo_extra=" (bruto)")
            st.pyplot(fig1, use_container_width=True)

            fig2 = figura_registro(t_proc, y_proc, canal, titulo_extra=" (processado)")
            st.pyplot(fig2, use_container_width=True)

        fig3 = figura_espectrograma(
            y_proc,
            fs_proc,
            canal,
            fmax,
            nperseg,
            noverlap,
            titulo_extra=" (processado)",
        )
        st.pyplot(fig3, use_container_width=True)

else:
    st.info("Envie um arquivo para começar.")
