import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# ==================================================
# CONFIG MAPBOX / ÍCONE
# ==================================================
MAPBOX_TOKEN = "pk.eyJ1IjoiZnBhc3NvczEwIiwiYSI6ImNtaWZ3ZDMyMjAwbmszZW4ybnI5dGlja2oifQ.X1kZolPs5cQsH1QCHXH9Dw"
pdk.settings.mapbox_api_key = MAPBOX_TOKEN

ICON_URL = "https://raw.githubusercontent.com/fabianopassos50-png/micropower-icons/main/micropower_icon_transparent_128.png"


# ==================================================
# CONFIGURAÇÃO DA PÁGINA
# ==================================================
st.set_page_config(
    page_title="Micropower | SIGFI e Microredes",
    layout="wide"
)
st.title("Visão geral de projetos Micropower")
st.caption(
    "Painel integrado de instalações SIGFI e microredes, com visão "
    "geográfica e consolidação de capacidade instalada."
)


# ==================================================
# FUNÇÃO: CARREGAR E TRATAR DADOS
# ==================================================
@st.cache_data
def carregar_dados(arquivo):
    xls = pd.ExcelFile(arquivo)
    sheet_names = xls.sheet_names
    dfs = []

    for sheet in sheet_names:
        df_all = pd.read_excel(arquivo, sheet_name=sheet, header=None, engine="openpyxl")

        # Acha linha de cabeçalho: precisa ter LAT e LON
        header_row = None
        for idx, row in df_all.iterrows():
            norm_vals = ["".join(str(v).upper().split()) for v in row.tolist()]
            has_lat = any("LAT" in v for v in norm_vals)
            has_lon = any("LON" in v for v in norm_vals)
            if has_lat and has_lon:
                header_row = idx
                break

        if header_row is None:
            continue

        header = df_all.iloc[header_row]
        df_sheet = df_all.iloc[header_row + 1:].copy()
        df_sheet.columns = header
        df_sheet["ORIGEM"] = sheet  # nome da aba
        dfs.append(df_sheet)

    if not dfs:
        raise ValueError("Nenhuma aba com cabeçalho contendo LAT / LON foi encontrada na planilha.")

    df = pd.concat(dfs, ignore_index=True)

    # Normaliza UF
    if "UF" in df.columns:
        df["UF"] = df["UF"].astype(str).str.strip()

    # Padroniza nomes de colunas opcionais
    if "MUNICIPIO" in df.columns and "MUNICÍPIO" not in df.columns:
        df["MUNICÍPIO"] = df["MUNICIPIO"]
    if "REGIAO" in df.columns and "REGIÃO" not in df.columns:
        df["REGIÃO"] = df["REGIAO"]

    # Unificar NOME / PROJETO em "NOME"
    if "NOME" not in df.columns and "PROJETO" in df.columns:
        df["NOME"] = df["PROJETO"]
    elif "NOME" in df.columns and "PROJETO" in df.columns:
        df["NOME"] = df["NOME"].fillna(df["PROJETO"])
    elif "NOME" not in df.columns:
        df["NOME"] = ""

    # ---------------- LAT / LON ----------------
    col_lat = None
    col_lon = None
    for c in df.columns:
        norm_c = "".join(str(c).upper().split())
        if col_lat is None and "LAT" in norm_c:
            col_lat = c
        if col_lon is None and "LON" in norm_c:
            col_lon = c

    if col_lat is None or col_lon is None:
        raise ValueError("Não foi possível identificar colunas de latitude/longitude.")

    def limpa_coord(s):
        s = (
            s.astype(str)
            .str.replace("_x000D_", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.extract(r"(-?\d+\.?\d*)", expand=False)
        )
        return pd.to_numeric(s, errors="coerce")

    df["lat"] = limpa_coord(df[col_lat])
    df["lon"] = limpa_coord(df[col_lon])
    df = df.dropna(subset=["lat", "lon"])
    df = df[df["lat"].between(-35, 10) & df["lon"].between(-80, -30)]

    # ---------------- DATAS ----------------
    df["DATA_DA_INSTALACAO"] = pd.NaT
    if "DATA DA INSTALAÇÃO" in df.columns:
        df["DATA_DA_INSTALACAO"] = pd.to_datetime(df["DATA DA INSTALAÇÃO"], errors="coerce")
    if "DATA ENTRADA EM OPERAÇÃO" in df.columns:
        mask = df["DATA_DA_INSTALACAO"].isna()
        df.loc[mask, "DATA_DA_INSTALACAO"] = pd.to_datetime(
            df.loc[mask, "DATA ENTRADA EM OPERAÇÃO"], errors="coerce"
        )

    # ---------------- ENERGIA SIGFI (baterias) ----------------
    df["SIGFI"] = pd.NA
    df["SIGFI_TIPO"] = pd.NA
    df["SIGFI_NUM"] = pd.NA
    df["ENERGIA_KWH"] = np.nan  # energia dos SIGFI

    sigfi_col = None
    for c in df.columns:
        if "SIGFI" in str(c).upper():
            sigfi_col = c
            break

    if sigfi_col is not None:
        raw = df[sigfi_col].astype(str)
        sigfi_num = raw.str.extract(r"(\d+)", expand=False)
        sigfi_num = pd.to_numeric(sigfi_num, errors="coerce")
        df["SIGFI_NUM"] = sigfi_num

        df["SIGFI_TIPO"] = np.where(
            sigfi_num.notna(),
            "SIGFI" + sigfi_num.round().astype("Int64").astype(str),
            pd.NA,
        )
        df["SIGFI"] = df["SIGFI_TIPO"].fillna(raw)

        def calc_energia(num):
            if pd.isna(num):
                return np.nan
            n = float(num)
            if n == 45:
                return 50 * 48 / 1000
            if n == 80:
                return 100 * 48 / 1000
            if n == 160:
                return 2 * 100 * 48 / 1000
            return np.nan

        df["ENERGIA_KWH"] = df["SIGFI_NUM"].round().apply(calc_energia)

    # ---------------- ENERGIA MICROREDES ----------------
    pot_col = None
    for c in df.columns:
        nome = str(c).upper()
        if "POTENCIA" in nome and "KWH" in nome:
            pot_col = c
            break

    if pot_col is not None:
        df["POTENCIA_NOMINAL_KWH"] = pd.to_numeric(
            df[pot_col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
    else:
        df["POTENCIA_NOMINAL_KWH"] = np.nan

    # Energia total
    df["ENERGIA_TOTAL_KWH"] = df[["ENERGIA_KWH", "POTENCIA_NOMINAL_KWH"]].sum(
        axis=1, skipna=True
    )

    return df


# ==================================================
# UPLOAD E BARRA LATERAL
# ==================================================
with st.sidebar:
    st.subheader("Configuração da base")
    arquivo = st.file_uploader("Selecione a planilha (.xlsx)", type=["xlsx"])

if not arquivo:
    st.info("Carregue a planilha consolidada de projetos para visualizar o painel.")
    st.stop()

df = carregar_dados(arquivo)

# --------------------------------------------------
# FILTROS
# --------------------------------------------------
with st.sidebar:
    st.subheader("Filtros")

    with st.expander("Origem (aba da planilha)", True):
        projetos = sorted(df["ORIGEM"].dropna().unique()) if "ORIGEM" in df.columns else []
        proj_sel = st.multiselect("Origem", projetos, default=projetos) if projetos else None

    with st.expander("Localização", True):
        ufs = sorted(df["UF"].dropna().unique()) if "UF" in df.columns else []
        ufs_sel = st.multiselect("UF", ufs, default=ufs) if ufs else None

    with st.expander("Categoria de projeto", False):
        cats = sorted(df["CATEGORIA"].dropna().unique()) if "CATEGORIA" in df.columns else []
        cats_sel = st.multiselect("Categoria", cats, default=cats) if cats else None

    with st.expander("Período de entrada em operação", False):
        data_range = None
        if "DATA_DA_INSTALACAO" in df.columns and df["DATA_DA_INSTALACAO"].notna().any():
            min_data = df["DATA_DA_INSTALACAO"].dropna().min().date()
            max_data = df["DATA_DA_INSTALACAO"].dropna().max().date()
            data_range = st.date_input("Período", [min_data, max_data])

    st.subheader("Aparência do mapa")
    modo_mapa = st.selectbox(
        "Modo de visualização",
        ["Ícones Micropower", "Mapa de calor", "Clusters 3D (hexágonos)"],
    )
    map_style_nome = st.selectbox(
        "Estilo do mapa base",
        ["Satélite", "Profissional (Ruas)", "Claro", "Escuro"],
        index=0,
    )
    MAP_STYLES = {
        "Profissional (Ruas)": "mapbox://styles/mapbox/streets-v12",
        "Claro": "mapbox://styles/mapbox/light-v11",
        "Escuro": "mapbox://styles/mapbox/dark-v11",
        "Satélite": "mapbox://styles/mapbox/satellite-streets-v12",
    }
    map_style = MAP_STYLES[map_style_nome]

    tamanho_icone = (
        st.slider("Tamanho do ícone", 1, 10, 2, step=1)
        if modo_mapa == "Ícones Micropower"
        else 2
    )
    raio_hex = (
        st.slider("Raio dos hexágonos (m)", 5000, 50000, 15000, step=5000)
        if modo_mapa == "Clusters 3D (hexágonos)"
        else 15000
    )
    heat_radius = (
        st.slider("Raio do mapa de calor (pixels)", 10, 100, 40, step=5)
        if modo_mapa == "Mapa de calor"
        else 40
    )
    zoom_inicial = st.slider("Zoom inicial", 3, 12, 7)
    pitch_inicial = st.slider("Inclinação 3D (pitch)", 0, 60, 29)
    bearing_inicial = 0
    camada_categoria = st.checkbox("Diferenciar categorias no mapa", value=True)


# --------------------------------------------------
# APLICA FILTROS
# --------------------------------------------------
df_f = df.copy()

if proj_sel is not None:
    df_f = df_f[df_f["ORIGEM"].isin(proj_sel)]
if ufs_sel is not None:
    df_f = df_f[df_f["UF"].isin(ufs_sel)]
if cats_sel is not None:
    df_f = df_f[df_f["CATEGORIA"].isin(cats_sel)]

if (
    "DATA_DA_INSTALACAO" in df_f.columns
    and data_range
    and isinstance(data_range, (list, tuple))
    and len(data_range) == 2
):
    inicio, fim = data_range
    inicio = pd.to_datetime(inicio)
    fim = pd.to_datetime(fim)
    mask_intervalo = df_f["DATA_DA_INSTALACAO"].between(inicio, fim)
    mask_sem_data = df_f["DATA_DA_INSTALACAO"].isna()
    df_f = df_f[mask_intervalo | mask_sem_data]


# ==================================================
# CÁLCULO DOS INDICADORES
# ==================================================
unidades_operacao = len(df_f)
estados_presentes = df_f["UF"].nunique() if "UF" in df_f.columns else 0

# Microredes (com potência nominal)
is_micro = df_f["ORIGEM"].str.upper().str.contains("MICROREDE", na=False)
df_mr = df_f[is_micro & df_f["POTENCIA_NOMINAL_KWH"].notna()].copy()
energia_micro_kwh = df_mr["POTENCIA_NOMINAL_KWH"].fillna(0).sum()

# SIGFI (com SIGFI_TIPO)
is_sigfi = df_f["SIGFI_TIPO"].notna()
energia_sigfi_kwh = df_f.loc[is_sigfi, "ENERGIA_KWH"].fillna(0).sum()

energia_total_kwh = energia_sigfi_kwh + energia_micro_kwh
energia_sigfi_mwh = energia_sigfi_kwh / 1000.0
energia_micro_mwh = energia_micro_kwh / 1000.0
energia_total_mwh = energia_total_kwh / 1000.0

# Microredes por categoria
micro_categoria_df = None
num_categorias_micro = 0
total_projetos_micro = len(df_mr)

if not df_mr.empty and "CATEGORIA" in df_mr.columns:
    micro_categoria_df = (
        df_mr.groupby("CATEGORIA")["POTENCIA_NOMINAL_KWH"]
        .sum()
        .reset_index()
        .sort_values("POTENCIA_NOMINAL_KWH", ascending=False)
    )
    micro_categoria_df["Energia_MWh"] = (
        micro_categoria_df["POTENCIA_NOMINAL_KWH"] / 1000.0
    ).round(2)
    micro_categoria_df["Energia_MWh_fmt"] = micro_categoria_df["Energia_MWh"].map(
        lambda v: f"{v:,.2f} MWh".replace(",", ".")
    )
    micro_categoria_df = micro_categoria_df[["CATEGORIA", "Energia_MWh", "Energia_MWh_fmt"]]
    num_categorias_micro = micro_categoria_df["CATEGORIA"].nunique()


# ==================================================
# FUNÇÕES DE MÉTRICAS E MAPA
# ==================================================
def render_metricas(layout="wide"):
    if layout == "wide":
        col1, col2, col3, col4 = st.columns(4)
    else:
        col1 = col2 = col3 = col4 = st

    col1.metric("Unidades em operação (filtro)", unidades_operacao)
    col2.metric("Capacidade instalada total", f"{energia_total_mwh:,.2f} MWh")
    col3.metric("Capacidade instalada SIGFI (baterias)", f"{energia_sigfi_mwh:,.2f} MWh")
    col4.metric("Capacidade instalada Microredes (nominal)", f"{energia_micro_mwh:,.2f} MWh")


def render_metricas_exec_lado():
    st.metric("Unidades em operação", unidades_operacao)
    st.metric("Estados atendidos", estados_presentes)
    st.metric("Capacidade instalada total", f"{energia_total_mwh:,.2f} MWh")
    st.metric("Capacidade instalada SIGFI (baterias)", f"{energia_sigfi_mwh:,.2f} MWh")
    st.metric("Capacidade instalada Microredes (nominal)", f"{energia_micro_mwh:,.2f} MWh")
    st.metric("Projetos de Microrede", total_projetos_micro)
    st.metric("Categorias de Microrede", num_categorias_micro)


def desenhar_mapa(df_dados, height=500):
    if df_dados.empty:
        st.warning("Nenhum ponto encontrado com os filtros atuais.")
        return

    df_mapa = df_dados.copy()
    df_mapa["icon_data"] = df_mapa.apply(
        lambda row: {"url": ICON_URL, "width": 128, "height": 128, "anchorY": 128},
        axis=1,
    )

    # Cores por categoria
    if camada_categoria and "CATEGORIA" in df_mapa.columns:
        palette = [
            (255, 99, 132),
            (54, 162, 235),
            (255, 206, 86),
            (75, 192, 192),
            (153, 102, 255),
            (255, 159, 64),
            (0, 201, 167),
            (160, 160, 160),
        ]
        categorias = df_mapa["CATEGORIA"].fillna("Outros").astype(str)
        unique_cats = list(categorias.unique())
        color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(unique_cats)}
        rgb = categorias.map(color_map)
        df_mapa["cat_r"] = rgb.apply(lambda c: int(c[0]))
        df_mapa["cat_g"] = rgb.apply(lambda c: int(c[1]))
        df_mapa["cat_b"] = rgb.apply(lambda c: int(c[2]))
    else:
        df_mapa["cat_r"] = 200
        df_mapa["cat_g"] = 30
        df_mapa["cat_b"] = 0

    cols_keep = [
        "lat",
        "lon",
        "NOME",
        "MUNICÍPIO",
        "UF",
        "ENDEREÇO",
        "CATEGORIA",
        "SIGFI",
        "SIGFI_TIPO",
        "ENERGIA_TOTAL_KWH",
        "ORIGEM",
        "icon_data",
        "cat_r",
        "cat_g",
        "cat_b",
    ]
    cols_keep = [c for c in cols_keep if c in df_mapa.columns]
    df_mapa = df_mapa[cols_keep]
    data_records = df_mapa.to_dict(orient="records")

    view_state = pdk.ViewState(
        latitude=float(df_mapa["lat"].mean()),
        longitude=float(df_mapa["lon"].mean()),
        zoom=zoom_inicial,
        pitch=pitch_inicial,
        bearing=bearing_inicial,
    )

    layers = []

    if modo_mapa == "Ícones Micropower":
        icon_layer = pdk.Layer(
            "IconLayer",
            data=data_records,
            get_icon="icon_data",
            get_position="[lon, lat]",
            get_size=tamanho_icone,
            size_scale=15,
            pickable=True,
        )
        layers.append(icon_layer)

        if camada_categoria:
            cat_layer = pdk.Layer(
                "ScatterplotLayer",
                data=data_records,
                get_position="[lon, lat]",
                get_radius=3000,
                get_fill_color="[cat_r, cat_g, cat_b]",
                get_line_color="[cat_r, cat_g, cat_b]",
                opacity=0.35,
                pickable=False,
            )
            layers.append(cat_layer)

    elif modo_mapa == "Mapa de calor":
        heat_layer = pdk.Layer(
            "HeatmapLayer",
            data=data_records,
            get_position="[lon, lat]",
            radiusPixels=heat_radius,
        )
        layers.append(heat_layer)

    else:  # Hexágonos
        hex_layer = pdk.Layer(
            "HexagonLayer",
            data=data_records,
            get_position="[lon, lat]",
            radius=raio_hex,
            elevation_scale=50,
            elevation_range=[0, 3000],
            extruded=True,
            pickable=True,
        )
        layers.append(hex_layer)

    tooltip_html = """
    <b>{NOME}</b><br/>
    Origem: {ORIGEM}<br/>
    Categoria: {CATEGORIA}<br/>
    {MUNICÍPIO} - {UF}<br/>
    {ENDEREÇO}<br/>
    SIGFI: {SIGFI}<br/>
    Energia total: {ENERGIA_TOTAL_KWH} kWh
    """

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_provider="mapbox",
        api_keys={"mapbox": MAPBOX_TOKEN},
        map_style=map_style,
        tooltip={
            "html": tooltip_html,
            "style": {"backgroundColor": "white", "color": "black"},
        },
    )

    st.pydeck_chart(deck, use_container_width=True, height=height)


# ==================================================
# ABAS: PAINEL EXECUTIVO x OPERAÇÃO
# ==================================================
tab_exec, tab_oper = st.tabs(["📊 Painel Executivo", "🔧 Operação / Detalhe"])

# ---------------- PAINEL EXECUTIVO ----------------
with tab_exec:
    exec_full = st.checkbox(
        "Exibir painel em tela inteira (somente mapa e indicadores laterais)",
        value=False,
    )

    if exec_full:
        # Mapa máximo + faixa lateral bem estreita
        col_map, col_info = st.columns([8, 1])

        with col_map:
            desenhar_mapa(df_f, height=780)

        with col_info:
            st.markdown(" ")  # pequeno espaçamento vertical
            st.subheader("Indicadores")
            render_metricas_exec_lado()

    else:
        st.subheader("Painel Executivo de Projetos")
        render_metricas(layout="wide")

        st.markdown("---")
        st.subheader("Mapa 3D – visão executiva")
        desenhar_mapa(df_f, height=520)

        if micro_categoria_df is not None:
            st.markdown("#### Microredes – capacidade instalada por categoria")
            st.dataframe(
                micro_categoria_df[["CATEGORIA", "Energia_MWh_fmt"]],
                use_container_width=True,
                height=260,
            )

# ---------------- OPERAÇÃO / DETALHE ----------------
with tab_oper:
    st.subheader("Operação / Detalhe")

    render_metricas(layout="wide")

    st.markdown("---")
    st.subheader("Mapa 3D – visão operacional")
    desenhar_mapa(df_f, height=520)

    st.subheader("Tabela de instalações (filtro atual)")
    cols_mostrar = [
        c
        for c in [
            "ID MICROPOWER/CGB",
            "ORIGEM",
            "UF",
            "REGIÃO",
            "MUNICÍPIO",
            "ENDEREÇO",
            "NOME",
            "PROJETO",
            "CATEGORIA",
            "DATA_DA_INSTALACAO",
            "SIGFI",
            "POTENCIA_NOMINAL_KWH",
            "ENERGIA_KWH",
            "ENERGIA_TOTAL_KWH",
        ]
        if c in df_f.columns
    ]
    st.dataframe(df_f[cols_mostrar], use_container_width=True, height=450)
