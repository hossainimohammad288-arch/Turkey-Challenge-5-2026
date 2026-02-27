import streamlit as st
import numpy as np
import pandas as pd
import pickle
import io
import base64
from sklearn.preprocessing import LabelEncoder

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@st.cache_data(show_spinner=False)
def _model_performance_png_bytes():
    if plt is None:
        return None

    df = pd.DataFrame(
        [
            {"Model": "Logistic Regression", "Metric Type": "Accuracy", "Value": 59.89},
            {"Model": "Decision Tree Classifier", "Metric Type": "Accuracy", "Value": 95.94},
            {"Model": "Random Forest Classifier", "Metric Type": "Accuracy", "Value": 95.99},
            {"Model": "SVM Classifier", "Metric Type": "Accuracy", "Value": 47.27},
            {"Model": "KNN Classifier", "Metric Type": "Accuracy", "Value": 93.01},
            {"Model": "Naive Bayes Classifier", "Metric Type": "Accuracy", "Value": 81.21},
            {"Model": "Linear Regression", "Metric Type": "MAE", "Value": 0.03022},
            {"Model": "Decision Tree Regressor", "Metric Type": "MAE", "Value": 0.00295},
            {"Model": "Random Forest Regressor", "Metric Type": "MAE", "Value": 0.00243},
            {"Model": "SVM Regressor", "Metric Type": "MAE", "Value": 0.04107},
            {"Model": "KNN Regressor", "Metric Type": "MAE", "Value": 0.00609},
        ]
    )

    acc_top = (
        df.loc[df["Metric Type"].eq("Accuracy")]
        .nlargest(2, "Value")
        .index
        .tolist()
    )
    mae_top = (
        df.loc[df["Metric Type"].eq("MAE")]
        .nsmallest(2, "Value")
        .index
        .tolist()
    )
    highlight_rows = set(acc_top + mae_top)

    fig, ax = plt.subplots(figsize=(12.5, 6.2), dpi=240)
    ax.axis("off")

    header_color = "#2c3e50"
    stripe_color = "#f2f5f7"
    highlight_color = "#d9fbe5"
    edge_color = "#9aa4ad"

    cell_text = []
    for _, r in df.iterrows():
        val = r["Value"]
        val_txt = f"{val:.2f}" if val >= 1 else f"{val:.5f}"
        cell_text.append([r["Model"], r["Metric Type"], val_txt])

    tbl = ax.table(
        cellText=cell_text,
        colLabels=["Model", "Metric Type", "Value"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.42)

    # Header styling
    for j in range(3):
        c = tbl[(0, j)]
        c.set_facecolor(header_color)
        c.set_text_props(color="white", weight="bold")
        c.set_edgecolor(edge_color)
        c.set_linewidth(1.1)

    # Body styling + highlights
    for i in range(1, len(df) + 1):
        row_idx = df.index[i - 1]
        for j in range(3):
            c = tbl[(i, j)]
            base = stripe_color if i % 2 == 0 else "white"
            if row_idx in highlight_rows:
                base = highlight_color
                c.set_text_props(weight="bold")
            c.set_facecolor(base)
            c.set_edgecolor(edge_color)
            c.set_linewidth(0.9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.6, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _background_image_base64():
    try:
        with open("pic.jpeg", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


st.set_page_config(
    page_title="Earthquake Building Damage Prediction",
    layout="centered"
)

if "app_started" not in st.session_state:
    st.session_state.app_started = False

# ================= Translation dictionary =================
TRANSLATIONS = {
    "en": {
        "language_name": "English",
        "hero_title": "Predicting Building Collapse After an Earthquake",
        "made_by": "Made By:",
        "main_title": "Earthquake Building Damage Prediction",
        "subtitle": "Predict structural damage based on building and site characteristics.",
        "input_section": "Input Features",
        "struct_type": "Structural Type",
        "occ_type": "Occupancy Type",
        "year_built": "Year Built",
        "no_stories": "Number of Stories",
        "magnitude": "Earthquake Magnitude",
        "distance": "Distance from Epicenter (km)",
        "predict_button": "Predict Damage",
        "results_title": "Prediction Results",
        "mean_damage": "Mean Damage Index",
        "damage_class": "Damage Class",
        "safe": "Safe",
        "high_risk": "High Risk",
        "collapsed": "Collapsed",
        "prediction_done": "Prediction completed successfully.",
        "model_summary": "Model Performance Summary",
        "expander_title": "Show model comparison results",
        "expander_caption": "Classification & Regression Model Performance",
    },
    "tr": {
        "language_name": "Türkçe",
        "hero_title": "Deprem Sonrası Bina Çökme Tahmini",
        "made_by": "Hazırlayanlar:",
        "main_title": "Deprem Bina Hasar Tahmini",
        "subtitle": "Bina ve zemin özelliklerine göre yapısal hasarı tahmin edin.",
        "input_section": "Girdi Özellikleri",
        "struct_type": "Taşıyıcı Sistem Türü",
        "occ_type": "Kullanım Türü",
        "year_built": "Yapım Yılı",
        "no_stories": "Kat Sayısı",
        "magnitude": "Deprem Büyüklüğü",
        "distance": "Merkez Üssüne Uzaklık (km)",
        "predict_button": "Hasarı Tahmin Et",
        "results_title": "Tahmin Sonuçları",
        "mean_damage": "Ortalama Hasar İndeksi",
        "damage_class": "Hasar Sınıfı",
        "safe": "Güvenli",
        "high_risk": "Yüksek Risk",
        "collapsed": "Yıkılmış",
        "prediction_done": "Tahmin başarıyla tamamlandı.",
        "model_summary": "Model Performans Özeti",
        "expander_title": "Model karşılaştırma sonuçlarını göster",
        "expander_caption": "Sınıflandırma ve Regresyon Model Performansı",
    },
    "fr": {
        "language_name": "Français",
        "hero_title": "Prédiction de l'effondrement des bâtiments après un séisme",
        "made_by": "Réalisé par :",
        "main_title": "Prédiction des dommages aux bâtiments",
        "subtitle": "Prédisez les dégâts structurels selon les caractéristiques du bâtiment et du site.",
        "input_section": "Caractéristiques en entrée",
        "struct_type": "Type de structure",
        "occ_type": "Type d'occupation",
        "year_built": "Année de construction",
        "no_stories": "Nombre d'étages",
        "magnitude": "Magnitude du séisme",
        "distance": "Distance à l'épicentre (km)",
        "predict_button": "Prédire les dégâts",
        "results_title": "Résultats de la prédiction",
        "mean_damage": "Indice moyen de dégâts",
        "damage_class": "Classe de dégâts",
        "safe": "Sûr",
        "high_risk": "Risque élevé",
        "collapsed": "Effondré",
        "prediction_done": "Prédiction terminée avec succès.",
        "model_summary": "Résumé des performances du modèle",
        "expander_title": "Afficher la comparaison des modèles",
        "expander_caption": "Performances des modèles de classification et de régression",
    },
    "de": {
        "language_name": "Deutsch",
        "hero_title": "Prognose von Gebäudeeinstürzen nach Erdbeben",
        "made_by": "Erstellt von:",
        "main_title": "Vorhersage von Erdbebenschäden an Gebäuden",
        "subtitle": "Sagen Sie strukturelle Schäden anhand von Gebäude- und Standortmerkmalen voraus.",
        "input_section": "Eingabemerkmale",
        "struct_type": "Strukturtyp",
        "occ_type": "Nutzungstyp",
        "year_built": "Baujahr",
        "no_stories": "Anzahl der Stockwerke",
        "magnitude": "Erdbebenstärke",
        "distance": "Entfernung zum Epizentrum (km)",
        "predict_button": "Schäden vorhersagen",
        "results_title": "Vorhersageergebnisse",
        "mean_damage": "Mittlerer Schadensindex",
        "damage_class": "Schadensklasse",
        "safe": "Sicher",
        "high_risk": "Hohes Risiko",
        "collapsed": "Eingestürzt",
        "prediction_done": "Vorhersage erfolgreich abgeschlossen.",
        "model_summary": "Zusammenfassung der Modellleistung",
        "expander_title": "Modellvergleich anzeigen",
        "expander_caption": "Leistung der Klassifikations- und Regressionsmodelle",
    },
    "zh": {
        "language_name": "中文",
        "hero_title": "地震后建筑物倒塌预测",
        "made_by": "制作：",
        "main_title": "地震建筑损坏预测",
        "subtitle": "根据建筑物和场地特征预测结构损坏程度。",
        "input_section": "输入特征",
        "struct_type": "结构类型",
        "occ_type": "使用类型",
        "year_built": "建造年份",
        "no_stories": "楼层数",
        "magnitude": "地震震级",
        "distance": "距震中距离（千米）",
        "predict_button": "预测损坏",
        "results_title": "预测结果",
        "mean_damage": "平均损坏指数",
        "damage_class": "损坏等级",
        "safe": "安全",
        "high_risk": "高风险",
        "collapsed": "倒塌",
        "prediction_done": "预测已成功完成。",
        "model_summary": "模型性能概览",
        "expander_title": "显示模型对比结果",
        "expander_caption": "分类与回归模型性能",
    },
    "ru": {
        "language_name": "Русский",
        "hero_title": "Прогноз обрушения зданий после землетрясения",
        "made_by": "Авторы:",
        "main_title": "Прогноз повреждений зданий при землетрясении",
        "subtitle": "Прогнозируйте структурные повреждения по характеристикам здания и участка.",
        "input_section": "Входные характеристики",
        "struct_type": "Тип конструкции",
        "occ_type": "Тип использования",
        "year_built": "Год постройки",
        "no_stories": "Количество этажей",
        "magnitude": "Магнитуда землетрясения",
        "distance": "Расстояние до эпицентра (км)",
        "predict_button": "Прогнозировать повреждения",
        "results_title": "Результаты прогноза",
        "mean_damage": "Средний индекс повреждений",
        "damage_class": "Класс повреждений",
        "safe": "Безопасно",
        "high_risk": "Высокий риск",
        "collapsed": "Обрушено",
        "prediction_done": "Прогноз успешно завершён.",
        "model_summary": "Сводка по точности модели",
        "expander_title": "Показать сравнение моделей",
        "expander_caption": "Эффективность моделей классификации и регрессии",
    },
    "fa": {
        "language_name": "فارسی",
        "hero_title": "پیش‌بینی فروریختن ساختمان پس از زلزله",
        "made_by": "تهیه‌کنندگان:",
        "main_title": "پیش‌بینی خسارت ساختمان در زلزله",
        "subtitle": "بر اساس ویژگی‌های ساختمان و محل، میزان خسارت سازه را پیش‌بینی کنید.",
        "input_section": "ویژگی‌های ورودی",
        "struct_type": "نوع سیستم سازه‌ای",
        "occ_type": "نوع کاربری",
        "year_built": "سال ساخت",
        "no_stories": "تعداد طبقات",
        "magnitude": "بزرگی زلزله",
        "distance": "فاصله تا کانون زلزله (کیلومتر)",
        "predict_button": "پیش‌بینی خسارت",
        "results_title": "نتایج پیش‌بینی",
        "mean_damage": "شاخص میانگین خسارت",
        "damage_class": "کلاس خسارت",
        "safe": "ایمن",
        "high_risk": "پرخطر",
        "collapsed": "فروریخته",
        "prediction_done": "پیش‌بینی با موفقیت انجام شد.",
        "model_summary": "خلاصه عملکرد مدل",
        "expander_title": "نمایش نتایج مقایسه مدل‌ها",
        "expander_caption": "عملکرد مدل‌های طبقه‌بندی و رگرسیون",
    },
    "ar": {
        "language_name": "العربية",
        "hero_title": "تنبؤ بانهيار المباني بعد الزلازل",
        "made_by": "إعداد:",
        "main_title": "تنبؤ أضرار المباني في الزلازل",
        "subtitle": "تنبأ بالأضرار الإنشائية بناءً على خصائص المبنى والموقع.",
        "input_section": "المدخلات",
        "struct_type": "نوع الهيكل الإنشائي",
        "occ_type": "نوع الإشغال",
        "year_built": "سنة البناء",
        "no_stories": "عدد الطوابق",
        "magnitude": "قوة الزلزال",
        "distance": "المسافة عن مركز الزلزال (كم)",
        "predict_button": "تنبؤ الأضرار",
        "results_title": "نتائج التنبؤ",
        "mean_damage": "مؤشر متوسط الأضرار",
        "damage_class": "فئة الأضرار",
        "safe": "آمن",
        "high_risk": "عالي الخطورة",
        "collapsed": "منهار",
        "prediction_done": "اكتمل التنبؤ بنجاح.",
        "model_summary": "ملخص أداء النموذج",
        "expander_title": "عرض نتائج مقارنة النماذج",
        "expander_caption": "أداء نماذج التصنيف والانحدار",
    },
}

_bg_image_b64 = _background_image_base64()

_base_css = """
<style>
/* =============== Background: earthquake & city =============== */
.stApp {
    background:
        /* make gradient much lighter so photo is clearer */
        linear-gradient(
            135deg,
            rgba(15, 23, 42, 0.45),
            rgba(127, 29, 29, 0.55)
        ),
        url("data:image/jpeg;base64,___BG_IMAGE___");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* faint seismic grid overlay */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(to right, rgba(148, 163, 184, 0.12) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(148, 163, 184, 0.12) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: -1;
}

/* =============== Motion & micro-interactions =============== */
@keyframes fadeInUp {
    from { opacity: 0; transform: translate3d(0, 10px, 0); }
    to   { opacity: 1; transform: translate3d(0, 0, 0); }
}

@keyframes popIn {
    0%   { opacity: 0; transform: translate3d(0, 6px, 0) scale(0.985); }
    100% { opacity: 1; transform: translate3d(0, 0, 0) scale(1); }
}

@keyframes auroraMove {
    0%   { transform: translate3d(-2%, -1%, 0) scale(1); filter: blur(0px); }
    100% { transform: translate3d(2%, 1%, 0) scale(1.04); filter: blur(0.2px); }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
        scroll-behavior: auto !important;
    }
}

/* soft animated glow overlay (very subtle) */
.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(circle at 18% 25%, rgba(249, 115, 22, 0.14), transparent 55%),
        radial-gradient(circle at 82% 60%, rgba(59, 130, 246, 0.10), transparent 58%);
    animation: auroraMove 10s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: -1;
    mix-blend-mode: screen;
}

/* =============== Global typography =============== */
html, body, label, span, p, div,
h1, h2, h3, h4, h5, h6 {
    color: rgba(248, 250, 252, 0.96) !important;
}

/* remove default Streamlit chrome a bit */
[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stToolbar"] {
    right: 1rem;
}

/* =============== Main content cards =============== */
section[data-testid="stVerticalBlock"] {
    background: rgba(2, 6, 23, 0.62) !important;
    padding: 24px;
    border-radius: 18px;
    box-shadow:
        0 20px 45px rgba(15, 23, 42, 0.45),
        0 0 0 1px rgba(148, 163, 184, 0.25);
    border: 1px solid rgba(148, 163, 184, 0.18);
    backdrop-filter: blur(10px);
    animation: fadeInUp 520ms ease both;
    transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
}

section[data-testid="stVerticalBlock"]:hover {
    transform: translate3d(0, -2px, 0);
    border-color: rgba(249, 115, 22, 0.30);
    box-shadow:
        0 26px 58px rgba(15, 23, 42, 0.55),
        0 0 0 1px rgba(249, 115, 22, 0.22);
}

/* =============== Inputs =============== */
input, textarea {
    background-color: rgba(15, 23, 42, 0.75) !important;
    color: rgba(248, 250, 252, 0.96) !important;
    border: 1px solid rgba(148, 163, 184, 0.35) !important;
    border-radius: 10px !important;
    transition: border-color 160ms ease, box-shadow 160ms ease, background-color 160ms ease;
}

/* Selectbox (BaseWeb) */
div[data-baseweb="select"] {
    background-color: rgba(15, 23, 42, 0.75) !important;
    border-radius: 10px;
}

div[data-baseweb="select"] > div {
    background-color: rgba(15, 23, 42, 0.75) !important;
    color: rgba(248, 250, 252, 0.96) !important;
}

div[data-baseweb="select"] > div:focus,
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="select"][aria-expanded="true"] > div {
    background-color: rgba(15, 23, 42, 0.85) !important;
    color: rgba(248, 250, 252, 0.96) !important;
    border: 1px solid #f97316 !important;
    box-shadow: 0 0 0 1px rgba(249, 115, 22, 0.3);
}

ul[role="listbox"] {
    background-color: rgba(2, 6, 23, 0.98) !important;
    color: rgba(248, 250, 252, 0.96) !important;
    border: 1px solid rgba(148, 163, 184, 0.35) !important;
}

li[role="option"] {
    background-color: rgba(2, 6, 23, 0.98) !important;
    color: rgba(248, 250, 252, 0.96) !important;
    border: none !important;
}

li[role="option"]:hover {
    background-color: rgba(30, 41, 59, 0.8) !important;
    color: rgba(248, 250, 252, 0.96) !important;
}

/* =============== Language control (top-left) =============== */
div[data-testid="stLanguageControl"] {
    position: fixed;
    top: 16px;
    left: 16px;
    z-index: 1000;
    width: 360px;
}

div[data-testid="stLanguageControl"] [role="radiogroup"] {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(2, 6, 23, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.22);
    backdrop-filter: blur(10px);
}

div[data-testid="stLanguageControl"] label {
    background: rgba(148, 163, 184, 0.16) !important;
    border: 1px solid rgba(148, 163, 184, 0.32) !important;
    border-radius: 999px !important;
    padding: 6px 10px !important;
    font-size: 14px !important;
    line-height: 1.1 !important;
    transition: transform 160ms ease, background-color 160ms ease, border-color 160ms ease;
}

div[data-testid="stLanguageControl"] label:hover {
    background: rgba(148, 163, 184, 0.26) !important;
    transform: translate3d(0, -1px, 0);
}

/* =============== Buttons =============== */
button {
    background: linear-gradient(135deg, #f97316, #b91c1c) !important;
    color: #f9fafb !important;
    border-radius: 999px !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    transition: transform 160ms ease, filter 160ms ease, box-shadow 160ms ease;
}

button:hover {
    background: linear-gradient(135deg, #fb923c, #dc2626) !important;
    transform: translate3d(0, -2px, 0);
    filter: brightness(1.05);
    box-shadow: 0 14px 28px rgba(0, 0, 0, 0.25);
}

/* number input +/- */
div[data-testid="stNumberInput"] button {
    background-color: rgba(15, 23, 42, 0.75) !important;
    color: rgba(248, 250, 252, 0.96) !important;
    border-radius: 8px !important;
}

/* =============== Metrics (damage cards) =============== */
div[data-testid="metric-container"] {
    background: radial-gradient(circle at top left, rgba(127, 29, 29, 0.55), rgba(2, 6, 23, 0.85)) !important;
    color: rgba(248, 250, 252, 0.96) !important;
    border-radius: 14px;
    padding: 16px;
    border: 1px solid rgba(248, 113, 113, 0.35);
    box-shadow: 0 12px 30px rgba(127, 29, 29, 0.25);
    animation: popIn 420ms cubic-bezier(0.2, 0.8, 0.2, 1) both;
    transition: transform 200ms ease, box-shadow 200ms ease, border-color 200ms ease;
}

div[data-testid="metric-container"]:hover {
    transform: translate3d(0, -2px, 0);
    border-color: rgba(248, 113, 113, 0.55);
    box-shadow: 0 18px 44px rgba(127, 29, 29, 0.35);
}

/* =============== Expander =============== */
details {
    background-color: rgba(2, 6, 23, 0.45) !important;
    border-radius: 12px;
    padding: 10px;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    animation: fadeInUp 520ms ease both;
}

summary {
    background-color: rgba(15, 23, 42, 0.55) !important;
    color: rgba(248, 250, 252, 0.96) !important;
    padding: 10px;
    border-radius: 10px;
    font-weight: 600;
}

summary:hover {
    background-color: rgba(30, 41, 59, 0.7) !important;
    color: rgba(248, 250, 252, 0.96) !important;
}

details > div {
    background-color: rgba(2, 6, 23, 0.25) !important;
    color: rgba(248, 250, 252, 0.96) !important;
}

/* =============== Landing page =============== */
.landing {
    max-width: 820px;
    margin: 10vh auto 0 auto;
    padding: 28px 26px;
    border-radius: 22px;
    background: rgba(2, 6, 23, 0.62);
    border: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow: 0 25px 60px rgba(2, 6, 23, 0.55);
    backdrop-filter: blur(10px);
    animation: fadeInUp 650ms ease both;
}
.landing h1 {
    margin: 0 0 10px 0;
    font-size: 38px;
    line-height: 1.2;
    color: rgba(248, 250, 252, 0.98) !important;
}
.landing p {
    margin: 0;
    font-size: 16px;
    color: rgba(226, 232, 240, 0.9) !important;
}
</style>
"""

st.markdown(
    _base_css.replace("___BG_IMAGE___", _bg_image_b64),
    unsafe_allow_html=True
)

if not st.session_state.app_started:
    st.markdown(
        """
        <div class="landing">
            <h1>Are you ready to predict the damage?</h1>
            <p>Click <b>Yes!</b> to start.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        if st.button("Yes!", use_container_width=True):
            st.session_state.app_started = True
            st.rerun()
    st.stop()

else:
    # When app has started (after Yes), make background image softer/dimmer
    st.markdown(
        """
        <style>
        .stApp {
            background:
                linear-gradient(
                    135deg,
                    rgba(15, 23, 42, 0.70),
                    rgba(127, 29, 29, 0.60)
                ),
                url("data:image/jpeg;base64,___BG_IMAGE___");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        </style>
        """.replace("___BG_IMAGE___", _bg_image_b64),
        unsafe_allow_html=True,
    )

st.markdown('<div data-testid="stLanguageControl">', unsafe_allow_html=True)
lang = st.radio(
    "",
    options=list(TRANSLATIONS.keys()),
    format_func=lambda code: TRANSLATIONS[code]["language_name"],
    horizontal=True,
    label_visibility="collapsed",
    key="language",
)
st.markdown("</div>", unsafe_allow_html=True)
t = TRANSLATIONS[lang]

if lang in ["fa", "ar"]:
    direction_css = """
    <style>
    body, .stApp, .block-container {
        direction: rtl;
        text-align: right;
    }

    /* fix metric alignment */
    div[data-testid="metric-container"] {
        direction: rtl;
        text-align: right;
    }

    /* fix selectboxes */
    div[data-baseweb="select"] * {
        direction: rtl !important;
        text-align: right !important;
    }
    </style>
    """
else:
    direction_css = """
    <style>
    body, .stApp, .block-container {
        direction: ltr;
        text-align: left;
    }

    div[data-testid="metric-container"] {
        direction: ltr;
        text-align: left;
    }

    div[data-baseweb="select"] * {
        direction: ltr !important;
        text-align: left !important;
    }
    </style>
    """

st.markdown(direction_css, unsafe_allow_html=True)

col_flag, col_info = st.columns([1, 4])

with col_flag:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/b/b4/Flag_of_Turkey.svg",
        width=80,
    )

with col_info:
    authors = [
        "Seyed Mohammad Hosseini",
        "Mohammad Mahan Haghi",
        "Kourosh Ameri Far",
        "Seyed Mohammadparsa Azimi",
    ]
    if lang in ["fa", "ar"]:
        authors = [
            "سید محمد حسینی",
            "محمد ماهان حقی",
            "کوروش عامری فر",
            "سید محمد پارسا عظیمی",
        ]
    authors_md = "  \n".join(authors)
    st.markdown(
        f"""
        **{t['hero_title']}**  
        **{t['made_by']}**  
        {authors_md}
        """
    )

st.markdown("---")

st.markdown(f"## 🏗️ {t['main_title']}")
st.write(t["subtitle"])

with open("models/model_forest_classifier.pickle", "rb") as f:
    clf_model = pickle.load(f)

with open("models/model_tree_regressor.pickle", "rb") as f:
    reg_model = pickle.load(f)

data = pd.read_csv("building_damage.csv")
data = data.drop("Unnamed: 0", axis=1)

OCC_TYPES_BY_LANG = {
    "en": {
        "Residential": ["RES1", "RES3", "RES4"],
        "Commercial": ["COM1", "COM2", "COM3", "COM4", "COM7", "COM8"],
        "Industrial": ["IND1", "IND2", "IND3"],
        "Agricultural": ["AGR1"],
        "Educational": ["EDU1"],
        "Religious": ["REL1"],
        "Governmental": ["GOV1"],
    },
    "tr": {
        "Konut (Residential)": ["RES1", "RES3", "RES4"],
        "Ticari (Commercial)": ["COM1", "COM2", "COM3", "COM4", "COM7", "COM8"],
        "Endüstriyel (Industrial)": ["IND1", "IND2", "IND3"],
        "Tarımsal (Agricultural)": ["AGR1"],
        "Eğitim (Educational)": ["EDU1"],
        "Dini (Religious)": ["REL1"],
        "Kamu (Governmental)": ["GOV1"],
    },
    "fr": {
        "Résidentiel": ["RES1", "RES3", "RES4"],
        "Commercial": ["COM1", "COM2", "COM3", "COM4", "COM7", "COM8"],
        "Industriel": ["IND1", "IND2", "IND3"],
        "Agricole": ["AGR1"],
        "Éducatif": ["EDU1"],
        "Religieux": ["REL1"],
        "Gouvernemental": ["GOV1"],
    },
    "de": {
        "Wohngebäude": ["RES1", "RES3", "RES4"],
        "Gewerblich": ["COM1", "COM2", "COM3", "COM4", "COM7", "COM8"],
        "Industriell": ["IND1", "IND2", "IND3"],
        "Landwirtschaftlich": ["AGR1"],
        "Bildungseinrichtung": ["EDU1"],
        "Religiös": ["REL1"],
        "Staatlich": ["GOV1"],
    },
    "zh": {
        "住宅 (Residential)": ["RES1", "RES3", "RES4"],
        "商业 (Commercial)": ["COM1", "COM2", "COM3", "COM4", "COM7", "COM8"],
        "工业 (Industrial)": ["IND1", "IND2", "IND3"],
        "农业 (Agricultural)": ["AGR1"],
        "教育 (Educational)": ["EDU1"],
        "宗教 (Religious)": ["REL1"],
        "政府 (Governmental)": ["GOV1"],
    },
    "ru": {
        "Жилое (Residential)": ["RES1", "RES3", "RES4"],
        "Коммерческое (Commercial)": ["COM1", "COM2", "COM3", "COM4", "COM7", "COM8"],
        "Промышленное (Industrial)": ["IND1", "IND2", "IND3"],
        "Сельскохозяйственное (Agricultural)": ["AGR1"],
        "Образовательное (Educational)": ["EDU1"],
        "Религиозное (Religious)": ["REL1"],
        "Государственное (Governmental)": ["GOV1"],
    },
    "fa": {
        "مسکونی": ["RES1", "RES3", "RES4"],
        "تجاری": ["COM1", "COM2", "COM3", "COM4", "COM7", "COM8"],
        "صنعتی": ["IND1", "IND2", "IND3"],
        "کشاورزی": ["AGR1"],
        "آموزشی": ["EDU1"],
        "مذهبی": ["REL1"],
        "دولتی": ["GOV1"],
    },
    "ar": {
        "سكني": ["RES1", "RES3", "RES4"],
        "تجاري": ["COM1", "COM2", "COM3", "COM4", "COM7", "COM8"],
        "صناعي": ["IND1", "IND2", "IND3"],
        "زراعي": ["AGR1"],
        "تعليمي": ["EDU1"],
        "ديني": ["REL1"],
        "حكومي": ["GOV1"],
    },
}

occ_type_display = OCC_TYPES_BY_LANG.get(lang, OCC_TYPES_BY_LANG["en"])

encoders = {}
for col in data.columns:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

STRUCT_TYPES_BY_LANG = {
    "en": {
        "Unreinforced Masonry (URM)": "URM",
        "Steel Moment Frame (S1)": "S1",
        "Reinforced Concrete Moment Frame (C4)": "C4",
        "Wooden Frame (W1)": "W1",
        "Precast Concrete (PC1)": "PC1",
        "Reinforced Concrete Shear Wall (C1)": "C1",
    },
    "tr": {
        "Yığma (URM)": "URM",
        "Çelik Moment Çerçeve (S1)": "S1",
        "Betonarme Moment Çerçeve (C4)": "C4",
        "Ahşap Çerçeve (W1)": "W1",
        "Ön Dökümlü Beton (PC1)": "PC1",
        "Betonarme Perde Duvar (C1)": "C1",
    },
    "fr": {
        "Maçonnerie non armée (URM)": "URM",
        "Charpente métallique à portique (S1)": "S1",
        "Portique en béton armé (C4)": "C4",
        "Structure en bois (W1)": "W1",
        "Béton préfabriqué (PC1)": "PC1",
        "Voiles en béton armé (C1)": "C1",
    },
    "de": {
        "Unbewehrtes Mauerwerk (URM)": "URM",
        "Stahlmomentrahmen (S1)": "S1",
        "Stahlbetonmomentrahmen (C4)": "C4",
        "Holzrahmen (W1)": "W1",
        "Fertigbetonbau (PC1)": "PC1",
        "Stahlbeton-Scheibenwand (C1)": "C1",
    },
    "zh": {
        "未加固砌体结构 (URM)": "URM",
        "钢框架结构 (S1)": "S1",
        "钢筋混凝土框架 (C4)": "C4",
        "木结构框架 (W1)": "W1",
        "预制混凝土结构 (PC1)": "PC1",
        "钢筋混凝土剪力墙 (C1)": "C1",
    },
    "ru": {
        "Ненармированная кладка (URM)": "URM",
        "Стальной рамный каркас (S1)": "S1",
        "Железобетонный рамный каркас (C4)": "C4",
        "Деревянный каркас (W1)": "W1",
        "Сборный железобетон (PC1)": "PC1",
        "Железобетонные стены-диафрагмы (C1)": "C1",
    },
    "fa": {
        "مصالح بنایی بدون مسلح (URM)": "URM",
        "قاب خمشی فولادی (S1)": "S1",
        "قاب خمشی بتن‌آرمه (C4)": "C4",
        "قاب چوبی (W1)": "W1",
        "بتن پیش‌ساخته (PC1)": "PC1",
        "دیوار برشی بتن‌آرمه (C1)": "C1",
    },
    "ar": {
        "مباني طوب غير مسلحة (URM)": "URM",
        "إطار لحظي فولاذي (S1)": "S1",
        "إطار لحظي خرسانة مسلحة (C4)": "C4",
        "إطار خشبي (W1)": "W1",
        "خرسانة مسبقة الصب (PC1)": "PC1",
        "جدار قص خرسانة مسلحة (C1)": "C1",
    },
}

struct_type_display = STRUCT_TYPES_BY_LANG.get(lang, STRUCT_TYPES_BY_LANG["en"])

st.subheader(f"🔢 {t['input_section']}")

struct_display_choice = st.selectbox(
    t["struct_type"],
    list(struct_type_display.keys())
)
struct_typ = struct_type_display[struct_display_choice]

occ_choice = st.selectbox(
    t["occ_type"],
    list(occ_type_display.keys())
)
occ_type_code = occ_type_display[occ_choice][0]

year_built = st.number_input(t["year_built"], 1985, 2017, 2000)
no_stories = st.number_input(t["no_stories"], 0, 30, 0)
magnitude = st.number_input(t["magnitude"], value=5.0)
distance = st.number_input(t["distance"], value=3.0)

X = np.array([[
    encoders["struct_typ"].transform([struct_typ])[0],
    encoders["occ_type"].transform([occ_type_code])[0],
    year_built,
    no_stories,
    magnitude,
    distance
]])

if st.button(f"🚀 {t['predict_button']}"):
    meandamage_pred = reg_model.predict(X)[0]
    damage_class_pred = clf_model.predict(X)[0]

    damage_map = {
        0: f"🟢 {t['safe']}",
        1: f"🟠 {t['high_risk']}",
        2: f"🔴 {t['collapsed']}",
    }

    st.subheader(f"📊 {t['results_title']}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(t["mean_damage"], round(float(meandamage_pred), 4))
    with col2:
        st.metric(t["damage_class"], damage_map[int(damage_class_pred)])

    st.success(f"{t['prediction_done']} ✅")

st.markdown("---")
st.subheader(f"📈 {t['model_summary']}")

with st.expander(t["expander_title"], expanded=True):
    table_html = """
    <style>
    .perf-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    .perf-table thead tr {
        background-color: #1f2933;
        color: #f9fafb;
    }
    .perf-table th,
    .perf-table td {
        border: 1px solid rgba(148, 163, 184, 0.6);
        padding: 6px 10px;
        text-align: center;
        white-space: nowrap;
    }
    .perf-table tbody tr:nth-child(even) {
        background-color: rgba(15, 23, 42, 0.6);
    }
    .perf-table tbody tr:nth-child(odd) {
        background-color: rgba(15, 23, 42, 0.9);
    }
    .perf-table tbody tr.highlight {
        background-color: rgba(34, 197, 94, 0.25);
        font-weight: 700;
    }
    </style>
    <div style="overflow-x:auto;">
    <table class="perf-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Metric Type</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Logistic Regression</td>
          <td>Accuracy</td>
          <td>59.89</td>
        </tr>
        <tr>
          <td>Decision Tree Classifier</td>
          <td>Accuracy</td>
          <td><b>95.94</b></td>
        </tr>
        <tr class="highlight">
          <td>Random Forest Classifier</td>
          <td>Accuracy</td>
          <td><b>95.99</b></td>
        </tr>
        <tr>
          <td>SVM Classifier</td>
          <td>Accuracy</td>
          <td>47.27</td>
        </tr>
        <tr>
          <td>KNN Classifier</td>
          <td>Accuracy</td>
          <td>93.01</td>
        </tr>
        <tr>
          <td>Naive Bayes Classifier</td>
          <td>Accuracy</td>
          <td>81.21</td>
        </tr>
        <tr>
          <td>Linear Regression</td>
          <td>MAE</td>
          <td>0.03022</td>
        </tr>
        <tr class="highlight">
          <td>Decision Tree Regressor</td>
          <td>MAE</td>
          <td><b>0.00295</b></td>
        </tr>
        <tr>
          <td>Random Forest Regressor</td>
          <td>MAE</td>
          <td>0.00243</td>
        </tr>
        <tr>
          <td>SVM Regressor</td>
          <td>MAE</td>
          <td>0.04107</td>
        </tr>
        <tr>
          <td>KNN Regressor</td>
          <td>MAE</td>
          <td>0.00609</td>
        </tr>
      </tbody>
    </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.caption(t["expander_caption"])

st.markdown("---")