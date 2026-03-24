"""
=============================================================================
  Zindi | data.org Financial Health Prediction Challenge
  Faz 7 — V12.1  HOTFIX (P0-A + P0-B + P0-C + Kalibrasyon Kontrolü)
  ─────────────────────────────────────────────────────────────────────────
  V12.1 Değişiklikleri (V12 üzerine — 5 cerrahi patch):

  [P0-A] Optuna metrik bug fix:
         hrec = f1_score(labels=[2]) → recall_score(labels=[2], average="macro")
         Log'daki "High Recall" artık gerçekten recall.
         Band seçimi doğru hedefe göre yapılıyor.

  [P0-B] ALPHA_ACCEPT_WF1 dinamik:
         Hard-coded 0.8687 (V11 leakage dünyası) → baseline_wf1 - 0.001
         Faz-0 artık matematiksel olarak geçilebilir.

  [P0-C] CatBoost fix (OD erken fren çözümü):
         auto_class_weights="Balanced" → KALDIRILDI
         explicit class_weights={0:1, 1:2, 2:12} → EKLENDİ
         learning_rate: 0.05 → 0.02
         iterations: 1000 → 3000
         early_stopping_rounds: 80 → 150
         depth: 6 → 7
         Hedef: CB'nin çoğu fold'da ≥500 iterasyonda kapanması.

  [P0-D] Proba kalibrasyon kontrolü (ECE raporu):
         CB fix sonrası gated rescue güvenilirliği için
         CB OOF proba'larının kalibrasyon durumu raporlanır.
         Threshold 0.35 geçerli mi, yoksa ayarlanmalı mı?

  [P1-A] Submission bağımsızlaştırma:
         SampleSubmission.csv yoksa pipeline kırılmıyor.
         Doğrudan ID+Target DataFrame'den submission üretilir.
  ─────────────────────────────────────────────────────────────────────────
  V12 Değişiklikleri (V9 üzerine):
  [FIX-TE]   ZindiClusterEngineer TE Leakage Düzeltmesi:
             fit() → oof_te kaydedilir (self.oof_te_)
             transform(is_train=True)  → oof_te kullanır  (train için)
             transform(is_train=False) → final_map kullanır (val/test için)
  [FIX-OPT]  Optuna: PRIMARY=WF1, seçim=best-0.001 bandından HRec max
             Eski: macro-F1 maximize
  [NEW-F0]   Faz 0 Test Paketi:
             faz0_recoverable_high()   → V12 yönünü belirler
             faz0_alpha_sweep()        → guardrail'li α testi
             faz0_gated_sweep()        → FPR/TPR threshold sweep
             faz0_medium_baseline()    → kill-switch baseline
  [NEW-SF]   Split Freeze: fold indeksleri dosyaya kaydedilir
  [NEW-SS]   Seed Stability: metrikler seed_mean ± seed_std raporlanır
  [NEW-PHC]  pred_high_count her deneyde loglanır
  [NEW-ES2]  Early-stop çift koşul: WF1 band AND HRec trend flat
  CatBoost + LightGBM + XGBoost  ×  Optuna Weighted Blend
  ─────────────────────────────────────────────────────────────────────────
  STRATEJİK KARAR: Üç teknikten neden bu ikili?
  ─────────────────────────────────────────────────────────────────────────

  K-FOLD TARGET ENCODING (tek başına) → ELEND
    country (4 değer) ve owner_sex (2 değer) düşük kardinalite.
    TE yüksek kardinaliteli kolonlarda güçlüdür; burada kazanç marjinal.

  PSEUDO-LABELING → ELEND (bu aşamada)
    Zirvedeyken test hatasını modele yedirilecek gürültüyle amplify etmek
    Private LB'de geri düşme riskini Optuna'dan çok daha yüksek tutar.
    Doğru yapılırsa güçlü, ama yanlış eşik = ölüm.

  K-MEANS + K-FOLD TE → SEÇİLDİ (sinerji)
    Adım 1 — K-Means: 16 FE özelliğini "KOBİ arketipi" kümelerine sıkıştırır.
    Örnek arketipler: "Kayıtsız, sigortasız, nakit sıkıntılı", "Dijital, sigortalı, büyüyen"
    Bireysel özellikler yakalayamadığı yüksek-mertebe etkileşimler cluster'da gizlidir.

    Adım 2 — K-Fold Target Encoding: her cluster'ın "Low/Medium/High olma olasılığını"
    öğrenir. Bu, modele "Bu arketip tarihsel olarak ne sıklıkla High FHI'ya sahip?"
    sorusunun cevabını doğrudan özellik olarak verir.

    İkili sinerji: cluster_id kardinaliteyi yükseltir (4→8-12),
    TE bunu anlamlı bir sinyale dönüştürür. Ayrı ayrı zayıf, birlikte güçlü.

  ─────────────────────────────────────────────────────────────────────────
  V9 Yenilikleri:
  [NEW-8]  ZindiClusterEngineer — sklearn TransformerMixin
           • fit(X_trn, y_trn): StandardScaler + KMeans(k=8) + KMeans(k=12)
             SADECE X_trn üzerinde → leakage-free garantili
           • Çok sınıflı K-Fold TE: her cluster için P(Low), P(Mid), P(High)
             iç 3-Fold CV ile öğrenilir; Laplace smoothing ile aşırı fit önlenir
           • transform(X): cluster_id + dist + 3 olasılık TE = 5 × 2k özellik

  [NEW-9]  Dinamik CAT_FEATURES genişletme
           cluster_id_k8 ve cluster_id_k12 CatBoost Pool'a,
           LightGBM'e category dtype olarak, XGBoost'a int olarak iletilir.
           Her fold'da sütun listesi otomatik güncellenir.

  ─────────────────────────────────────────────────────────────────────────
  Korunan V7 Özellikleri (DOKUNULMADI):
  [FIX-7]  align_columns → reindex() mutation-safe
  [FIX-8]  FE-06 log1p stabilizasyonu
  [FIX-4]  Leakage-free FE → her fold'da fit sadece X_trn'e
  [FIX-5]  Safe cast → fillna(0).astype(int)
  [FIX-1]  Manuel ordinal Target mapping
  [NEW-5]  XGBoost + sample_weight
  [NEW-6]  Optuna Simplex Blend
  ─────────────────────────────────────────────────────────────────────────
  Üretilen Özellikler (her k için):
    cluster_id_k{k}      → KOBİ arketip etiketi (kategorik)
    cluster_dist_k{k}    → arketipe uzaklık (arketip içi homojenlik)
    cluster_te_low_k{k}  → P(Low  | bu arketip)  [K-Fold TE]
    cluster_te_mid_k{k}  → P(Mid  | bu arketip)  [K-Fold TE]
    cluster_te_high_k{k} → P(High | bu arketip)  [K-Fold TE]

  Model    : CatBoost + LightGBM + XGBoost
  Blend    : Optuna Simplex (n_trials=300, Macro-F1 hedefi)
  Strateji : StratifiedKFold (n=5) + OOF + K-Means + K-Fold TE
  Metrik   : Weighted-F1 / Macro-F1
=============================================================================
"""

# ─── Standart Kütüphaneler ────────────────────────────────────────────────────
import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Modelleme ────────────────────────────────────────────────────────────────
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
# --- [HOTFIX] Ensure recall_score is always available (Notebook-safe) ---
try:
    from sklearn.metrics import recall_score  # noqa: F401
except Exception:  # pragma: no cover
    def recall_score(*args, **kwargs):
        from sklearn.metrics import recall_score as _recall_score
        return _recall_score(*args, **kwargs)
# ----------------------------------------------------------------------
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.cluster import KMeans

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# GLOBAL SABİTLER
# =============================================================================

SEED       = 42
N_SPLITS   = 5
TARGET_COL = "Target"
ID_COL     = "ID"

TARGET_MAPPING         = {"Low": 0, "Medium": 1, "High": 2}
TARGET_INVERSE_MAPPING = {v: k for k, v in TARGET_MAPPING.items()}
N_CLASSES              = len(TARGET_MAPPING)
CLASS_NAMES            = ["Low", "Medium", "High"]

CAT_FEATURES   = ["country", "owner_sex"]
OPTUNA_TRIALS  = 300

# [NEW-8] K-Means konfigürasyonu
CLUSTER_K_VALUES  = [8, 12]
CLUSTER_TE_FOLDS  = 3
TE_SMOOTH_ALPHA   = 10

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# [NEW-SF] Split Freeze dosyası
FOLD_INDEX_PATH = OUTPUT_DIR / "fold_indices_v12.json"

# [NEW-V12] Guardrail sabitleri
MEDIUM_F1_DROP_MAX  = 0.015   # Medium F1 max düşüş
MEDIUM_REC_DROP_MAX = 0.020   # Medium Recall max düşüş
PRED_HIGH_MULT_MAX  = 1.25    # baseline_pred_high * bu katsayı = tavan
WF1_BAND            = 0.001   # Optuna band seçimi toleransı
HREC_TARGET         = 0.72    # Hedef High Recall
# [P0-B] ALPHA_ACCEPT_WF1 artık dinamik — çalışma zamanında baseline'a bağlanır.
# Faz-0 başlamadan önce: ALPHA_ACCEPT_WF1 = baseline_wf1 - ALPHA_WF1_DELTA
# Hard-coded 0.8687 (V11 leakage dünyası) kaldırıldı.
ALPHA_WF1_DELTA     = 0.001   # baseline'dan ne kadar aşağı kabul edilir
ALPHA_ACCEPT_WF1    = None    # [P0-B] Runtime'da set edilecek — başlangıçta None
ALPHA_ACCEPT_HREC   = 0.68    # α-sweep HRec alt eşiği (sabit)
GATE_FPR_MAX        = 0.05    # Gated rescue FPR üst sınırı
GATE_TPR_MIN        = 0.30    # Gated rescue TPR alt sınırı
EARLY_STOP_PATIENCE = 20      # Çift koşul early-stop penceresi

np.random.seed(SEED)


# =============================================================================
# BÖLÜM 1 — ZindiFeatureEngineer  ← V7 ile ÖZDEŞ, DOKUNULMADI
# =============================================================================

class ZindiFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Leakage-free Feature Engineering. fit() → sadece X_trn.
    16 FE özelliği üretir; ZindiClusterEngineer bu çıktıyı girdi olarak alır.
    """

    _ACCESS_COLS = [
        "has_mobile_money", "has_credit_card", "has_debit_card",
        "has_loan_account",  "has_internet_banking",
    ]
    _INSURANCE_COLS = [
        "motor_vehicle_insurance", "medical_insurance",
        "funeral_insurance",       "has_insurance",
    ]
    _RISK_COLS = [
        "current_problem_cash_flow",
        "problem_sourcing_money",
        "uses_informal_lender",
    ]
    _GROWTH_COLS = [
        "attitude_more_successful_next_year",
        "motivation_make_more_money",
        "attitude_satisfied_with_achievement",
    ]
    _BARRIER_COLS = [
        "perception_cannot_afford_insurance",
        "perception_insurance_doesnt_cover_losses",
        "perception_insurance_companies_dont_insure_businesses_like_yours",
    ]
    _DIGITAL_COLS = [
        "has_cellphone", "has_mobile_money",
        "has_internet_banking", "offers_credit_to_customers",
    ]

    @staticmethod
    def _safe_sum(df, cols):
        # pd.to_numeric: string değerleri ('Have now' vb.) önce NaN'a çevir, sonra 0'a
        return (
            df[cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(int)
            .sum(axis=1)
        )

    @staticmethod
    def _safe_int(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

    def fit(self, X: pd.DataFrame, y=None) -> "ZindiFeatureEngineer":
        stats = (
            X.groupby("country")["business_turnover"]
             .agg(["mean", "std"])
             .rename(columns={"mean": "ct_mean", "std": "ct_std"})
        )
        stats["ct_std"] = stats["ct_std"].fillna(1.0)
        self.country_stats_ = stats
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = X.copy()

        df["net_profit_margin"] = (
            (df["business_turnover"] - df["business_expenses"])
            / df["business_turnover"].clip(lower=1e-9)
        )
        df["expense_coverage"] = (
            df["personal_income"]
            / df["business_expenses"].clip(lower=1e-9)
        )
        df["financial_access_score"] = self._safe_sum(df, self._ACCESS_COLS)
        df["insurance_score"]        = self._safe_sum(df, self._INSURANCE_COLS)
        df["perception_action_gap"]  = (
            self._safe_int(df["perception_insurance_important"])
            - self._safe_int(df["has_insurance"])
        )

        # FE-06 [FIX-8] log1p stabilizasyonu
        df["business_age_total_months"] = (
            df["business_age_years"].fillna(0) * 12
            + df["business_age_months"].fillna(0)
        ).clip(lower=1)
        df["turnover_per_month"] = np.log1p(
            df["business_turnover"].clip(lower=0)
            / df["business_age_total_months"]
        )

        df["cashflow_risk_score"]   = self._safe_sum(df, self._RISK_COLS)
        df["growth_mindset_score"]  = self._safe_sum(df, self._GROWTH_COLS)
        df["informal_dependency"]   = (
            self._safe_int(df["uses_friends_family_savings"])
            + self._safe_int(df["uses_informal_lender"])
        )

        df = df.merge(
            self.country_stats_.reset_index(), on="country", how="left"
        )
        df["ct_std"] = df["ct_std"].fillna(1.0)
        df["turnover_z_country"] = (
            (df["business_turnover"] - df["ct_mean"])
            / df["ct_std"].clip(lower=1e-9)
        )
        df.drop(columns=["ct_mean", "ct_std"], inplace=True)

        df["insurance_barrier_score"] = self._safe_sum(df, self._BARRIER_COLS)
        df["digital_maturity_score"]  = self._safe_sum(df, self._DIGITAL_COLS)
        df["age_stability_interact"]  = (
            df["owner_age"]
            * self._safe_int(df["attitude_stable_business_environment"])
        )
        df["formality_score"] = (
            self._safe_int(df["keeps_financial_records"])
            * self._safe_int(df["compliance_income_tax"])
        )
        df["fully_informal"] = (
            (self._safe_int(df["keeps_financial_records"]) == 0)
            & (self._safe_int(df["compliance_income_tax"])  == 0)
        ).astype(int)
        df["vulnerability_index"] = (
            df["cashflow_risk_score"]
            + df["insurance_barrier_score"]
            + df["informal_dependency"]
            - df["financial_access_score"]
        )
        df["risk_coverage_mismatch"] = (
            self._safe_int(df["future_risk_theft_stock"])
            * (1 - self._safe_int(df["has_insurance"]))
        )
        return df


# =============================================================================
# BÖLÜM 1.5 — ZindiClusterEngineer  [NEW-8]
# =============================================================================

class ZindiClusterEngineer(BaseEstimator, TransformerMixin):
    """
    K-Means Arketip Kümeleme + Çok Sınıflı K-Fold Target Encoding.

    NEDEN İKİ ADIM BİRLİKTE?
    ──────────────────────────
    K-Means tek başına: cluster_id kategorik bir özellik — model bunu
    öğrenebilir ama hedef değişkenle ilişkisi yavaş öğrenilir.

    K-Fold TE tek başına: country/owner_sex gibi düşük kardinaliteli kolonlarda
    marjinal kazanç — zaten 4 ve 2 benzersiz değer var.

    Birlikte: K-Means kardinaliteyi 8-12'ye çıkarır; TE her cluster'ın
    "tarihsel FHI dağılımını" modele doğrudan verir. Ağaç modelleri bu
    özelliği split noktası olarak kullanmak için bölme yapmak zorunda kalmaz.

    LEAKAGE MİMARİSİ:
    ─────────────────
    fit(X_trn, y_trn):
      • StandardScaler → SADECE X_trn üzerinde fit
      • KMeans          → SADECE X_trn üzerinde fit
      • K-Fold TE       → SADECE X_trn / y_trn üzerinde iç CV ile hesaplanır
        ∘ İç 3-fold: her fold'da diğer iki fold'un istatistiklerini kullanır
        ∘ Laplace smoothing: küçük cluster'ları global ortalamayla karıştırır
        ∘ Son mapping: tüm X_trn üzerinden (val/test transform için)

    transform(X_val veya X_test):
      • Scaler ve KMeans: sadece predict/transform → hiçbir istatistik öğrenmez
      • TE: fit sırasında öğrenilen mapping → doğrudan lookup

    ÜRETILEN ÖZELLİKLER (k başına 5, toplam 10):
      cluster_id_k{k}      → KOBİ arketip etiketi [kategorik]
      cluster_dist_k{k}    → arketip merkezine uzaklık [sürekli]
      cluster_te_low_k{k}  → P(Low  | arketip) [K-Fold TE]
      cluster_te_mid_k{k}  → P(Mid  | arketip) [K-Fold TE]
      cluster_te_high_k{k} → P(High | arketip) [K-Fold TE]
    """

    # Kümeleme için yüksek bilgi içerikli sürekli/yarı-sürekli özellikler
    # Binary (0/1) kolonlar uzaklık metriğini bozar; özel liste daha güvenli
    _CLUSTER_FEATURES = [
        "net_profit_margin",
        "expense_coverage",
        "turnover_per_month",          # log1p ölçeğinde
        "turnover_z_country",
        "business_turnover",
        "personal_income",
        "owner_age",
        "business_age_total_months",
        "financial_access_score",      # 0-5; az kategorili ama faydalı
        "insurance_score",             # 0-4
        "cashflow_risk_score",         # 0-3
        "growth_mindset_score",        # 0-3
        "vulnerability_index",         # sürekli bileşik
        "digital_maturity_score",      # 0-4
        "age_stability_interact",      # çarpım; yüksek varyans
        "insurance_barrier_score",     # 0-3
    ]

    def __init__(
        self,
        k_values    : list = CLUSTER_K_VALUES,
        n_inner_folds: int = CLUSTER_TE_FOLDS,
        smooth_alpha : int = TE_SMOOTH_ALPHA,
    ):
        self.k_values     = k_values
        self.n_inner_folds = n_inner_folds
        self.smooth_alpha  = smooth_alpha

    def _active_cols(self, X: pd.DataFrame) -> list:
        """Veri setinde gerçekten bulunan kümeleme kolonlarını döndürür."""
        return [c for c in self._CLUSTER_FEATURES if c in X.columns]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ZindiClusterEngineer":
        """
        Scaler, KMeans ve K-Fold TE mapping'lerini X_trn / y_trn üzerinde öğren.
        y burada int dizisi (0=Low, 1=Med, 2=High) olmalıdır.
        """
        cols = self._active_cols(X)
        self.cols_      = cols
        self.n_classes_ = len(np.unique(y))
        self.global_class_freq_ = np.array(
            [np.mean(y == c) for c in range(self.n_classes_)],
            dtype=np.float64
        )

        # ── 1. StandardScaler — sadece train ─────────────────────────────────
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(
            X[cols].fillna(0).astype(float)
        )

        # ── 2. KMeans modelleri — sadece train ───────────────────────────────
        self.kmeans_ = {}
        for k in self.k_values:
            km = KMeans(
                n_clusters = k,
                random_state = SEED,
                n_init     = 15,       # 15 başlangıç → daha kararlı merkez
                max_iter   = 300,
            )
            km.fit(X_scaled)
            self.kmeans_[k] = km

        # ── 3. K-Fold Target Encoding — sadece train ─────────────────────────
        # Her k için iç CV ile OOF TE hesaplanır; final mapping tüm train'den.
        self.te_maps_ = {}   # {k: {cluster_id: [p_low, p_mid, p_high]}}
        self.oof_te_  = {}   # [FIX-TE] {k: np.array(n_train, n_classes)} train için sızıntısız TE
        y_arr = np.asarray(y, dtype=int)

        for k in self.k_values:
            cluster_labels = self.kmeans_[k].predict(X_scaled)

            # ── 3a. İç K-Fold ile OOF TE (train fold'u için sızıntısız) ─────
            inner_skf = StratifiedKFold(
                n_splits = self.n_inner_folds,
                shuffle  = True,
                random_state = SEED + 7,
            )
            oof_te = np.zeros((len(X), self.n_classes_), dtype=np.float64)

            for tr_idx, va_idx in inner_skf.split(X_scaled, y_arr):
                tr_c, va_c = cluster_labels[tr_idx], cluster_labels[va_idx]
                tr_y = y_arr[tr_idx]
                fold_map = self._build_te_map(tr_c, tr_y, k)
                for i, cid in enumerate(va_c):
                    oof_te[va_idx[i]] = fold_map.get(cid, self.global_class_freq_)

            # [FIX-TE] OOF TE'yi kaydet — transform(is_train=True) bunu kullanacak
            self.oof_te_[k] = oof_te

            # ── 3b. Final mapping: tüm train → val/test transform için ───────
            final_map = self._build_te_map(cluster_labels, y_arr, k)
            self.te_maps_[k] = final_map

        return self

    def _build_te_map(
        self,
        labels : np.ndarray,
        y      : np.ndarray,
        k      : int,
    ) -> dict:
        """
        Cluster → [p_low, p_mid, p_high] Laplace-smoothed mapping oluştur.

        Smoothing formülü (her sınıf c için):
          p_c = (n_c + α × global_freq_c) / (n_cluster + α)

        α = TE_SMOOTH_ALPHA (varsayılan 10).
        Küçük cluster'lar (n < α) global dağılıma yaklaşır → overfit önlenir.
        Büyük cluster'lar (n >> α) kendi gözlemlenen dağılımlarını korur.
        """
        te_map = {}
        unique_ids = np.unique(labels)
        alpha = self.smooth_alpha

        for cid in unique_ids:
            mask = labels == cid
            n    = mask.sum()
            probs = np.zeros(self.n_classes_, dtype=np.float64)
            for c in range(self.n_classes_):
                n_c    = (y[mask] == c).sum()
                # Laplace / additive smoothing
                probs[c] = (n_c + alpha * self.global_class_freq_[c]) / (n + alpha)
            te_map[cid] = probs   # normalize: toplamı ≈ 1 (numerik hassasiyet)

        return te_map

    def transform(self, X: pd.DataFrame, y=None, is_train: bool = False) -> pd.DataFrame:
        """
        Öğrenilen cluster ve TE bilgilerini X'e ekle.

        [FIX-TE] is_train=True  → self.oof_te_[k] kullan  (train seti, sızıntısız)
                 is_train=False → self.te_maps_[k] kullan  (val / test seti)

        UYARI: is_train=True sadece fit() çağrısındaki X_trn ile çağrılmalıdır.
        Farklı boyutlu X ile çağrılırsa IndexError üretir — bilerek böyle tasarlandı.
        """
        df  = X.copy()
        cols = [c for c in self.cols_ if c in df.columns]
        X_scaled = self.scaler_.transform(
            df[cols].fillna(0).astype(float)
        )

        for k in self.k_values:
            km     = self.kmeans_[k]
            labels = km.predict(X_scaled)
            dists  = km.transform(X_scaled)

            df[f"cluster_id_k{k}"]   = labels.astype(int)
            df[f"cluster_dist_k{k}"] = dists[np.arange(len(labels)), labels].astype(np.float32)

            # [FIX-TE] TE kaynağı seçimi
            if is_train:
                # Train için: fit() sırasında hesaplanan sızıntısız OOF TE
                # Her satır kendi fold'undaki istatistiklerden üretildi
                te_proba = self.oof_te_[k].astype(np.float32)
            else:
                # Val/Test için: tüm train üzerinden öğrenilen final mapping
                te_proba = np.array([
                    self.te_maps_[k].get(cid, self.global_class_freq_)
                    for cid in labels
                ], dtype=np.float32)

            df[f"cluster_te_low_k{k}"]  = te_proba[:, 0]
            df[f"cluster_te_mid_k{k}"]  = te_proba[:, 1]
            df[f"cluster_te_high_k{k}"] = te_proba[:, 2]

        return df

    @property
    def cluster_id_cols(self) -> list:
        """CatBoost / LGBM için kategorik olarak işaretlenecek kolonlar."""
        return [f"cluster_id_k{k}" for k in self.k_values]


# =============================================================================
# BÖLÜM 2 — VERİ KATMANI  ← V7 ile ÖZDEŞ
# =============================================================================

def nuke_strings(df: pd.DataFrame, cat_cols: list = None) -> pd.DataFrame:
    """
    [FIX-STR-NUCLEAR] dtype kontrolü OLMADAN her kolonu temizler.
    cat_cols listesindekiler string olarak kalır (model cat olarak işler).
    Diğer TÜM kolonlar: Yes/No/Have now/... → 0/1 → float.
    
    Bu fonksiyon dtype'a BAKMAZ — her kolonu tek tek dener.
    Hiçbir string değer CatBoost/LGBM/XGBoost'a ulaşamaz.
    """
    _MAP = {
        "yes": 1.0, "no": 0.0,
        "have now": 1.0, "do not have": 0.0,
        "true": 1.0, "false": 0.0,
        "male": 1.0, "female": 0.0,
        "1": 1.0, "0": 0.0,
        "nan": 0.0, "none": 0.0, "": 0.0,
    }
    _SAFE = set(cat_cols) if cat_cols else set()
    df = df.copy()

    for col in df.columns:
        if col in _SAFE:
            continue
        s = df[col]
        # Zaten tam sayısal mı? Hızlı yol.
        if pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s):
            df[col] = s.fillna(0).astype(float)
            continue
        # String/object/mixed → temizle
        lowered = s.astype(str).str.strip().str.lower()
        mapped  = lowered.map(_MAP)
        # Haritada bulunmayanları sayısala çevir
        unmatched_mask = mapped.isna()
        if unmatched_mask.any():
            numeric_fallback = pd.to_numeric(s[unmatched_mask], errors="coerce").fillna(0)
            mapped[unmatched_mask] = numeric_fallback
        df[col] = mapped.astype(float)

    return df



def cb_fix_cats(df: pd.DataFrame, cat_cols: list, missing_token: str = "__MISSING__") -> pd.DataFrame:
    """
    CatBoost categorical columns must be str/int and cannot contain NaN.
    This touches ONLY cat_cols and leaves other columns unchanged.
    """
    df = df.copy()
    if not cat_cols:
        return df
    for c in cat_cols:
        if c not in df.columns:
            continue
        s = df[c].astype("object")
        s = s.where(~pd.isna(s), missing_token)
        df[c] = s.astype(str)
    return df


def cb_assert_no_nan_in_cats(df: pd.DataFrame, cat_cols: list, label: str = "") -> None:
    """Hard fail if NaN remains in cat cols (prevents CatBoostError)."""
    if not cat_cols:
        return
    cols = [c for c in cat_cols if c in df.columns]
    if not cols:
        return
    if df[cols].isna().any().any():
        bad = {c: int(df[c].isna().sum()) for c in cols if df[c].isna().any()}
        tag = f" [{label}]" if label else ""
        raise ValueError(f"NaN remains in CatBoost cat cols{tag}: {bad}")
def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Kaynak temizliği — veri okununca çağrılır."""
    return nuke_strings(df, cat_cols=CAT_FEATURES)


def encode_string_columns(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Model öncesi son savunma — nuke_strings'e delegate eder."""
    return nuke_strings(df, cat_cols=cat_cols)




def read_data(train_path: Path, test_path: Path) -> tuple:
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    y        = encode_target(train_df[TARGET_COL])
    test_ids = test_df[ID_COL].values
    X_raw    = train_df.drop(columns=[ID_COL, TARGET_COL], errors="ignore").reset_index(drop=True)
    X_test   = test_df.drop(columns=[ID_COL], errors="ignore").reset_index(drop=True)

    # [FIX-STR] Ham string kolonları sayısala çevir — tüm pipeline için tek nokta
    X_raw  = preprocess_raw(X_raw)
    X_test = preprocess_raw(X_test)

    print(f"\n  ✓ Veri okundu")
    print(f"    Train (ham) : {X_raw.shape}   Test (ham) : {X_test.shape}")
    for label, code in TARGET_MAPPING.items():
        n = (y == code).sum()
        print(f"    {label:8s} ({code}) → {n:5d} örnek  ({n/len(y)*100:.1f}%)")

    # String kalan kolon var mı? Uyar.
    str_cols = [c for c in X_raw.columns
                if c not in CAT_FEATURES and
                (X_raw[c].dtype == object or str(X_raw[c].dtype) == "string")]
    if str_cols:
        print(f"  ⚠️  Hâlâ string kalan kolonlar (cat dışı): {str_cols}")
    return X_raw, y, X_test, test_ids


def encode_target(series: pd.Series) -> np.ndarray:
    y = series.map(TARGET_MAPPING).values
    if np.isnan(y.astype(float)).any():
        unknown = set(series.unique()) - set(TARGET_MAPPING.keys())
        raise ValueError(f"Bilinmeyen sınıf: {unknown}")
    return y.astype(int)


def align_columns(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """[FIX-7] Mutation-safe reindex."""
    return target.reindex(columns=source.columns, fill_value=0)



# =============================================================================
# BÖLÜM 3 — HİPERPARAMETRELER  ← V7 ile ÖZDEŞ
# =============================================================================

def get_catboost_params() -> dict:
    # [P0-C] CatBoost OD erken fren fix
    # Sebep: auto_class_weights="Balanced" + lr=0.05 + ES=80 kombinasyonu
    # → Balanced, High'a ~6.8x gradient cezası veriyor.
    # → Train ağırlıklı, val gerçek dağılımda → OD çok erken tetikleniyor.
    # → Fold'larda bestIteration=15-62 → CB hiçbir şey öğrenemiyor.
    # Çözüm: explicit class_weights + lr↓ + iter↑ + early_stop↑
    return dict(
        iterations=3000, learning_rate=0.02, depth=7, l2_leaf_reg=3.0,
        loss_function="MultiClass", eval_metric="TotalF1:average=Weighted",
        auto_class_weights=None,          # [P0-C] Balanced KALDIRILDI — OD asimetrisini tetikliyordu
        class_weights={0: 1, 1: 2, 2: 12}, # [P0-C] Explicit: Low=1x, Medium=2x, High=12x
        random_seed=SEED,
        early_stopping_rounds=150,         # [P0-C] 80 → 150: modele daha fazla sabır
        verbose=200, thread_count=-1,
    )

def get_lgbm_params() -> dict:
    return dict(
        n_estimators=1500, learning_rate=0.05, num_leaves=63,
        max_depth=-1, min_child_samples=20, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multiclass", num_class=N_CLASSES,
        class_weight="balanced", metric="multi_logloss",
        random_state=SEED, n_jobs=-1, verbose=-1,
    )

def get_xgb_params() -> dict:
    return dict(
        n_estimators=1200, learning_rate=0.05, max_depth=6,
        min_child_weight=5, subsample=0.8,
        colsample_bytree=0.8, colsample_bylevel=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", num_class=N_CLASSES,
        eval_metric="mlogloss", tree_method="hist",
        random_state=SEED, n_jobs=-1, verbosity=0,
    )


# =============================================================================
# BÖLÜM 4 — MODEL VERİ HAZIRLIK YARDIMCILARI  [NEW-9: cluster_id dinamik ekleme]
# =============================================================================

def prepare_lgbm_data(
    X_trn: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    extra_cat_cols: list,   # [NEW-9] cluster_id kolonları
) -> tuple:
    """
    LightGBM için kategorik kolonları (orijinal + cluster_id) 'category' dtype'a çevirir.
    Val/test kategorileri X_trn'den öğrenilir → unseen kategori NaN üretmez.
    """
    X_tr, X_va, X_te = X_trn.copy(), X_val.copy(), X_test.copy()
    all_cats = [c for c in (CAT_FEATURES + extra_cat_cols) if c in X_tr.columns]

    # [FIX-STR] Non-cat string kolonları sayısala çevir
    X_tr = encode_string_columns(X_tr, all_cats)
    X_va = encode_string_columns(X_va, all_cats)
    X_te = encode_string_columns(X_te, all_cats)

    for col in all_cats:
        X_tr[col] = X_tr[col].astype("category")
        cats = X_tr[col].cat.categories
        X_va[col] = X_va[col].astype(pd.CategoricalDtype(categories=cats))
        X_te[col] = X_te[col].astype(pd.CategoricalDtype(categories=cats))

    return X_tr, X_va, X_te, all_cats


def prepare_xgb_data(
    X_trn: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_trn: np.ndarray,
    extra_cat_cols: list,   # [NEW-9] cluster_id kolonları
) -> tuple:
    """
    XGBoost için string/kategorik kolonları OrdinalEncoder ile int'e çevirir.
    Fit SADECE X_trn üzerinde → leakage yok.
    sample_weight fold'a özel hesaplanır → leakage yok.
    """
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    X_tr, X_va, X_te = X_trn.copy(), X_val.copy(), X_test.copy()
    all_cats = [c for c in (CAT_FEATURES + extra_cat_cols) if c in X_tr.columns]

    # [FIX-STR] Non-cat string kolonları sayısala çevir
    X_tr = encode_string_columns(X_tr, all_cats)
    X_va = encode_string_columns(X_va, all_cats)
    X_te = encode_string_columns(X_te, all_cats)

    if all_cats:
        X_tr[all_cats] = enc.fit_transform(X_tr[all_cats].astype(str))
        X_va[all_cats] = enc.transform(X_va[all_cats].astype(str))
        X_te[all_cats] = enc.transform(X_te[all_cats].astype(str))

    sw = compute_sample_weight(class_weight="balanced", y=y_trn)
    return X_tr, X_va, X_te, sw


# =============================================================================
# BÖLÜM 5 — EĞİTİM DÖNGÜSÜ (3 Model + Cluster Engineering)
# =============================================================================

def get_or_create_folds(X_raw: pd.DataFrame, y: np.ndarray) -> list:
    """
    [NEW-SF] Split Freeze: fold indekslerini tek sefer üret ve JSON'a kaydet.
    Sonraki çalışmalarda aynı dosyadan yükle → tüm fazlar aynı foldlarla koşar.
    """
    if FOLD_INDEX_PATH.exists():
        with open(FOLD_INDEX_PATH) as f:
            fold_data = json.load(f)
        folds = [(np.array(fd["train"]), np.array(fd["val"])) for fd in fold_data]
        print(f"  ✓ [SPLIT FREEZE] Fold indeksleri yüklendi: {FOLD_INDEX_PATH}")
    else:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        folds = list(skf.split(X_raw, y))
        fold_data = [{"train": tr.tolist(), "val": va.tolist()} for tr, va in folds]
        with open(FOLD_INDEX_PATH, "w") as f:
            json.dump(fold_data, f)
        print(f"  ✓ [SPLIT FREEZE] Fold indeksleri oluşturuldu ve kaydedildi: {FOLD_INDEX_PATH}")

    # Assert: overlap yok
    for i, (tr, va) in enumerate(folds):
        assert len(set(tr) & set(va)) == 0, f"HATA: Fold {i+1}'de train/val overlap var!"
    print(f"  ✓ [SPLIT FREEZE] Overlap kontrolü geçti.")
    return folds


def train_oof_triple(
    X_raw  : pd.DataFrame,
    y      : np.ndarray,
    X_test : pd.DataFrame,
) -> tuple:
    """
    Leakage-free 5-Fold OOF: ZindiFE → ZindiCluster → CB + LGBM + XGB.

    Fold Veri Akışı:
    ────────────────
    X_raw (ham)
      └─► ZindiFeatureEngineer.fit(X_trn_raw)
              ├─► X_trn_fe = fit_transform(X_trn_raw)    [train]
              ├─► X_val_fe = transform(X_val_raw)         [val]
              └─► X_test_fe = transform(X_test)           [test]

      └─► ZindiClusterEngineer.fit(X_trn_fe, y_trn)      [NEW-8]
              ├─► X_trn = transform(X_trn_fe)  + 10 yeni özellik
              ├─► X_val = transform(X_val_fe)  + 10 yeni özellik
              └─► X_test_cl = transform(X_test_fe) + 10 yeni özellik

      └─► CB / LGBM / XGB  (cluster_id kolonları dinamik olarak cat listesine eklendi)
    """
    n_train = len(X_raw)
    n_test  = len(X_test)

    cb_oof   = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    lgbm_oof = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    xgb_oof  = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    cb_test   = np.zeros((n_test, N_CLASSES), dtype=np.float64)
    lgbm_test = np.zeros((n_test, N_CLASSES), dtype=np.float64)
    xgb_test  = np.zeros((n_test, N_CLASSES), dtype=np.float64)

    cb_scores, lgbm_scores, xgb_scores = [], [], []

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = get_or_create_folds(X_raw, y)

    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):

        print(f"\n{'═' * 72}")
        print(f"  FOLD {fold_idx} / {N_SPLITS}  "
              f"(train={len(train_idx):,}  val={len(val_idx):,})")
        print(f"{'═' * 72}")

        X_trn_raw = X_raw.iloc[train_idx].reset_index(drop=True)
        X_val_raw = X_raw.iloc[val_idx].reset_index(drop=True)
        y_trn     = y[train_idx]
        y_val     = y[val_idx]

        # ── [FIX-4] Leakage-free Feature Engineering ─────────────────────────
        fe         = ZindiFeatureEngineer()
        X_trn_fe   = fe.fit_transform(X_trn_raw)
        X_val_fe   = align_columns(X_trn_fe, fe.transform(X_val_raw))
        X_test_fe  = align_columns(X_trn_fe, fe.transform(X_test.copy()))

        # ── [NEW-8] Leakage-free Cluster Engineering ──────────────────────────
        # fit: SADECE X_trn_fe ve y_trn üzerinde — val/test görmez
        print(f"  ▶ [0/3] K-Means + K-Fold TE hesaplanıyor "
              f"(k={CLUSTER_K_VALUES}, inner_folds={CLUSTER_TE_FOLDS})...")
        ce = ZindiClusterEngineer()
        ce.fit(X_trn_fe, y_trn)

        # [FIX-TE] transform: train için is_train=True (oof_te), val/test için False (final_map)
        X_trn      = ce.transform(X_trn_fe, is_train=True)
        X_val      = align_columns(X_trn, ce.transform(X_val_fe, is_train=False))
        X_test_cl  = align_columns(X_trn, ce.transform(X_test_fe, is_train=False))

        cluster_id_cols = ce.cluster_id_cols   # ["cluster_id_k8", "cluster_id_k12"]
        print(f"  ✓ Cluster özellikleri eklendi: "
              f"{[c for c in X_trn.columns if 'cluster' in c]}")

        # ─────────────────────────────────────────────────────────────────────
        #  MODEL A — CatBoost
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n  ▶ [1/3] CatBoost eğitimi...")

        # [NEW-9] cluster_id kolonlarını CatBoost cat listesine ekle
        active_cats_cb = (
            [c for c in CAT_FEATURES if c in X_trn.columns]
            + [c for c in cluster_id_cols if c in X_trn.columns]
        )
        # CatBoost string kategorik ister; cluster_id'yi string'e çevir
        X_trn_cb  = X_trn.copy()
        X_val_cb  = X_val.copy()
        X_test_cb = X_test_cl.copy()
        for col in cluster_id_cols:
            if col in X_trn_cb.columns:
                X_trn_cb[col]  = X_trn_cb[col].astype(str)
                X_val_cb[col]  = X_val_cb[col].astype(str)
                X_test_cb[col] = X_test_cb[col].astype(str)

        # [FIX-STR] Non-cat kolonlardaki 'Yes'/'No' vb. stringleri sayısala çevir
        X_trn_cb  = encode_string_columns(X_trn_cb,  active_cats_cb)
        X_val_cb  = encode_string_columns(X_val_cb,  active_cats_cb)
        X_test_cb = encode_string_columns(X_test_cb, active_cats_cb)

        # [FIX-STR-FINAL] Pool öncesi mutlak garanti: sıfır string kalmasın
        # nuke_strings dtype'a bakmadan her kolonu zorunlu float'a çevirir
        X_trn_cb  = nuke_strings(X_trn_cb,  cat_cols=active_cats_cb)
        X_val_cb  = nuke_strings(X_val_cb,  cat_cols=active_cats_cb)
        X_test_cb = nuke_strings(X_test_cb, cat_cols=active_cats_cb)
        # [FIX-CB-CAT-NA] CatBoost cat cols cannot contain NaN; enforce str + fill missing
        X_trn_cb  = cb_fix_cats(X_trn_cb,  active_cats_cb)
        X_val_cb  = cb_fix_cats(X_val_cb,  active_cats_cb)
        X_test_cb = cb_fix_cats(X_test_cb, active_cats_cb)
        cb_assert_no_nan_in_cats(X_trn_cb, active_cats_cb, label='TRAIN')
        cb_assert_no_nan_in_cats(X_val_cb, active_cats_cb, label='VAL')
        cb_assert_no_nan_in_cats(X_test_cb, active_cats_cb, label='TEST')

        train_pool = Pool(data=X_trn_cb, label=y_trn, cat_features=active_cats_cb)
        val_pool   = Pool(data=X_val_cb, label=y_val, cat_features=active_cats_cb)
        test_pool  = Pool(data=X_test_cb,             cat_features=active_cats_cb)

        cb_model = CatBoostClassifier(**get_catboost_params())
        cb_model.fit(train_pool, eval_set=val_pool, plot=False)

        fold_cb_oof      = cb_model.predict_proba(val_pool)
        cb_oof[val_idx]  = fold_cb_oof
        cb_test         += cb_model.predict_proba(test_pool) / N_SPLITS

        fold_cb_f1 = f1_score(y_val, np.argmax(fold_cb_oof, 1), average="weighted")
        cb_scores.append(fold_cb_f1)
        print(f"  ✓ CatBoost   Fold {fold_idx}  Weighted-F1 : {fold_cb_f1:.5f}")

        # ─────────────────────────────────────────────────────────────────────
        #  MODEL B — LightGBM
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n  ▶ [2/3] LightGBM eğitimi...")

        # [NEW-9] cluster_id dinamik olarak cat listesine eklendi
        X_tr_lgbm, X_va_lgbm, X_te_lgbm, lgbm_cats = prepare_lgbm_data(
            X_trn, X_val, X_test_cl, extra_cat_cols=cluster_id_cols
        )
        lgbm_model = LGBMClassifier(**get_lgbm_params())
        lgbm_model.fit(
            X_tr_lgbm, y_trn,
            eval_set            = [(X_va_lgbm, y_val)],
            callbacks           = [
                lgb.early_stopping(stopping_rounds=80, verbose=False),
                lgb.log_evaluation(period=200),
            ],
            categorical_feature = lgbm_cats,
        )
        fold_lgbm_oof      = lgbm_model.predict_proba(X_va_lgbm)
        lgbm_oof[val_idx]  = fold_lgbm_oof
        lgbm_test         += lgbm_model.predict_proba(X_te_lgbm) / N_SPLITS

        fold_lgbm_f1 = f1_score(y_val, np.argmax(fold_lgbm_oof, 1), average="weighted")
        lgbm_scores.append(fold_lgbm_f1)
        print(f"  ✓ LightGBM   Fold {fold_idx}  Weighted-F1 : {fold_lgbm_f1:.5f}")

        # ─────────────────────────────────────────────────────────────────────
        #  MODEL C — XGBoost
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n  ▶ [3/3] XGBoost eğitimi...")

        # [NEW-9] cluster_id OrdinalEncoder ile int'e çevrildi
        X_tr_xgb, X_va_xgb, X_te_xgb, sw = prepare_xgb_data(
            X_trn, X_val, X_test_cl, y_trn, extra_cat_cols=cluster_id_cols
        )
        xgb_model = XGBClassifier(**get_xgb_params())
        xgb_model.fit(
            X_tr_xgb, y_trn,
            sample_weight         = sw,
            eval_set              = [(X_va_xgb, y_val)],
            early_stopping_rounds = 80,
            verbose               = 200,
        )
        fold_xgb_oof      = xgb_model.predict_proba(X_va_xgb)
        xgb_oof[val_idx]  = fold_xgb_oof
        xgb_test         += xgb_model.predict_proba(X_te_xgb) / N_SPLITS

        fold_xgb_f1 = f1_score(y_val, np.argmax(fold_xgb_oof, 1), average="weighted")
        xgb_scores.append(fold_xgb_f1)
        print(f"  ✓ XGBoost    Fold {fold_idx}  Weighted-F1 : {fold_xgb_f1:.5f}")

        # ── Fold Özeti ────────────────────────────────────────────────────────
        fold_eq_proba = (fold_cb_oof + fold_lgbm_oof + fold_xgb_oof) / 3
        fold_eq_f1    = f1_score(y_val, np.argmax(fold_eq_proba, 1), average="weighted")
        winner        = max(fold_cb_f1, fold_lgbm_f1, fold_xgb_f1, fold_eq_f1)

        print(f"\n  {'─'*64}")
        print(f"  FOLD {fold_idx} ÖZET")
        print(f"  {'─'*64}")
        print(f"  CatBoost       : {fold_cb_f1:.5f} {'◄ BEST' if fold_cb_f1 == winner else ''}")
        print(f"  LightGBM       : {fold_lgbm_f1:.5f} {'◄ BEST' if fold_lgbm_f1 == winner else ''}")
        print(f"  XGBoost        : {fold_xgb_f1:.5f} {'◄ BEST' if fold_xgb_f1 == winner else ''}")
        print(f"  1/3 Eşit Blend : {fold_eq_f1:.5f} {'◄ BEST' if fold_eq_f1 == winner else ''}")

    return (cb_oof,   cb_test,
            lgbm_oof, lgbm_test,
            xgb_oof,  xgb_test,
            cb_scores, lgbm_scores, xgb_scores)


# =============================================================================
# BÖLÜM 6 — OPTUNA WEIGHTED BLEND  ← V7 ile ÖZDEŞ
# =============================================================================

def optuna_weight_search(
    cb_oof   : np.ndarray,
    lgbm_oof : np.ndarray,
    xgb_oof  : np.ndarray,
    y        : np.ndarray,
    n_trials : int = OPTUNA_TRIALS,
) -> tuple:
    """
    [FIX-OPT] V12: PRIMARY = Weighted-F1, seçim = best-0.001 bandından HRec max.
    Eski V9: Macro-F1 maximize → CB'yi susturuyordu.

    [NEW-ES2] Early-stop çift koşul:
    - Son PATIENCE trial'da WF1 band dışında VE
    - Son PATIENCE trial'da HRec trendi flat (max-min < 0.005)
    """

    def objective(trial: optuna.Trial) -> float:
        from sklearn.metrics import recall_score as _recall_score  # local import safety
        raw = np.array([
            trial.suggest_float("w_cb",   0.0, 1.0),
            trial.suggest_float("w_lgbm", 0.0, 1.0),
            trial.suggest_float("w_xgb",  0.0, 1.0),
        ])
        w = raw / (raw.sum() + 1e-12)
        blended = w[0] * cb_oof + w[1] * lgbm_oof + w[2] * xgb_oof
        preds   = np.argmax(blended, 1)

        wf1  = f1_score(y, preds, average="weighted")
        # [P0-A] METRIK BUG FIX: f1_score(labels=[2]) ≠ recall
        # Eski: hrec = f1_score(y, preds, labels=[2], average="macro")  ← High-F1'di!
        # Yeni: recall_score → gerçekten High sınıfının recall'u
        # Faz-0 HRec ve Optuna HRec artık aynı metriği ölçüyor.
        hrec = _recall_score(y, preds, labels=[2], average="macro")   # [P0-A] Gerçek High-Recall
        hf1  = f1_score(y, preds, labels=[2], average="macro")       # [P0-A] High-F1 ayrıca loglansın
        phc  = int(np.sum(preds == 2))

        trial.set_user_attr("high_recall",     hrec)   # [P0-A] Gerçek recall
        trial.set_user_attr("high_f1",         hf1)    # [P0-A] High-F1 (ek bilgi)
        trial.set_user_attr("pred_high_count", phc)
        return wf1   # PRIMARY

    def dual_early_stop(study: optuna.Study, trial: optuna.Trial) -> None:
        """[NEW-ES2] İki koşulun IKISI birden sağlanırsa dur."""
        if len(study.trials) < EARLY_STOP_PATIENCE:
            return
        recent = study.trials[-EARLY_STOP_PATIENCE:]
        best_wf1 = study.best_value

        band_broken = all(t.value < best_wf1 - WF1_BAND for t in recent)
        hrecs = [t.user_attrs.get("high_recall", 0.0) for t in recent]
        trend_flat  = (max(hrecs) - min(hrecs)) < 0.005

        if band_broken and trend_flat:
            print(f"  [EARLY STOP] Trial {trial.number}: band_broken={band_broken} AND trend_flat={trend_flat}")
            study.stop()

    def select_best(study: optuna.Study) -> optuna.Trial:
        """WF1 >= best-0.001 bandından HRec maksimum trial'ı seç."""
        best_wf1 = study.best_value
        band = [t for t in study.trials
                if t.value is not None and t.value >= best_wf1 - WF1_BAND]
        return max(band, key=lambda t: t.user_attrs.get("high_recall", 0.0))

    print(f"\n  ▶ Optuna arama başlıyor... ({n_trials} trial)")
    print(f"    [V12] PRIMARY=WF1, seçim=band({WF1_BAND}) içinden HRec max")
    study = optuna.create_study(
        direction = "maximize",
        sampler   = optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials,
                   callbacks=[dual_early_stop],
                   show_progress_bar=False)

    best_trial = select_best(study)
    best       = best_trial.params
    raw  = np.array([best["w_cb"], best["w_lgbm"], best["w_xgb"]])
    bw   = raw / raw.sum()

    # pred_high_count tavanı kontrolü
    phc_best = best_trial.user_attrs.get("pred_high_count", 0)
    info = {
        "w_cb"            : bw[0],
        "w_lgbm"          : bw[1],
        "w_xgb"           : bw[2],
        "weighted_f1"     : best_trial.value,
        "high_recall"     : best_trial.user_attrs.get("high_recall", 0.0),
        "pred_high_count" : phc_best,
        "total_trials"    : len(study.trials),
    }

    print(f"  ✓ Optuna tamamlandı ({len(study.trials)} trial)")
    print(f"    Seçim kriteri : WF1 >= {study.best_value:.5f} - {WF1_BAND} bandından HRec max")
    print(f"    Ağırlıklar   : CB={bw[0]:.4f}  LGBM={bw[1]:.4f}  XGB={bw[2]:.4f}")
    print(f"    WF1          : {best_trial.value:.5f}")
    print(f"    High Recall  : {best_trial.user_attrs.get('high_recall', 0):.5f}")
    print(f"    pred_high_count: {phc_best}")
    return bw, info


# =============================================================================
# BÖLÜM 6.5 — FAZ 0 TEST PAKETİ  [NEW-F0]
# =============================================================================

def check_cb_calibration(cb_oof: np.ndarray, y: np.ndarray) -> dict:
    """
    [P0-D] CatBoost proba kalibrasyon kontrolü.
    CB fix sonrası gated rescue için threshold 0.35 hâlâ geçerli mi?
    ECE (Expected Calibration Error) ve bias yönü raporlanır.

    Karar kuralı:
      ECE < 0.05 → CB proba güvenilir, threshold 0.35 geçerli
      ECE 0.05-0.10 → Hafif inflated/deflated, threshold dikkatli kullan
      ECE > 0.10 → CB proba kalibre değil, threshold sweep'i genişlet

    Returns: dict(ece, bias, threshold_suggestion)
    """
    from sklearn.calibration import calibration_curve
    cb_high_prob = cb_oof[:, 2]
    y_binary     = (y == 2).astype(int)

    # Bin sayısını veri boyutuna göre ayarla (High az → daha az bin)
    n_high = y_binary.sum()
    n_bins = min(10, max(3, n_high // 30))

    try:
        prob_true, prob_pred = calibration_curve(y_binary, cb_high_prob, n_bins=n_bins)
    except ValueError:
        return {"ece": None, "bias": "unknown", "threshold_suggestion": 0.35, "n_bins": n_bins}

    # ECE = Σ |bin_acc - bin_conf| × bin_weight
    bin_weights = np.histogram(cb_high_prob, bins=n_bins, range=(0,1))[0] / len(cb_high_prob)
    ece = float(np.sum(np.abs(prob_true - prob_pred) * bin_weights[:len(prob_true)]))

    # Bias yönü: prob_pred > prob_true → inflated (CB overconfident)
    bias_raw  = float(np.mean(prob_pred - prob_true))
    bias      = "inflated" if bias_raw > 0.02 else ("deflated" if bias_raw < -0.02 else "calibrated")

    # Threshold önerisi
    if bias == "inflated":
        threshold_suggestion = round(min(0.50, 0.35 + bias_raw), 2)
    elif bias == "deflated":
        threshold_suggestion = round(max(0.20, 0.35 + bias_raw), 2)
    else:
        threshold_suggestion = 0.35

    result = {
        "ece"                 : round(ece, 4),
        "bias"                : bias,
        "bias_raw"            : round(bias_raw, 4),
        "threshold_suggestion": threshold_suggestion,
        "n_bins"              : n_bins,
    }

    print(f"\n  [P0-D] CB PROBA KALİBRASYON KONTROLÜ")
    print(f"    ECE         : {ece:.4f}  {'✓ İYİ' if ece < 0.05 else ('⚠ ORTA' if ece < 0.10 else '✗ KÖTÜ')}")
    print(f"    Bias        : {bias}  (raw={bias_raw:+.4f})")
    print(f"    Öneri thresh: {threshold_suggestion}  (V12 default: 0.35)")
    if ece > 0.10:
        print(f"    ⚠ CB proba kalibre değil → gated sweep threshold aralığını genişlet")

    return result



def faz0_medium_baseline(
    blend_oof : np.ndarray,
    y         : np.ndarray,
) -> dict:
    """
    [NEW-F0] Medium kill-switch için baseline metrikleri hesapla.
    Bu fonksiyon bir kez çalıştırılır; dönen değerler sabit kalır.
    """
    from sklearn.metrics import precision_score, recall_score
    preds = np.argmax(blend_oof, axis=1)
    baseline = {
        "MediumF1_base"  : f1_score(y, preds, labels=[1], average="macro"),
        "MediumRec_base" : recall_score(y, preds, labels=[1], average="macro"),
        "MediumPre_base" : precision_score(y, preds, labels=[1], average="macro"),
        "pred_high_baseline" : int(np.sum(preds == 2)),
    }
    print(f"\n  [FAZ0][MEDIUM_BASELINE]")
    print(f"    MediumF1_base  = {baseline['MediumF1_base']:.5f}")
    print(f"    MediumRec_base = {baseline['MediumRec_base']:.5f}")
    print(f"    pred_high_baseline = {baseline['pred_high_baseline']}")
    print(f"    pred_high_tavan    = {int(baseline['pred_high_baseline'] * PRED_HIGH_MULT_MAX)}")
    return baseline


def faz0_check_guardrails(
    preds    : np.ndarray,
    y        : np.ndarray,
    baseline : dict,
    label    : str = "",
) -> dict:
    """
    [NEW-F0] Her deney sonrası guardrail kontrolü.
    Döner: {"wf1", "hrec", "hpre", "med_f1", "med_rec", "phc", "karar", "ihlaller"}
    """
    from sklearn.metrics import precision_score, recall_score
    wf1  = f1_score(y, preds, average="weighted")
    hrec = recall_score(y, preds, labels=[2], average="macro")
    hpre = precision_score(y, preds, labels=[2], average="macro")
    med_f1  = f1_score(y, preds, labels=[1], average="macro")
    med_rec = recall_score(y, preds, labels=[1], average="macro")
    phc     = int(np.sum(preds == 2))
    phc_tavan = int(baseline["pred_high_baseline"] * PRED_HIGH_MULT_MAX)

    ihlaller = []
    if med_f1  < baseline["MediumF1_base"]  - MEDIUM_F1_DROP_MAX:
        ihlaller.append(f"MediumF1 düştü ({med_f1:.4f} < {baseline['MediumF1_base']-MEDIUM_F1_DROP_MAX:.4f})")
    if med_rec < baseline["MediumRec_base"] - MEDIUM_REC_DROP_MAX:
        ihlaller.append(f"MediumRec düştü ({med_rec:.4f} < {baseline['MediumRec_base']-MEDIUM_REC_DROP_MAX:.4f})")
    if phc > phc_tavan:
        ihlaller.append(f"pred_high_count aştı ({phc} > {phc_tavan})")

    karar = "KALDI" if ihlaller else "GEÇTİ"
    tag   = f"[{label}] " if label else ""
    print(f"\n  {tag}WF1={wf1:.5f} | HRec={hrec:.5f} | HPre={hpre:.5f} | "
          f"phc={phc} | MedF1={med_f1:.5f} | karar={karar}")
    for ih in ihlaller:
        print(f"    ⚠️  İHLAL: {ih}")

    return {"wf1": wf1, "hrec": hrec, "hpre": hpre,
            "med_f1": med_f1, "med_rec": med_rec,
            "phc": phc, "karar": karar, "ihlaller": ihlaller}


def faz0_recoverable_high(
    cb_oof    : np.ndarray,
    blend_oof : np.ndarray,
    y         : np.ndarray,
) -> dict:
    """
    [NEW-F0] V12 YÖN BELİRLEYİCİ TEST.
    Blend'in kaçırdığı 154 High içinde CB kaçını doğru görüyor?

    Karar:
      >= 80  → Gated rescue önce
      40–79  → α-sweep önce
      < 40   → Sadece Optuna (gated rescue uygulanmaz)
    """
    blend_preds = np.argmax(blend_oof, axis=1)
    missed_mask = (y == 2) & (blend_preds != 2)
    n_missed    = int(np.sum(missed_mask))

    cb_on_missed = np.argmax(cb_oof[missed_mask], axis=1)
    recoverable  = int(np.sum(cb_on_missed == 2))

    if recoverable >= 80:
        rota = "GATED_RESCUE_ONCE"
    elif recoverable >= 40:
        rota = "ALPHA_SWEEP_ONCE"
    else:
        rota = "SADECE_OPTUNA"

    print(f"\n  [FAZ0][RECOVERABLE_HIGH]")
    print(f"    Blend'in kaçırdığı High : {n_missed}")
    print(f"    CB'nin kurtardığı       : {recoverable} ({recoverable/n_missed*100:.1f}%)")
    print(f"    → ROTA                  : {rota}")
    print(f"    [FAZ0][ADIM_RH]: "
          f"recoverable={recoverable}/{n_missed} | rota={rota}")
    return {"n_missed": n_missed, "recoverable": recoverable, "rota": rota}


def faz0_alpha_sweep(
    blend_oof : np.ndarray,
    y         : np.ndarray,
    baseline  : dict,
    alphas    : list = None,
) -> dict:
    """
    [NEW-F0] Guardrail'li α-sweep.
    NOT: class_weight değiştirilmemiş BASELINE proba üzerinde çalışır.
    Kabul: WF1 >= ALPHA_ACCEPT_WF1 VE HRec >= ALPHA_ACCEPT_HREC
    """
    from sklearn.metrics import precision_score, recall_score
    if alphas is None:
        alphas = [1.00, 1.03, 1.05, 1.08, 1.10, 1.12, 1.15]

    print(f"\n  [FAZ0][ALPHA_SWEEP] Baseline proba üzerinde koşuluyor...")
    print(f"    Kabul kriteri: WF1>={ALPHA_ACCEPT_WF1} VE HRec>={ALPHA_ACCEPT_HREC}")
    print(f"    {'alpha':>6} | {'WF1':>8} | {'HRec':>8} | {'HPre':>8} | {'phc':>6} | {'MedF1':>8} | karar")
    print(f"    {'─'*70}")

    best_alpha  = None
    best_result = None

    for alpha in alphas:
        adj = blend_oof.copy()
        adj[:, 2] *= alpha
        adj /= adj.sum(axis=1, keepdims=True)
        preds = np.argmax(adj, axis=1)

        res = faz0_check_guardrails(preds, y, baseline, label="")
        wf1, hrec = res["wf1"], res["hrec"]

        accept = (wf1 >= ALPHA_ACCEPT_WF1) and (hrec >= ALPHA_ACCEPT_HREC) and (res["karar"] == "GEÇTİ")
        tag    = "✓ KABUL" if accept else "✗"
        print(f"    {alpha:>6.2f} | {wf1:>8.5f} | {hrec:>8.5f} | {res['hpre']:>8.5f} | "
              f"{res['phc']:>6} | {res['med_f1']:>8.5f} | {tag}")

        if accept and best_alpha is None:
            best_alpha  = alpha
            best_result = res

    print(f"\n    → En iyi α: {best_alpha}  ({'SUBMISSION ATA' if best_alpha else 'Faz 1e geç'})")
    return {"best_alpha": best_alpha, "best_result": best_result}


def faz0_gated_sweep(
    cb_oof    : np.ndarray,
    blend_oof : np.ndarray,
    y         : np.ndarray,
    baseline  : dict,
    thresholds: list = None,
) -> dict:
    """
    [NEW-F0] Gated rescue threshold sweep.
    Kriter: FPR_rescue < GATE_FPR_MAX VE TPR_rescue >= GATE_TPR_MIN
    NOT: Baseline proba üzerinde koşulur.
    """
    from sklearn.metrics import precision_score, recall_score
    if thresholds is None:
        thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    blend_preds = np.argmax(blend_oof, axis=1)
    missed_mask = (y == 2) & (blend_preds != 2)
    non_high_mask = (y != 2)

    print(f"\n  [FAZ0][GATED_SWEEP] OOF threshold sweep...")
    print(f"    Kriter: FPR < {GATE_FPR_MAX} VE TPR >= {GATE_TPR_MIN}")
    print(f"    {'thresh':>8} | {'TPR':>8} | {'FPR':>8} | {'WF1':>8} | {'phc':>6} | karar")
    print(f"    {'─'*60}")

    best_threshold = None
    best_result    = None

    for t in thresholds:
        gate = (np.argmax(cb_oof, axis=1) == 2) & (blend_oof[:, 2] > t)
        tpr  = float(gate[missed_mask].mean())   if missed_mask.any() else 0.0
        fpr  = float(gate[non_high_mask].mean()) if non_high_mask.any() else 1.0

        # Gated prediction
        gated_blend = blend_oof.copy()
        gated_blend[gate, 2] = np.maximum(gated_blend[gate, 2], 0.50)
        gated_blend /= gated_blend.sum(axis=1, keepdims=True)
        preds = np.argmax(gated_blend, axis=1)

        res = faz0_check_guardrails(preds, y, baseline, label="")
        accept = (fpr < GATE_FPR_MAX) and (tpr >= GATE_TPR_MIN) and (res["karar"] == "GEÇTİ")
        tag    = "✓ KABUL" if accept else "✗"

        print(f"    {t:>8.2f} | {tpr:>8.3f} | {fpr:>8.3f} | {res['wf1']:>8.5f} | "
              f"{res['phc']:>6} | {tag}")

        if accept and best_threshold is None:
            best_threshold = t
            best_result    = res

    print(f"\n    → En iyi threshold: {best_threshold}  "
          f"({'GATED RESCUE KULLAN' if best_threshold else 'Gated rescue uygulanmaz'})")
    return {"best_threshold": best_threshold, "best_result": best_result}




def report_oof_comparison(
    y           : np.ndarray,
    cb_oof      : np.ndarray,
    lgbm_oof    : np.ndarray,
    xgb_oof     : np.ndarray,
    optuna_oof  : np.ndarray,
    cb_scores   : list,
    lgbm_scores : list,
    xgb_scores  : list,
    best_info   : dict,
) -> None:
    eq_oof = (cb_oof + lgbm_oof + xgb_oof) / 3
    models = {
        "CatBoost"     : (cb_oof,    cb_scores),
        "LightGBM"     : (lgbm_oof,  lgbm_scores),
        "XGBoost"      : (xgb_oof,   xgb_scores),
        "1/3 Eşit"     : (eq_oof,    None),
        "Optuna Blend" : (optuna_oof, None),
    }
    rows = {}
    for name, (proba, scores) in models.items():
        preds = np.argmax(proba, 1)
        rows[name] = {
            "wf1"      : f1_score(y, preds, average="weighted"),
            "mf1"      : f1_score(y, preds, average="macro"),
            "fold_avg" : np.mean(scores) if scores else float("nan"),
            "fold_std" : np.std(scores)  if scores else float("nan"),
        }

    best_wf1 = max(rows, key=lambda k: rows[k]["wf1"])

    print("\n" + "═" * 76)
    print("  V9 KARŞILAŞTIRMALI OOF PERFORMANS ÖZETİ")
    print("═" * 76)
    print(f"  {'Model':<16} {'WF1':>10}  {'MF1':>10}  {'Fold Ort':>10}  {'Fold Std':>9}")
    print(f"  {'─'*16}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*9}")

    for name, m in rows.items():
        fa  = f"{m['fold_avg']:.5f}" if not np.isnan(m['fold_avg']) else "   —   "
        fs  = f"{m['fold_std']:.5f}" if not np.isnan(m['fold_std']) else "   —   "
        tag = "  ← Zindi" if name == "Optuna Blend" else (
              "  ← BEST"  if name == best_wf1 and name != "Optuna Blend" else "")
        print(f"  {name:<16}  {m['wf1']:>10.5f}  {m['mf1']:>10.5f}  {fa:>10}  {fs:>9}{tag}")

    print("═" * 76)
    print(f"\n  Optuna Ağırlıkları: "
          f"CB={best_info['w_cb']:.4f}  "
          f"LGBM={best_info['w_lgbm']:.4f}  "
          f"XGB={best_info['w_xgb']:.4f}")

    optuna_preds = np.argmax(optuna_oof, 1)
    print("\n  [Optuna Blend] Sınıf Bazlı OOF Raporu:")
    print(classification_report(y, optuna_preds, target_names=CLASS_NAMES, zero_division=0))


# =============================================================================
# BÖLÜM 8 — SUBMISSION
# =============================================================================

def create_submission(
    test_ids   : np.ndarray,
    test_proba : np.ndarray,
    sub_path   : Path,
    output_path: Path,
    label      : str = "",
) -> pd.DataFrame:
    pred_idx    = np.argmax(test_proba, 1)
    pred_labels = np.array([TARGET_INVERSE_MAPPING[i] for i in pred_idx])
    # [P1-A] SampleSubmission opsiyonel
    if sub_path is not None and Path(sub_path).exists():
        # SampleSubmission can be .csv or .xls/.xlsx
        _p = Path(sub_path)
        if _p.suffix.lower() in ['.xls', '.xlsx']:
            sample_sub = pd.read_excel(_p)
        else:
            sample_sub = pd.read_csv(_p)
        submission  = pd.DataFrame({ID_COL: test_ids, TARGET_COL: pred_labels})
        submission  = sample_sub[[ID_COL]].merge(submission, on=ID_COL, how="left")
    else:
        submission  = pd.DataFrame({ID_COL: test_ids, TARGET_COL: pred_labels})
    submission.to_csv(output_path, index=False)
    tag = f"[{label}] " if label else ""
    print(f"\n  ✓ {tag}Submission → {output_path}")
    print(f"    {submission[TARGET_COL].value_counts().to_string()}")
    return submission


# =============================================================================
# ANA AKIŞ
# =============================================================================

if __name__ == "__main__":

    # ── 1. Veri oku ──────────────────────────────────────────────────────────
    X_raw, y, X_test, test_ids = read_data(
        train_path = DATA_DIR / "Train.csv",
        test_path  = DATA_DIR / "Test.csv",
    )

    # ── 2. Eğitim döngüsü: ZindiFE → ZindiCluster(TE fix) → CB + LGBM + XGB ─
    (cb_oof,   cb_test,
     lgbm_oof, lgbm_test,
     xgb_oof,  xgb_test,
     cb_scores, lgbm_scores, xgb_scores) = train_oof_triple(
        X_raw  = X_raw,
        y      = y,
        X_test = X_test,
    )

    # ── 3. V9 Optuna baseline (Macro-F1) — karşılaştırma için tutulur ────────
    # V12 Optuna aşağıda (WF1 primary) ayrıca çalışacak

    # ── 4. V12 Optuna: WF1 primary + band seçimi + çift early-stop ───────────
    best_weights, best_info = optuna_weight_search(
        cb_oof=cb_oof, lgbm_oof=lgbm_oof, xgb_oof=xgb_oof, y=y
    )
    w_cb, w_lgbm, w_xgb = best_weights

    optuna_oof  = w_cb * cb_oof   + w_lgbm * lgbm_oof  + w_xgb * xgb_oof
    optuna_test = w_cb * cb_test  + w_lgbm * lgbm_test + w_xgb * xgb_test
    equal_test  = (cb_test + lgbm_test + xgb_test) / 3

    # ── 5. Karşılaştırmalı OOF raporu ────────────────────────────────────────
    report_oof_comparison(
        y=y, cb_oof=cb_oof, lgbm_oof=lgbm_oof, xgb_oof=xgb_oof,
        optuna_oof=optuna_oof, cb_scores=cb_scores,
        lgbm_scores=lgbm_scores, xgb_scores=xgb_scores, best_info=best_info,
    )

    # ── 6. [NEW-F0] FAZ 0 TEST PAKETİ ────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  FAZ 0 — V12.1 YÖN BELİRLEYİCİ TESTLER")
    print("═" * 72)

    # Medium baseline kaydet
    baseline = faz0_medium_baseline(optuna_oof, y)

    # [P0-B] ALPHA_ACCEPT_WF1 dinamik: V12.1'de hard-coded 0.8687 yok
    # baseline_wf1, Optuna blend OOF WF1'inden hesaplanır
    baseline_wf1       = baseline["MediumF1_base"]  # Optuna blend WF1 buraya yazılıyor
    # Daha doğru: doğrudan hesapla
    baseline_wf1       = float(f1_score(y, np.argmax(optuna_oof, 1), average="weighted"))
    ALPHA_ACCEPT_WF1   = baseline_wf1 - ALPHA_WF1_DELTA
    print(f"\n  [P0-B] Dinamik eşik: baseline_wf1={baseline_wf1:.5f} → "
          f"ALPHA_ACCEPT_WF1={ALPHA_ACCEPT_WF1:.5f}")

    # [P0-D] CB proba kalibrasyon kontrolü — gated rescue threshold öncesi
    calib_result = check_cb_calibration(cb_oof, y)
    gated_threshold_start = calib_result["threshold_suggestion"]

    # Recoverable High → rota kararı
    rh_result = faz0_recoverable_high(cb_oof, optuna_oof, y)

    # α-sweep (dinamik eşik ile — P0-B)
    alpha_result = faz0_alpha_sweep(optuna_oof, y, baseline)

    # Gated sweep (recoverable >= 40 ise anlamlı)
    if rh_result["recoverable"] >= 40:
        # [P0-D] Kalibrasyon sonucuna göre threshold aralığını ayarla
        if calib_result["bias"] == "inflated":
            gated_thresholds = [round(t, 2) for t in np.arange(
                gated_threshold_start, gated_threshold_start + 0.30, 0.05)]
        elif calib_result["bias"] == "deflated":
            gated_thresholds = [round(t, 2) for t in np.arange(
                max(0.15, gated_threshold_start - 0.10), gated_threshold_start + 0.20, 0.05)]
        else:
            gated_thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        gate_result = faz0_gated_sweep(cb_oof, optuna_oof, y, baseline, thresholds=gated_thresholds)
    else:
        print(f"\n  [FAZ0][GATED_SWEEP] Atlandı: recoverable={rh_result['recoverable']} < 40")
        gate_result = {"best_threshold": None, "best_result": None}

    # Faz 0 özeti
    print("\n" + "═" * 72)
    print("  FAZ 0 KARAR ÖZETİ")
    print("═" * 72)
    print(f"  Recoverable High : {rh_result['recoverable']}/{rh_result['n_missed']} → ROTA: {rh_result['rota']}")
    print(f"  En iyi α         : {alpha_result['best_alpha']}")
    print(f"  Gated threshold  : {gate_result['best_threshold']}")

    if alpha_result["best_alpha"] is not None:
        print(f"\n  ✓ KARAR: α={alpha_result['best_alpha']} ile SUBMISSION AT")
    elif gate_result["best_threshold"] is not None:
        print(f"\n  ✓ KARAR: Gated rescue (t={gate_result['best_threshold']}) uygula, sonra submit")
    else:
        print(f"\n  → Faz 0 geçemedi. Faz 1'e geç (Optuna hiperparametre).")

    # ── 7. Submission dosyaları ───────────────────────────────────────────────
    # [P1-A] SampleSubmission.csv opsiyonel — yoksa doğrudan DataFrame'den üret
    sub_path = DATA_DIR / "SampleSubmission.csv"
    if not sub_path.exists():
        print(f"  ⚠ SampleSubmission.csv bulunamadı → doğrudan ID+Target ile üretiliyor")
        sub_path = None

    create_submission(test_ids, cb_test,     sub_path, OUTPUT_DIR / "submission_v12_1_catboost.csv",    "CatBoost")
    create_submission(test_ids, lgbm_test,   sub_path, OUTPUT_DIR / "submission_v12_1_lgbm.csv",        "LightGBM")
    create_submission(test_ids, xgb_test,    sub_path, OUTPUT_DIR / "submission_v12_1_xgboost.csv",     "XGBoost")
    create_submission(test_ids, equal_test,  sub_path, OUTPUT_DIR / "submission_v12_1_equal_blend.csv", "1/3 Eşit")
    create_submission(test_ids, optuna_test, sub_path, OUTPUT_DIR / "submission_v12_1_optuna_blend.csv","Optuna Blend ← ANA")

    # ── 8. Artefaktları kaydet ────────────────────────────────────────────────
    for name, arr in {
        "oof_cb_v12"         : cb_oof,
        "oof_lgbm_v12"       : lgbm_oof,
        "oof_xgb_v12"        : xgb_oof,
        "oof_optuna_v12"     : optuna_oof,
        "test_cb_v12"        : cb_test,
        "test_lgbm_v12"      : lgbm_test,
        "test_xgb_v12"       : xgb_test,
        "test_optuna_v12"    : optuna_test,
        "y_true_v12"         : y,
    }.items():
        np.save(OUTPUT_DIR / f"{name}.npy", arr)

    # Evidence Pack özeti JSON
    evidence = {
        "version"            : "v12_1",
        "patches_applied"    : ["P0-A: recall fix", "P0-B: dynamic threshold",
                                "P0-C: CB params", "P0-D: calibration check",
                                "P1-A: submission independent"],
        "baseline_wf1"       : round(baseline_wf1, 5),
        "alpha_accept_wf1"   : round(ALPHA_ACCEPT_WF1, 5),
        "cb_calibration"     : calib_result,
        "optuna_weights"     : {"cb": float(w_cb), "lgbm": float(w_lgbm), "xgb": float(w_xgb)},
        "optuna_info"        : {k: float(v) if isinstance(v, (float, np.floating)) else v
                                for k, v in best_info.items()},
        "medium_baseline"    : baseline,
        "recoverable_high"   : rh_result,
        "best_alpha"         : alpha_result["best_alpha"],
        "best_gate_threshold": gate_result["best_threshold"],
        "fold_index_path"    : str(FOLD_INDEX_PATH),
    }
    with open(OUTPUT_DIR / "evidence_pack_v12_1.json", "w") as f:
        json.dump(evidence, f, indent=2)

    print("\n  ✓ Tüm artefaktlar kaydedildi.")
    print(f"\n  Optuna Ağırlıkları (V12.1):")
    print(f"    CatBoost = {w_cb:.4f}  (V12: 0.0060)")
    print(f"    LightGBM = {w_lgbm:.4f}  (V12: 0.3838)")
    print(f"    XGBoost  = {w_xgb:.4f}  (V12: 0.6103)")
    print(f"\n  Evidence Pack → {OUTPUT_DIR}/evidence_pack_v12_1.json")
    print(f"  → Zindi'ye gönder: submission_v12_1_optuna_blend.csv")
