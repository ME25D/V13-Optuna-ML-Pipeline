"""
=============================================================================
  Zindi | data.org Financial Health Prediction Challenge
  FAZ 7 — V13  DEEP OPTUNA (4 Fazlı Ayrıştırılmış Mimari)
  ─────────────────────────────────────────────────────────────────────────
  V13 Değişiklikleri (V12.1 üzerine):

  [V13-F1] FAZ 1 — Hiperparametre Optimizasyonu (3 bağımsız Optuna study):
           LGBM: num_leaves, min_child_samples, subsample, colsample_bytree,
                 reg_lambda, learning_rate  →  100 trial
           XGB:  max_depth, min_child_weight, subsample, colsample_bytree,
                 reg_lambda, learning_rate  →  100 trial
           CB:   depth, l2_leaf_reg, learning_rate (daraltıldı)  →  100 trial
           class_weights / class_weight SABİT (tüm modeller için)
           Pruning: MedianPruner(n_startup=20, n_warmup=10)
           User attrs: high_recall, high_f1, pred_high_count, medium_f1
           Optimizasyon: fold_data pre-compute → 5× FE+CE (3×100×5 değil!)

  [V13-F2] FAZ 2 — Ensemble Ağırlık (best params sabit, 50 trial):
           Seçim: WF1 >= best-0.001 BANDINDA HRec MAX
           ← BUG FIX: ilk geçen değil, en yüksek HRec'li seçilir

  [V13-F3] FAZ 3 — Faz 0 (iki kritik bug fix):
           [BUG-A] Alpha seçimi: MAX HRec from band (not first passing)
           [BUG-B] Gated gate: (cb_oof[:,2]>t_cb) AND (blend[:,2]>t_blend)
                   ← PROBA TABANLI (argmax yerine ikili threshold sweep)
           CB bestIteration uyarısı: fold < 100 → otomatik print

  [V13-F4] FAZ 4 — Seed Stabilitesi:
           seeds=[42, 123, 2025], aynı fold freeze
           OOF_WF1_mean ± std, HRec_mean ± std raporlanır
           std > 0.005 → uyarı: tek seed kararına güvenme

  [V13-PERF] fold_data ön-hesaplama:
           precompute_fold_data() → FE+CE tüm study'ler için 1 kez çalışır
           HP study'ler bu cache'den okur → ~60× hızlanma

  ─────────────────────────────────────────────────────────────────────────
  Korunanlar (V12.1'den dokunulmadı):
    ZindiClusterEngineer (FIX-TE, is_train logic) — seed param eklendi
    ZindiFeatureEngineer
    Split freeze (fold_indices_v12.json)
    Guardrail sistemi (Medium F1 drop kill-switch, phc tavan)
    recall_score modül seviyesinde import (NameError fix)
    check_cb_calibration (ECE raporu)
    tüm NaN fix ve string temizleme mantığı
  ─────────────────────────────────────────────────────────────────────────
  Hedef: Ensemble WF1 > 0.862 VE HRec >= 0.69
  Taban: LGBM standalone OOF WF1 = 0.86066
=============================================================================
"""

# ─── Standart Kütüphaneler ────────────────────────────────────────────────────
import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Modelleme ────────────────────────────────────────────────────────────────
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, recall_score, precision_score, classification_report
)
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

# [V13-F1] HP Optuna trial sayıları
OPTUNA_HP_TRIALS_LGBM = 100
OPTUNA_HP_TRIALS_XGB  = 100
OPTUNA_HP_TRIALS_CB   = 50   # [V13-SPEED] CB ağır → 50 trial yeterli sinyal verir

# [V13-F2] Ensemble trial sayısı
OPTUNA_BLEND_TRIALS   = 50

# [V13-F4] Seed stabilitesi
SEED_STABILITY_SEEDS    = [42, 123, 2025]
SEED_STABILITY_STD_WARN = 0.005

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
MEDIUM_F1_DROP_MAX  = 0.015
MEDIUM_REC_DROP_MAX = 0.020
PRED_HIGH_MULT_MAX  = 1.25
WF1_BAND            = 0.001
HREC_TARGET         = 0.72
ALPHA_WF1_DELTA     = 0.001
ALPHA_ACCEPT_WF1    = None   # Runtime'da set edilir
ALPHA_ACCEPT_HREC   = 0.68
GATE_FPR_MAX        = 0.05
GATE_TPR_MIN        = 0.30
EARLY_STOP_PATIENCE = 20

np.random.seed(SEED)


# =============================================================================
# BÖLÜM 1 — ZindiFeatureEngineer  ← V12.1 ile ÖZDEŞ
# =============================================================================

class ZindiFeatureEngineer(BaseEstimator, TransformerMixin):
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
# BÖLÜM 1.5 — ZindiClusterEngineer  [NEW-8] + [V13: seed param]
# =============================================================================

class ZindiClusterEngineer(BaseEstimator, TransformerMixin):
    """
    K-Means Arketip Kümeleme + Çok Sınıflı K-Fold Target Encoding.
    [V13] seed parametresi eklendi → seed stabilitesi için kontrollü varyans.
    """

    _CLUSTER_FEATURES = [
        "net_profit_margin",
        "expense_coverage",
        "turnover_per_month",
        "turnover_z_country",
        "business_turnover",
        "personal_income",
        "owner_age",
        "business_age_total_months",
        "financial_access_score",
        "insurance_score",
        "cashflow_risk_score",
        "growth_mindset_score",
        "vulnerability_index",
        "digital_maturity_score",
        "age_stability_interact",
        "insurance_barrier_score",
    ]

    def __init__(
        self,
        k_values     : list = CLUSTER_K_VALUES,
        n_inner_folds: int  = CLUSTER_TE_FOLDS,
        smooth_alpha : int  = TE_SMOOTH_ALPHA,
        seed         : int  = SEED,           # [V13] seed param
    ):
        self.k_values      = k_values
        self.n_inner_folds = n_inner_folds
        self.smooth_alpha  = smooth_alpha
        self.seed          = seed

    def _active_cols(self, X: pd.DataFrame) -> list:
        return [c for c in self._CLUSTER_FEATURES if c in X.columns]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ZindiClusterEngineer":
        cols = self._active_cols(X)
        self.cols_      = cols
        self.n_classes_ = len(np.unique(y))
        self.global_class_freq_ = np.array(
            [np.mean(y == c) for c in range(self.n_classes_)],
            dtype=np.float64
        )
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(
            X[cols].fillna(0).astype(float)
        )
        self.kmeans_ = {}
        for k in self.k_values:
            km = KMeans(
                n_clusters=k, random_state=self.seed,
                n_init=15, max_iter=300,
            )
            km.fit(X_scaled)
            self.kmeans_[k] = km

        self.te_maps_ = {}
        self.oof_te_  = {}
        y_arr = np.asarray(y, dtype=int)

        for k in self.k_values:
            cluster_labels = self.kmeans_[k].predict(X_scaled)
            inner_skf = StratifiedKFold(
                n_splits=self.n_inner_folds, shuffle=True,
                random_state=self.seed + 7,
            )
            oof_te = np.zeros((len(X), self.n_classes_), dtype=np.float64)
            for tr_idx, va_idx in inner_skf.split(X_scaled, y_arr):
                tr_c, va_c = cluster_labels[tr_idx], cluster_labels[va_idx]
                tr_y = y_arr[tr_idx]
                fold_map = self._build_te_map(tr_c, tr_y, k)
                for i, cid in enumerate(va_c):
                    oof_te[va_idx[i]] = fold_map.get(cid, self.global_class_freq_)
            self.oof_te_[k] = oof_te
            final_map = self._build_te_map(cluster_labels, y_arr, k)
            self.te_maps_[k] = final_map

        return self

    def _build_te_map(self, labels, y, k):
        te_map = {}
        alpha  = self.smooth_alpha
        for cid in np.unique(labels):
            mask  = labels == cid
            n     = mask.sum()
            probs = np.zeros(self.n_classes_, dtype=np.float64)
            for c in range(self.n_classes_):
                n_c = (y[mask] == c).sum()
                probs[c] = (n_c + alpha * self.global_class_freq_[c]) / (n + alpha)
            te_map[cid] = probs
        return te_map

    def transform(self, X: pd.DataFrame, y=None, is_train: bool = False) -> pd.DataFrame:
        df   = X.copy()
        cols = [c for c in self.cols_ if c in df.columns]
        X_scaled = self.scaler_.transform(df[cols].fillna(0).astype(float))

        for k in self.k_values:
            km     = self.kmeans_[k]
            labels = km.predict(X_scaled)
            dists  = km.transform(X_scaled)

            df[f"cluster_id_k{k}"]   = labels.astype(int)
            df[f"cluster_dist_k{k}"] = dists[np.arange(len(labels)), labels].astype(np.float32)

            if is_train:
                te_proba = self.oof_te_[k].astype(np.float32)
            else:
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
        return [f"cluster_id_k{k}" for k in self.k_values]


# =============================================================================
# BÖLÜM 2 — VERİ KATMANI  ← V12.1 ile ÖZDEŞ
# =============================================================================

def nuke_strings(df: pd.DataFrame, cat_cols: list = None) -> pd.DataFrame:
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
        if pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s):
            df[col] = s.fillna(0).astype(float)
            continue
        lowered = s.astype(str).str.strip().str.lower()
        mapped  = lowered.map(_MAP)
        unmatched_mask = mapped.isna()
        if unmatched_mask.any():
            numeric_fallback = pd.to_numeric(s[unmatched_mask], errors="coerce").fillna(0)
            mapped[unmatched_mask] = numeric_fallback
        df[col] = mapped.astype(float)
    return df


def cb_fix_cats(df, cat_cols, missing_token="__MISSING__"):
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


def cb_assert_no_nan_in_cats(df, cat_cols, label=""):
    if not cat_cols:
        return
    cols = [c for c in cat_cols if c in df.columns]
    if not cols:
        return
    if df[cols].isna().any().any():
        bad = {c: int(df[c].isna().sum()) for c in cols if df[c].isna().any()}
        tag = f" [{label}]" if label else ""
        raise ValueError(f"NaN remains in CatBoost cat cols{tag}: {bad}")


def preprocess_raw(df):
    return nuke_strings(df, cat_cols=CAT_FEATURES)


def encode_string_columns(df, cat_cols):
    return nuke_strings(df, cat_cols=cat_cols)


def read_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    y        = encode_target(train_df[TARGET_COL])
    test_ids = test_df[ID_COL].values
    X_raw    = train_df.drop(columns=[ID_COL, TARGET_COL], errors="ignore").reset_index(drop=True)
    X_test   = test_df.drop(columns=[ID_COL], errors="ignore").reset_index(drop=True)
    X_raw    = preprocess_raw(X_raw)
    X_test   = preprocess_raw(X_test)
    print(f"\n  ✓ Veri okundu")
    print(f"    Train (ham) : {X_raw.shape}   Test (ham) : {X_test.shape}")
    for label, code in TARGET_MAPPING.items():
        n = (y == code).sum()
        print(f"    {label:8s} ({code}) → {n:5d} örnek  ({n/len(y)*100:.1f}%)")
    return X_raw, y, X_test, test_ids


def encode_target(series):
    y = series.map(TARGET_MAPPING).values
    if np.isnan(y.astype(float)).any():
        unknown = set(series.unique()) - set(TARGET_MAPPING.keys())
        raise ValueError(f"Bilinmeyen sınıf: {unknown}")
    return y.astype(int)


def align_columns(source, target):
    return target.reindex(columns=source.columns, fill_value=0)


# =============================================================================
# BÖLÜM 3 — HİPERPARAMETRE FONKSİYONLARI  [V13: override dict desteği]
# =============================================================================

def get_catboost_params(overrides: dict = None) -> dict:
    """[V13] Temel CB parametreleri. overrides ile HP study sırasında patch edilir.
    '_' ile başlayan key'ler (evidence-only meta) otomatik filtrelenir.
    """
    base = dict(
        iterations=3000, learning_rate=0.02, depth=7, l2_leaf_reg=3.0,
        loss_function="MultiClass", eval_metric="TotalF1:average=Weighted",
        auto_class_weights=None,
        class_weights={0: 1, 1: 2, 2: 12},
        random_seed=SEED,
        early_stopping_rounds=150,
        verbose=0, thread_count=-1,
    )
    if overrides:
        # '_' ile başlayan key'ler evidence-only meta verilerdir, CB'ye geçirilmez
        clean = {k: v for k, v in overrides.items() if not k.startswith("_")}
        base.update(clean)
    return base


def get_lgbm_params(overrides: dict = None) -> dict:
    """[V13] Temel LGBM parametreleri. overrides ile HP study sırasında patch edilir."""
    base = dict(
        n_estimators=1500, learning_rate=0.05, num_leaves=63,
        max_depth=-1, min_child_samples=20, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multiclass", num_class=N_CLASSES,
        class_weight="balanced", metric="multi_logloss",
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    if overrides:
        base.update(overrides)
    return base


def get_xgb_params(overrides: dict = None) -> dict:
    """[V13] Temel XGB parametreleri. overrides ile HP study sırasında patch edilir."""
    base = dict(
        n_estimators=1200, learning_rate=0.05, max_depth=6,
        min_child_weight=5, subsample=0.8,
        colsample_bytree=0.8, colsample_bylevel=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", num_class=N_CLASSES,
        eval_metric="mlogloss", tree_method="hist",
        random_state=SEED, n_jobs=-1, verbosity=0,
    )
    if overrides:
        base.update(overrides)
    return base


# =============================================================================
# BÖLÜM 4 — MODEL VERİ HAZIRLIK YARDIMCILARI  ← V12.1 ile ÖZDEŞ
# =============================================================================

def prepare_lgbm_data(X_trn, X_val, X_test, extra_cat_cols):
    X_tr, X_va, X_te = X_trn.copy(), X_val.copy(), X_test.copy()
    all_cats = [c for c in (CAT_FEATURES + extra_cat_cols) if c in X_tr.columns]
    X_tr = encode_string_columns(X_tr, all_cats)
    X_va = encode_string_columns(X_va, all_cats)
    X_te = encode_string_columns(X_te, all_cats)
    for col in all_cats:
        X_tr[col] = X_tr[col].astype("category")
        cats = X_tr[col].cat.categories
        X_va[col] = X_va[col].astype(pd.CategoricalDtype(categories=cats))
        X_te[col] = X_te[col].astype(pd.CategoricalDtype(categories=cats))
    return X_tr, X_va, X_te, all_cats


def prepare_xgb_data(X_trn, X_val, X_test, y_trn, extra_cat_cols):
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    X_tr, X_va, X_te = X_trn.copy(), X_val.copy(), X_test.copy()
    all_cats = [c for c in (CAT_FEATURES + extra_cat_cols) if c in X_tr.columns]
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
# BÖLÜM 5 — FOLD VERİSİ ÖN-HESAPLAMA  [V13-PERF]
# =============================================================================

def get_or_create_folds(X_raw, y):
    """[NEW-SF] Split Freeze."""
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
        print(f"  ✓ [SPLIT FREEZE] Fold indeksleri oluşturuldu: {FOLD_INDEX_PATH}")
    for i, (tr, va) in enumerate(folds):
        assert len(set(tr) & set(va)) == 0, f"HATA: Fold {i+1}'de overlap var!"
    print(f"  ✓ [SPLIT FREEZE] Overlap kontrolü geçti.")
    return folds


def precompute_fold_data(X_raw: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame,
                          folds: list, seed: int = SEED) -> list:
    """
    [V13-PERF] FE + Cluster Engineering tüm fold'lar için ön-hesaplanır.

    HP Optuna study'leri bu precomputed veriden okur → FE+CE overhead'i
    3×100×5=1500 kez yerine sadece 5 kez çalışır.

    Returns: list of dicts per fold:
        X_trn, y_trn, X_val, y_val, X_test_cl, cluster_id_cols,
        train_idx, val_idx
    """
    print(f"\n  ▶ [V13-PERF] Fold verisi ön-hesaplanıyor (seed={seed})...")
    fold_cache = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"    Fold {fold_idx}/{len(folds)}: FE + CE...", end=" ", flush=True)

        X_trn_raw = X_raw.iloc[train_idx].reset_index(drop=True)
        X_val_raw = X_raw.iloc[val_idx].reset_index(drop=True)
        y_trn     = y[train_idx]
        y_val     = y[val_idx]

        # FE — leakage-free
        fe = ZindiFeatureEngineer()
        X_trn_fe  = fe.fit_transform(X_trn_raw)
        X_val_fe  = align_columns(X_trn_fe, fe.transform(X_val_raw))
        X_test_fe = align_columns(X_trn_fe, fe.transform(X_test.copy()))

        # CE — leakage-free, seed kontrollü
        ce = ZindiClusterEngineer(seed=seed)
        ce.fit(X_trn_fe, y_trn)
        X_trn_cl  = ce.transform(X_trn_fe, is_train=True)
        X_val_cl  = align_columns(X_trn_cl, ce.transform(X_val_fe,  is_train=False))
        X_test_cl = align_columns(X_trn_cl, ce.transform(X_test_fe, is_train=False))

        fold_cache.append({
            "X_trn"          : X_trn_cl,
            "y_trn"          : y_trn,
            "X_val"          : X_val_cl,
            "y_val"          : y_val,
            "X_test_cl"      : X_test_cl,
            "cluster_id_cols": ce.cluster_id_cols,
            "train_idx"      : train_idx,
            "val_idx"        : val_idx,
        })
        print(f"✓ {X_trn_cl.shape[1]} özellik")

    print(f"  ✓ Fold verisi hazır. ({len(fold_cache)} fold)")
    return fold_cache


# =============================================================================
# BÖLÜM 6 — EĞİTİM DÖNGÜSÜ  [V13: params + bestIter uyarısı]
# =============================================================================

def train_oof_triple(
    X_raw      : pd.DataFrame,
    y          : np.ndarray,
    X_test     : pd.DataFrame,
    lgbm_params: dict = None,
    xgb_params : dict = None,
    cb_params  : dict = None,
    seed       : int  = SEED,
    fold_cache : list = None,   # [V13] precomputed fold data, None → recompute
) -> tuple:
    """
    Leakage-free 5-Fold OOF: ZindiFE → ZindiCluster → CB + LGBM + XGB.
    [V13] Accepts best params from FAZ 1 HP study.
    [V13] CB bestIteration uyarısı: fold < 100 → otomatik print.
    [V13] fold_cache → FE+CE tekrar çalıştırılmaz.
    """
    # Parametre birleştirme
    _lgbm_p = get_lgbm_params(lgbm_params)
    _xgb_p  = get_xgb_params(xgb_params)
    _cb_p   = get_catboost_params(cb_params)

    # Seed güncelle
    _lgbm_p["random_state"] = seed
    _xgb_p["random_state"]  = seed
    _cb_p["random_seed"]    = seed

    folds = get_or_create_folds(X_raw, y)

    # fold_cache yoksa yeni hesapla
    if fold_cache is None:
        fold_cache = precompute_fold_data(X_raw, y, X_test, folds, seed=seed)

    n_train = len(X_raw)
    n_test  = len(X_test)

    cb_oof   = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    lgbm_oof = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    xgb_oof  = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    cb_test   = np.zeros((n_test, N_CLASSES), dtype=np.float64)
    lgbm_test = np.zeros((n_test, N_CLASSES), dtype=np.float64)
    xgb_test  = np.zeros((n_test, N_CLASSES), dtype=np.float64)

    cb_scores, lgbm_scores, xgb_scores = [], [], []
    cb_best_iters = []   # [V13] bestIteration per fold

    for fold_idx, fd in enumerate(fold_cache, start=1):
        train_idx        = fd["train_idx"]
        val_idx          = fd["val_idx"]
        X_trn            = fd["X_trn"]
        y_trn            = fd["y_trn"]
        X_val            = fd["X_val"]
        y_val            = fd["y_val"]
        X_test_cl        = fd["X_test_cl"]
        cluster_id_cols  = fd["cluster_id_cols"]

        print(f"\n{'═' * 72}")
        print(f"  FOLD {fold_idx} / {N_SPLITS}  "
              f"(train={len(train_idx):,}  val={len(val_idx):,})")
        print(f"{'═' * 72}")

        # ── MODEL A — CatBoost ────────────────────────────────────────────────
        print(f"\n  ▶ [1/3] CatBoost eğitimi...")

        active_cats_cb = (
            [c for c in CAT_FEATURES if c in X_trn.columns]
            + [c for c in cluster_id_cols if c in X_trn.columns]
        )
        X_trn_cb  = X_trn.copy()
        X_val_cb  = X_val.copy()
        X_test_cb = X_test_cl.copy()
        for col in cluster_id_cols:
            if col in X_trn_cb.columns:
                X_trn_cb[col]  = X_trn_cb[col].astype(str)
                X_val_cb[col]  = X_val_cb[col].astype(str)
                X_test_cb[col] = X_test_cb[col].astype(str)

        X_trn_cb  = encode_string_columns(X_trn_cb,  active_cats_cb)
        X_val_cb  = encode_string_columns(X_val_cb,  active_cats_cb)
        X_test_cb = encode_string_columns(X_test_cb, active_cats_cb)
        X_trn_cb  = nuke_strings(X_trn_cb,  cat_cols=active_cats_cb)
        X_val_cb  = nuke_strings(X_val_cb,  cat_cols=active_cats_cb)
        X_test_cb = nuke_strings(X_test_cb, cat_cols=active_cats_cb)
        X_trn_cb  = cb_fix_cats(X_trn_cb,  active_cats_cb)
        X_val_cb  = cb_fix_cats(X_val_cb,  active_cats_cb)
        X_test_cb = cb_fix_cats(X_test_cb, active_cats_cb)
        cb_assert_no_nan_in_cats(X_trn_cb, active_cats_cb, label="TRAIN")
        cb_assert_no_nan_in_cats(X_val_cb, active_cats_cb, label="VAL")

        train_pool = Pool(data=X_trn_cb, label=y_trn, cat_features=active_cats_cb)
        val_pool   = Pool(data=X_val_cb, label=y_val, cat_features=active_cats_cb)
        test_pool  = Pool(data=X_test_cb,             cat_features=active_cats_cb)

        # [FIX] evidence-only meta keys (_cb_early_stop_alert vb.) CB'ye geçirilmez
        _cb_p_clean = {k: v for k, v in _cb_p.items() if not k.startswith("_")}
        cb_model = CatBoostClassifier(**_cb_p_clean)
        cb_model.fit(train_pool, eval_set=val_pool, plot=False)

        best_iter = cb_model.get_best_iteration()
        cb_best_iters.append(best_iter)

        # [V13-F3] bestIteration uyarısı
        if best_iter is not None and best_iter < 100:
            print(f"  ⚠ [CB UYARI] Fold {fold_idx} bestIteration={best_iter} < 100 → "
                  f"CB bu fold'da erken durdu, overfit riski!")

        fold_cb_oof     = cb_model.predict_proba(val_pool)
        cb_oof[val_idx] = fold_cb_oof
        cb_test        += cb_model.predict_proba(test_pool) / N_SPLITS

        fold_cb_f1 = f1_score(y_val, np.argmax(fold_cb_oof, 1), average="weighted")
        cb_scores.append(fold_cb_f1)
        print(f"  ✓ CatBoost   Fold {fold_idx}  WF1={fold_cb_f1:.5f}  "
              f"bestIter={best_iter}")

        # ── MODEL B — LightGBM ────────────────────────────────────────────────
        print(f"\n  ▶ [2/3] LightGBM eğitimi...")

        X_tr_lgbm, X_va_lgbm, X_te_lgbm, lgbm_cats = prepare_lgbm_data(
            X_trn, X_val, X_test_cl, extra_cat_cols=cluster_id_cols
        )
        lgbm_model = LGBMClassifier(**_lgbm_p)
        lgbm_model.fit(
            X_tr_lgbm, y_trn,
            eval_set   = [(X_va_lgbm, y_val)],
            callbacks  = [
                lgb.early_stopping(stopping_rounds=80, verbose=False),
                lgb.log_evaluation(period=9999),
            ],
            categorical_feature = lgbm_cats,
        )
        fold_lgbm_oof     = lgbm_model.predict_proba(X_va_lgbm)
        lgbm_oof[val_idx] = fold_lgbm_oof
        lgbm_test        += lgbm_model.predict_proba(X_te_lgbm) / N_SPLITS

        fold_lgbm_f1 = f1_score(y_val, np.argmax(fold_lgbm_oof, 1), average="weighted")
        lgbm_scores.append(fold_lgbm_f1)
        print(f"  ✓ LightGBM   Fold {fold_idx}  WF1={fold_lgbm_f1:.5f}")

        # ── MODEL C — XGBoost ─────────────────────────────────────────────────
        print(f"\n  ▶ [3/3] XGBoost eğitimi...")

        X_tr_xgb, X_va_xgb, X_te_xgb, sw = prepare_xgb_data(
            X_trn, X_val, X_test_cl, y_trn, extra_cat_cols=cluster_id_cols
        )
        xgb_model = XGBClassifier(**_xgb_p)
        xgb_model.fit(
            X_tr_xgb, y_trn,
            sample_weight         = sw,
            eval_set              = [(X_va_xgb, y_val)],
            early_stopping_rounds = 80,
            verbose               = False,
        )
        fold_xgb_oof     = xgb_model.predict_proba(X_va_xgb)
        xgb_oof[val_idx] = fold_xgb_oof
        xgb_test        += xgb_model.predict_proba(X_te_xgb) / N_SPLITS

        fold_xgb_f1 = f1_score(y_val, np.argmax(fold_xgb_oof, 1), average="weighted")
        xgb_scores.append(fold_xgb_f1)
        print(f"  ✓ XGBoost    Fold {fold_idx}  WF1={fold_xgb_f1:.5f}")

        # ── Fold Özeti
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

    # [V13-F3] CB bestIter özeti
    print(f"\n  [V13] CB bestIteration özeti: {cb_best_iters}")
    low_iters = [(i+1, bi) for i, bi in enumerate(cb_best_iters) if bi is not None and bi < 100]
    if low_iters:
        print(f"  ⚠ [CB UYARI] Şu fold'larda bestIter < 100: {low_iters}")
        print(f"    → CB bu fold'larda unstable. Ensemble'da CB ağırlığı düşük tutulmalı.")
    else:
        print(f"  ✓ [CB] Tüm fold'larda bestIteration >= 100 → CB kararlı.")

    return (cb_oof,   cb_test,
            lgbm_oof, lgbm_test,
            xgb_oof,  xgb_test,
            cb_scores, lgbm_scores, xgb_scores,
            cb_best_iters)


# =============================================================================
# BÖLÜM 7 — FAZ 1: HİPERPARAMETRE OPTİMİZASYONU  [V13-F1]
# =============================================================================

def _lgbm_fold_oof(fold_cache, y, lgbm_params):
    """Tek LGBM parametresiyle 5-fold OOF WF1 hesapla (HP study helper)."""
    n_train = sum(len(fd["y_val"]) for fd in fold_cache)
    oof = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    for fd in fold_cache:
        X_tr, y_tr = fd["X_trn"], fd["y_trn"]
        X_va, y_va = fd["X_val"], fd["y_val"]
        val_idx    = fd["val_idx"]
        cid_cols   = fd["cluster_id_cols"]
        X_tr_l, X_va_l, _, cats = prepare_lgbm_data(X_tr, X_va, X_va, cid_cols)
        model = LGBMClassifier(**lgbm_params)
        model.fit(
            X_tr_l, y_tr,
            eval_set   = [(X_va_l, y_va)],
            callbacks  = [
                lgb.early_stopping(stopping_rounds=80, verbose=False),
                lgb.log_evaluation(period=9999),
            ],
            categorical_feature = cats,
        )
        oof[val_idx] = model.predict_proba(X_va_l)
    return oof


def _xgb_fold_oof(fold_cache, y, xgb_params):
    """Tek XGB parametresiyle 5-fold OOF WF1 hesapla (HP study helper)."""
    n_train = sum(len(fd["y_val"]) for fd in fold_cache)
    oof = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    for fd in fold_cache:
        X_tr, y_tr = fd["X_trn"], fd["y_trn"]
        X_va, y_va = fd["X_val"], fd["y_val"]
        val_idx    = fd["val_idx"]
        cid_cols   = fd["cluster_id_cols"]
        X_tr_x, X_va_x, _, sw = prepare_xgb_data(X_tr, X_va, X_va, y_tr, cid_cols)
        model = XGBClassifier(**xgb_params)
        model.fit(
            X_tr_x, y_tr,
            sample_weight         = sw,
            eval_set              = [(X_va_x, y_va)],
            early_stopping_rounds = 80,
            verbose               = False,
        )
        oof[val_idx] = model.predict_proba(X_va_x)
    return oof


def _cb_fold_oof(fold_cache, y, cb_params):
    """Tek CB parametresiyle 5-fold OOF WF1 hesapla (HP study helper)."""
    n_train = sum(len(fd["y_val"]) for fd in fold_cache)
    oof = np.zeros((n_train, N_CLASSES), dtype=np.float64)
    for fd in fold_cache:
        X_trn, y_trn = fd["X_trn"], fd["y_trn"]
        X_val, y_val = fd["X_val"], fd["y_val"]
        val_idx      = fd["val_idx"]
        cid_cols     = fd["cluster_id_cols"]

        active_cats = (
            [c for c in CAT_FEATURES if c in X_trn.columns]
            + [c for c in cid_cols if c in X_trn.columns]
        )
        Xtr = nuke_strings(cb_fix_cats(
            encode_string_columns(X_trn.copy(), active_cats), active_cats
        ), cat_cols=active_cats)
        Xva = nuke_strings(cb_fix_cats(
            encode_string_columns(X_val.copy(), active_cats), active_cats
        ), cat_cols=active_cats)

        tp = Pool(data=Xtr, label=y_trn, cat_features=active_cats)
        vp = Pool(data=Xva, label=y_val, cat_features=active_cats)
        model = CatBoostClassifier(**cb_params)
        model.fit(tp, eval_set=vp, plot=False)
        oof[val_idx] = model.predict_proba(vp)
    return oof


def _compute_oof_metrics(oof, y):
    """OOF proba → WF1, HRec, HF1, phc, MedF1 hesapla."""
    preds  = np.argmax(oof, 1)
    wf1    = f1_score(y, preds, average="weighted")
    hrec   = recall_score(y, preds, labels=[2], average="macro")
    hf1    = f1_score(y, preds, labels=[2], average="macro")
    phc    = int(np.sum(preds == 2))
    med_f1 = f1_score(y, preds, labels=[1], average="macro")
    return wf1, hrec, hf1, phc, med_f1


def _make_hp_progress_callback(label: str, log_every: int = 10):
    """
    [V13-PROGRESS] Her log_every trial'da bir progress satırı yazdırır.
    Trial numarası, mevcut best WF1, HRec ve pruned sayısı gösterilir.
    Bu sayede 100-trial study'nin donmuş gibi görünmesi önlenir.
    """
    def callback(study: optuna.Study, trial: optuna.Trial) -> None:
        n = trial.number + 1  # 1-indexed
        if n % log_every == 0 or n == 1:
            best = study.best_value if study.best_trial else float("nan")
            pruned = sum(1 for t in study.trials
                         if t.state == optuna.trial.TrialState.PRUNED)
            best_hrec = study.best_trial.user_attrs.get("high_recall", 0.0) \
                if study.best_trial else 0.0
            print(f"    [{label}] trial={n:>3}  best_WF1={best:.5f}  "
                  f"best_HRec={best_hrec:.5f}  pruned={pruned}")
    return callback


def optuna_hyper_lgbm(fold_cache: list, y: np.ndarray,
                       n_trials: int = OPTUNA_HP_TRIALS_LGBM) -> dict:
    """
    [V13-F1] LGBM hiperparametre optimizasyonu.
    class_weight='balanced' SABİT. Diğer 6 parametre aranır.
    """
    print(f"\n{'═' * 72}")
    print(f"  FAZ 1 / A — LGBM HİPERPARAMETRE OPTİMİZASYONU ({n_trials} trial)")
    print(f"{'═' * 72}")

    def objective(trial: optuna.Trial) -> float:
        overrides = {
            "num_leaves"        : trial.suggest_int("num_leaves", 50, 200),
            "min_child_samples" : trial.suggest_int("min_child_samples", 10, 50),
            "subsample"         : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda"        : trial.suggest_float("reg_lambda", 0.5, 5.0),
            "learning_rate"     : trial.suggest_float("learning_rate", 0.01, 0.05),
        }
        params = get_lgbm_params(overrides)

        # Per-fold intermediate reporting for pruning
        n_train = len(y)
        oof = np.zeros((n_train, N_CLASSES), dtype=np.float64)
        for step, fd in enumerate(fold_cache):
            X_tr, y_tr = fd["X_trn"], fd["y_trn"]
            X_va, y_va = fd["X_val"], fd["y_val"]
            val_idx    = fd["val_idx"]
            cid_cols   = fd["cluster_id_cols"]

            X_tr_l, X_va_l, _, cats = prepare_lgbm_data(X_tr, X_va, X_va, cid_cols)
            model = LGBMClassifier(**params)
            model.fit(
                X_tr_l, y_tr,
                eval_set   = [(X_va_l, y_va)],
                callbacks  = [
                    lgb.early_stopping(stopping_rounds=80, verbose=False),
                    lgb.log_evaluation(period=9999),
                ],
                categorical_feature = cats,
            )
            fold_proba = model.predict_proba(X_va_l)
            oof[val_idx] = fold_proba

            fold_wf1 = f1_score(y_va, np.argmax(fold_proba, 1), average="weighted")
            trial.report(fold_wf1, step=step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        wf1, hrec, hf1, phc, med_f1 = _compute_oof_metrics(oof, y)
        trial.set_user_attr("high_recall",     hrec)
        trial.set_user_attr("high_f1",         hf1)
        trial.set_user_attr("pred_high_count", phc)
        trial.set_user_attr("medium_f1",       med_f1)
        return wf1

    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=10)
    study  = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[_make_hp_progress_callback("LGBM")])

    pruned_count = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    completed    = [t for t in study.trials if t.value is not None]
    abs_best     = max(completed, key=lambda t: t.value)
    best_wf1     = study.best_value
    band         = [t for t in completed if t.value >= best_wf1 - WF1_BAND]
    hrec_best    = max(band, key=lambda t: t.user_attrs.get("high_recall", 0.0))

    print(f"\n  ✓ LGBM HP tamamlandı")
    print(f"    Toplam trial    : {len(study.trials)}  (pruned={pruned_count})")
    print(f"    [ABS BEST WF1]  trial={abs_best.number}  WF1={abs_best.value:.5f}  "
          f"HRec={abs_best.user_attrs.get('high_recall', 0):.5f}  params={abs_best.params}")
    print(f"    [BAND HRec MAX] trial={hrec_best.number}  WF1={hrec_best.value:.5f}  "
          f"HRec={hrec_best.user_attrs.get('high_recall', 0):.5f}  params={hrec_best.params}")
    if abs_best.number != hrec_best.number:
        print(f"    → [V13] WF1 band içinde daha iyi HRec: trial {hrec_best.number} seçildi")
    else:
        print(f"    → Absolute best = band HRec max → tek aday seçildi")

    return get_lgbm_params(hrec_best.params)


def optuna_hyper_xgb(fold_cache: list, y: np.ndarray,
                      n_trials: int = OPTUNA_HP_TRIALS_XGB) -> dict:
    """
    [V13-F1] XGB hiperparametre optimizasyonu.
    scale_pos_weight yerine sample_weight="balanced" SABİT. 6 param aranır.
    """
    print(f"\n{'═' * 72}")
    print(f"  FAZ 1 / B — XGB HİPERPARAMETRE OPTİMİZASYONU ({n_trials} trial)")
    print(f"{'═' * 72}")

    def objective(trial: optuna.Trial) -> float:
        overrides = {
            "max_depth"        : trial.suggest_int("max_depth", 4, 8),
            "min_child_weight" : trial.suggest_int("min_child_weight", 3, 10),
            "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 0.5, 5.0),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.05),
        }
        params = get_xgb_params(overrides)

        n_train = len(y)
        oof = np.zeros((n_train, N_CLASSES), dtype=np.float64)
        for step, fd in enumerate(fold_cache):
            X_tr, y_tr = fd["X_trn"], fd["y_trn"]
            X_va, y_va = fd["X_val"], fd["y_val"]
            val_idx    = fd["val_idx"]
            cid_cols   = fd["cluster_id_cols"]

            X_tr_x, X_va_x, _, sw = prepare_xgb_data(X_tr, X_va, X_va, y_tr, cid_cols)
            model = XGBClassifier(**params)
            model.fit(
                X_tr_x, y_tr,
                sample_weight         = sw,
                eval_set              = [(X_va_x, y_va)],
                early_stopping_rounds = 80,
                verbose               = False,
            )
            fold_proba = model.predict_proba(X_va_x)
            oof[val_idx] = fold_proba

            fold_wf1 = f1_score(y_va, np.argmax(fold_proba, 1), average="weighted")
            trial.report(fold_wf1, step=step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        wf1, hrec, hf1, phc, med_f1 = _compute_oof_metrics(oof, y)
        trial.set_user_attr("high_recall",     hrec)
        trial.set_user_attr("high_f1",         hf1)
        trial.set_user_attr("pred_high_count", phc)
        trial.set_user_attr("medium_f1",       med_f1)
        # fold_metrics_summary — per-fold WF1 for denetim
        fold_wf1s = [
            round(f1_score(y[fd["val_idx"]], np.argmax(oof[fd["val_idx"]], 1), average="weighted"), 5)
            for fd in fold_cache
        ]
        trial.set_user_attr("fold_metrics_summary", fold_wf1s)
        return wf1

    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=10)
    study  = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[_make_hp_progress_callback("XGB")])

    pruned_count = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    completed = [t for t in study.trials if t.value is not None]

    abs_best  = max(completed, key=lambda t: t.value)
    best_wf1  = study.best_value
    band      = [t for t in completed if t.value >= best_wf1 - WF1_BAND]
    hrec_best = max(band, key=lambda t: t.user_attrs.get("high_recall", 0.0))

    print(f"\n  ✓ XGB HP tamamlandı")
    print(f"    Toplam trial    : {len(study.trials)}  (pruned={pruned_count})")
    print(f"    [ABS BEST WF1]  trial={abs_best.number}  WF1={abs_best.value:.5f}  "
          f"HRec={abs_best.user_attrs.get('high_recall', 0):.5f}  params={abs_best.params}")
    print(f"    [BAND HRec MAX] trial={hrec_best.number}  WF1={hrec_best.value:.5f}  "
          f"HRec={hrec_best.user_attrs.get('high_recall', 0):.5f}  params={hrec_best.params}")
    if abs_best.number != hrec_best.number:
        print(f"    → WF1 band içinde daha iyi HRec: trial {hrec_best.number} seçildi")

    return get_xgb_params(hrec_best.params)


def optuna_hyper_cb(fold_cache: list, y: np.ndarray,
                     n_trials: int = OPTUNA_HP_TRIALS_CB) -> dict:
    """
    [V13-F1] CB hiperparametre optimizasyonu.
    class_weights SABİT {0:1, 1:2, 2:12}. 3 param aranır (OD riski için daraltılmış).
    iterations=3000, early_stopping=150 SABİT.
    """
    print(f"\n{'═' * 72}")
    print(f"  FAZ 1 / C — CB HİPERPARAMETRE OPTİMİZASYONU ({n_trials} trial)")
    print(f"    [NOT] learning_rate aralığı [0.005, 0.03] — OD riski için daraltıldı")
    print(f"{'═' * 72}")

    def objective(trial: optuna.Trial) -> float:
        overrides = {
            "depth"               : trial.suggest_int("depth", 5, 9),
            "l2_leaf_reg"         : trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "learning_rate"       : trial.suggest_float("learning_rate", 0.005, 0.03),
            # [V13-SPEED] HP search: 1000 iter + ES=80 yeterli sinyal verir (~6x hız)
            # Final train_oof_triple'da get_catboost_params() → 3000 iter kullanılır
            "iterations"           : 1000,
            "early_stopping_rounds": 80,
        }
        params = get_catboost_params(overrides)

        n_train = len(y)
        oof = np.zeros((n_train, N_CLASSES), dtype=np.float64)
        fold_best_iters = []
        fold_high_proba_means = []

        for step, fd in enumerate(fold_cache):
            X_trn, y_trn = fd["X_trn"], fd["y_trn"]
            X_val, y_val = fd["X_val"], fd["y_val"]
            val_idx      = fd["val_idx"]
            cid_cols     = fd["cluster_id_cols"]

            active_cats = (
                [c for c in CAT_FEATURES if c in X_trn.columns]
                + [c for c in cid_cols   if c in X_trn.columns]
            )
            Xtr = nuke_strings(cb_fix_cats(
                encode_string_columns(X_trn.copy(), active_cats), active_cats
            ), cat_cols=active_cats)
            Xva = nuke_strings(cb_fix_cats(
                encode_string_columns(X_val.copy(), active_cats), active_cats
            ), cat_cols=active_cats)

            tp = Pool(data=Xtr, label=y_trn, cat_features=active_cats)
            vp = Pool(data=Xva, label=y_val, cat_features=active_cats)
            model = CatBoostClassifier(**params)
            model.fit(tp, eval_set=vp, plot=False)

            fold_proba = model.predict_proba(vp)
            oof[val_idx] = fold_proba

            bi = model.get_best_iteration()
            fold_best_iters.append(bi if bi is not None else -1)
            fold_high_proba_means.append(round(float(np.mean(fold_proba[:, 2])), 5))

            fold_wf1 = f1_score(y_val, np.argmax(fold_proba, 1), average="weighted")
            trial.report(fold_wf1, step=step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        wf1, hrec, hf1, phc, med_f1 = _compute_oof_metrics(oof, y)
        trial.set_user_attr("high_recall",          hrec)
        trial.set_user_attr("high_f1",              hf1)
        trial.set_user_attr("pred_high_count",      phc)
        trial.set_user_attr("medium_f1",            med_f1)
        trial.set_user_attr("fold_best_iterations", fold_best_iters)
        trial.set_user_attr("fold_high_proba_means",fold_high_proba_means)
        early_stop_alert = any(bi < 100 and bi >= 0 for bi in fold_best_iters)
        trial.set_user_attr("cb_early_stop_alert",  early_stop_alert)
        fold_wf1s = [
            round(f1_score(y[fd["val_idx"]], np.argmax(oof[fd["val_idx"]], 1), average="weighted"), 5)
            for fd in fold_cache
        ]
        trial.set_user_attr("fold_metrics_summary", fold_wf1s)
        return wf1

    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=10)
    study  = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[_make_hp_progress_callback("CB")])

    pruned_count = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    completed = [t for t in study.trials if t.value is not None]
    abs_best  = max(completed, key=lambda t: t.value)
    best_wf1  = study.best_value
    band      = [t for t in completed if t.value >= best_wf1 - WF1_BAND]
    hrec_best = max(band, key=lambda t: t.user_attrs.get("high_recall", 0.0))

    # cb_early_stop_alert: en az bir trial'da herhangi fold < 100 olmuş mu?
    alert_trials = [t for t in completed if t.user_attrs.get("cb_early_stop_alert", False)]
    cb_early_stop_global = len(alert_trials) > 0

    print(f"\n  ✓ CB HP tamamlandı")
    print(f"    Toplam trial    : {len(study.trials)}  (pruned={pruned_count})")
    print(f"    [ABS BEST WF1]  trial={abs_best.number}  WF1={abs_best.value:.5f}  "
          f"HRec={abs_best.user_attrs.get('high_recall', 0):.5f}  params={abs_best.params}")
    print(f"    [BAND HRec MAX] trial={hrec_best.number}  WF1={hrec_best.value:.5f}  "
          f"HRec={hrec_best.user_attrs.get('high_recall', 0):.5f}  params={hrec_best.params}")
    if abs_best.number != hrec_best.number:
        print(f"    → WF1 band içinde daha iyi HRec: trial {hrec_best.number} seçildi")
    if cb_early_stop_global:
        print(f"  ⚠ [CB UYARI] {len(alert_trials)} trial'da en az 1 fold bestIteration < 100 "
              f"→ cb_early_stop_alert=True")
    sel_iters = hrec_best.user_attrs.get("fold_best_iterations", [])
    print(f"    Seçilen trial fold bestIter: {sel_iters}")

    # Dönerken cb_early_stop_alert bilgisini params içine göm (evidence_pack için)
    result_params = get_catboost_params(hrec_best.params)
    result_params["_cb_early_stop_alert"] = cb_early_stop_global
    result_params["_band_hrec_trial"]     = hrec_best.number
    result_params["_abs_best_wf1_trial"]  = abs_best.number
    return result_params


# =============================================================================
# BÖLÜM 8 — FAZ 2: ENSEMBLE AĞIRLIK OPTİMİZASYONU  [V13-F2: 50 trial + HRec MAX]
# =============================================================================

def optuna_weight_search(
    cb_oof   : np.ndarray,
    lgbm_oof : np.ndarray,
    xgb_oof  : np.ndarray,
    y        : np.ndarray,
    n_trials : int = OPTUNA_BLEND_TRIALS,
) -> tuple:
    """
    [V13-F2] Ensemble ağırlık arama — 50 trial, MAX HRec from band.
    [BUG FIX] V12.1: ilk geçen alpha → V13: MAX HRec'li seçilir.
    """
    def objective(trial: optuna.Trial) -> float:
        raw = np.array([
            trial.suggest_float("w_cb",   0.0, 1.0),
            trial.suggest_float("w_lgbm", 0.0, 1.0),
            trial.suggest_float("w_xgb",  0.0, 1.0),
        ])
        w = raw / (raw.sum() + 1e-12)
        blended = w[0]*cb_oof + w[1]*lgbm_oof + w[2]*xgb_oof
        preds   = np.argmax(blended, 1)

        wf1    = f1_score(y, preds, average="weighted")
        hrec   = recall_score(y, preds, labels=[2], average="macro")
        hf1    = f1_score(y, preds, labels=[2], average="macro")
        phc    = int(np.sum(preds == 2))
        med_f1 = f1_score(y, preds, labels=[1], average="macro")

        trial.set_user_attr("high_recall",     hrec)
        trial.set_user_attr("high_f1",         hf1)
        trial.set_user_attr("pred_high_count", phc)
        trial.set_user_attr("medium_f1",       med_f1)
        return wf1

    def dual_early_stop(study: optuna.Study, trial: optuna.Trial) -> None:
        if len(study.trials) < EARLY_STOP_PATIENCE:
            return
        recent   = study.trials[-EARLY_STOP_PATIENCE:]
        best_wf1 = study.best_value
        band_broken = all(t.value is not None and t.value < best_wf1 - WF1_BAND for t in recent)
        hrecs       = [t.user_attrs.get("high_recall", 0.0) for t in recent]
        trend_flat  = (max(hrecs) - min(hrecs)) < 0.005
        if band_broken and trend_flat:
            print(f"  [EARLY STOP] Trial {trial.number}: band_broken AND trend_flat")
            study.stop()

    def select_best_v13(study: optuna.Study) -> optuna.Trial:
        """
        [V13-F2 BUG FIX] Seçim hiyerarşisi:
        1. WF1 >= best-0.001 bandı
        2. HRec max (primary)
        3. Tiebreak: pred_high_count daha kontrollü (baseline'a yakın)
        4. Tiebreak: daha dengeli weight dağılımı (entropy max)
        """
        best_wf1    = study.best_value
        phc_baseline = int(np.sum(np.argmax(
            (1/3)*cb_oof + (1/3)*lgbm_oof + (1/3)*xgb_oof, 1) == 2))

        band = [t for t in study.trials
                if t.value is not None and t.value >= best_wf1 - WF1_BAND]

        # Sort: HRec desc, then phc_dist asc, then weight_entropy desc
        def sort_key(t):
            hrec = t.user_attrs.get("high_recall", 0.0)
            phc  = t.user_attrs.get("pred_high_count", phc_baseline)
            phc_dist = abs(phc - phc_baseline)
            # weight entropy (daha dengeli = yüksek entropy)
            raw = np.array([t.params.get("w_cb", 0.33),
                            t.params.get("w_lgbm", 0.33),
                            t.params.get("w_xgb", 0.33)])
            w = raw / (raw.sum() + 1e-12)
            entropy = -np.sum(w * np.log(w + 1e-12))
            return (hrec, -phc_dist, entropy)

        best_t = max(band, key=sort_key)
        return best_t

    print(f"\n{'═' * 72}")
    print(f"  FAZ 2 — ENSEMBLE AĞIRLIK OPTİMİZASYONU ({n_trials} trial)")
    print(f"    PRIMARY=WF1, seçim=band({WF1_BAND}) içinden HRec MAX")
    print(f"{'═' * 72}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials,
                   callbacks=[dual_early_stop, _make_hp_progress_callback("BLEND", log_every=10)],
                   show_progress_bar=False)

    best_trial = select_best_v13(study)
    best       = best_trial.params
    raw  = np.array([best["w_cb"], best["w_lgbm"], best["w_xgb"]])
    bw   = raw / raw.sum()

    phc_best = best_trial.user_attrs.get("pred_high_count", 0)
    raw_w    = np.array([best["w_cb"], best["w_lgbm"], best["w_xgb"]])

    # Dominant model tespiti
    dominant_idx   = np.argmax(bw)
    dominant_names = ["CB", "LGBM", "XGB"]
    dominant_model = dominant_names[dominant_idx]
    dominant_note  = (f"⚠ Blend fiilen tek modele yakın: {dominant_model}={bw[dominant_idx]:.3f}"
                      if bw[dominant_idx] > 0.70 else
                      f"Dengeli blend ({dominant_model} baskın)")

    info = {
        "w_cb"              : float(bw[0]),
        "w_lgbm"            : float(bw[1]),
        "w_xgb"             : float(bw[2]),
        "raw_w_cb"          : float(raw_w[0]),
        "raw_w_lgbm"        : float(raw_w[1]),
        "raw_w_xgb"         : float(raw_w[2]),
        "weighted_f1"       : best_trial.value,
        "high_recall"       : best_trial.user_attrs.get("high_recall", 0.0),
        "pred_high_count"   : phc_best,
        "total_trials"      : len(study.trials),
        "dominant_model"    : dominant_model,
        "dominant_note"     : dominant_note,
        "band_candidates_n" : len([t for t in study.trials
                                   if t.value is not None
                                   and t.value >= study.best_value - WF1_BAND]),
    }

    print(f"  ✓ Ensemble Optuna tamamlandı ({len(study.trials)} trial)")
    print(f"    Ağırlıklar : CB={bw[0]:.4f}  LGBM={bw[1]:.4f}  XGB={bw[2]:.4f}")
    print(f"    WF1        : {best_trial.value:.5f}")
    print(f"    HRec       : {best_trial.user_attrs.get('high_recall', 0):.5f}")
    print(f"    phc        : {phc_best}")
    print(f"    {dominant_note}")
    return bw, info


# =============================================================================
# BÖLÜM 9 — FAZ 3: FAZ 0 TEST PAKETİ  [V13-F3: iki bug fix]
# =============================================================================

def check_cb_calibration(cb_oof: np.ndarray, y: np.ndarray) -> dict:
    """[P0-D] CB proba kalibrasyon kontrolü — V12.1'den korundu."""
    from sklearn.calibration import calibration_curve
    cb_high_prob = cb_oof[:, 2]
    y_binary     = (y == 2).astype(int)
    n_high       = y_binary.sum()
    n_bins       = min(10, max(3, n_high // 30))
    try:
        prob_true, prob_pred = calibration_curve(y_binary, cb_high_prob, n_bins=n_bins)
    except ValueError:
        return {"ece": None, "bias": "unknown", "threshold_suggestion": 0.35, "n_bins": n_bins}

    bin_weights = np.histogram(cb_high_prob, bins=n_bins, range=(0,1))[0] / len(cb_high_prob)
    ece  = float(np.sum(np.abs(prob_true - prob_pred) * bin_weights[:len(prob_true)]))
    bias_raw = float(np.mean(prob_pred - prob_true))
    bias = "inflated" if bias_raw > 0.02 else ("deflated" if bias_raw < -0.02 else "calibrated")
    threshold_suggestion = (
        round(min(0.50, 0.35 + bias_raw), 2) if bias == "inflated" else
        round(max(0.20, 0.35 + bias_raw), 2) if bias == "deflated" else 0.35
    )
    result = {
        "ece": round(ece, 4), "bias": bias,
        "bias_raw": round(bias_raw, 4),
        "threshold_suggestion": threshold_suggestion, "n_bins": n_bins,
    }
    print(f"\n  [P0-D] CB PROBA KALİBRASYON KONTROLÜ")
    print(f"    ECE  : {ece:.4f}  {'✓ İYİ' if ece < 0.05 else ('⚠ ORTA' if ece < 0.10 else '✗ KÖTÜ')}")
    print(f"    Bias : {bias}  (raw={bias_raw:+.4f})")
    print(f"    Öneri thresh: {threshold_suggestion}")
    return result


def faz0_medium_baseline(blend_oof, y):
    preds = np.argmax(blend_oof, axis=1)
    baseline = {
        "MediumF1_base"      : f1_score(y, preds, labels=[1], average="macro"),
        "MediumRec_base"     : recall_score(y, preds, labels=[1], average="macro"),
        "MediumPre_base"     : precision_score(y, preds, labels=[1], average="macro"),
        "pred_high_baseline" : int(np.sum(preds == 2)),
    }
    print(f"\n  [FAZ0][MEDIUM_BASELINE]")
    print(f"    MediumF1_base      = {baseline['MediumF1_base']:.5f}")
    print(f"    MediumRec_base     = {baseline['MediumRec_base']:.5f}")
    print(f"    pred_high_baseline = {baseline['pred_high_baseline']}")
    print(f"    pred_high_tavan    = {int(baseline['pred_high_baseline'] * PRED_HIGH_MULT_MAX)}")
    return baseline


def faz0_check_guardrails(preds, y, baseline, label=""):
    wf1     = f1_score(y, preds, average="weighted")
    hrec    = recall_score(y, preds, labels=[2], average="macro")
    hpre    = precision_score(y, preds, labels=[2], average="macro")
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


def faz0_recoverable_high(cb_oof, blend_oof, y):
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
    print(f"    CB'nin kurtardığı       : {recoverable} ({recoverable/max(n_missed,1)*100:.1f}%)")
    print(f"    → ROTA                  : {rota}")
    return {"n_missed": n_missed, "recoverable": recoverable, "rota": rota}


def faz0_alpha_sweep(blend_oof, y, baseline, alphas=None):
    """
    [V13-F3 BUG-A] Alpha seçimi: WF1 >= eşik bandında HRec MAX seç.
    V12.1 bug: ilk geçen alpha seçiliyordu (best_alpha is None kontrolü).
    V13 fix: tüm geçen alpha'ları topla, en yüksek HRec'lisi seç.
    """
    global ALPHA_ACCEPT_WF1
    if alphas is None:
        alphas = [1.00, 1.03, 1.05, 1.08, 1.10, 1.12, 1.15]

    print(f"\n  [FAZ0][ALPHA_SWEEP] Baseline proba üzerinde koşuluyor...")
    print(f"    Kabul: WF1>={ALPHA_ACCEPT_WF1:.5f} VE HRec>={ALPHA_ACCEPT_HREC}")
    print(f"    [V13] Seçim: MAX HRec'li geçen alpha (ilk geçen değil)")
    print(f"    {'alpha':>6} | {'WF1':>8} | {'HRec':>8} | {'HPre':>8} | {'phc':>6} | {'MedF1':>8} | karar")
    print(f"    {'─'*72}")

    passing_results = []   # [V13] tüm kabul edilenler biriktirilir

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

        if accept:
            passing_results.append((alpha, res))   # [V13] biriktir

    # [V13 BUG FIX + tiebreak] ilk geçen değil, MAX HRec; eşitlikte phc daha kontrollü
    if passing_results:
        # phc baseline (alpha=1.00 / blend baseline)
        base_preds = np.argmax(blend_oof, axis=1)
        phc_base   = int(np.sum(base_preds == 2))

        def alpha_sort_key(item):
            _alpha, res = item
            hrec     = res["hrec"]
            phc_dist = abs(res["phc"] - phc_base)
            return (hrec, -phc_dist)

        best_alpha, best_result = max(passing_results, key=alpha_sort_key)
        first_alpha = passing_results[0][0]
        if best_alpha != first_alpha:
            print(f"\n  [V13] BUG FIX: α={first_alpha:.2f} (ilk geçen, V12.1 seçimi) yerine "
                  f"α={best_alpha:.2f} seçildi (MAX HRec={best_result['hrec']:.5f})")
        print(f"\n  → En iyi α: {best_alpha}  "
              f"(HRec={best_result['hrec']:.5f}, phc={best_result['phc']})  → SUBMISSION ATA")
    else:
        best_alpha, best_result = None, None
        print(f"\n  → Hiçbir alpha kabul edilmedi. Faz 1'e geç.")

    return {"best_alpha": best_alpha, "best_result": best_result, "passing_count": len(passing_results)}


def faz0_gated_sweep(cb_oof, blend_oof, y, baseline, t_cb_list=None, t_blend_list=None):
    """
    [V13-F3 BUG-B] Proba-tabanlı çift-threshold gated rescue.
    V12.1 bug: gate = (argmax(cb_oof)==2) & (blend[:,2]>t)
               → argmax, CB'nin çok düşük probayı "en iyi sınıf" olarak seçmesine dayanır.
               → CB Fold 1 unstable olduğunda TPR=0.
    V13 fix: gate = (cb_oof[:,2] > t_cb) AND (blend[:,2] > t_blend)
               → Her iki model de High için yeterli proba üretmelidir.
    """
    if t_cb_list is None:
        # [V13 Nihai] [0.20, 0.70] — geniş arama (proba-tabanlı gate için)
        t_cb_list = [round(t, 2) for t in np.arange(0.20, 0.75, 0.05)]
    if t_blend_list is None:
        t_blend_list = [round(t, 2) for t in np.arange(0.20, 0.75, 0.05)]

    total_combinations = len(t_cb_list) * len(t_blend_list)
    print(f"\n  [FAZ0][GATED_SWEEP] Proba-tabanlı çift-threshold sweep...")
    print(f"    [V13] Gate = (cb_oof[:,2] > t_cb) AND (blend[:,2] > t_blend)")
    print(f"    Arama yöntemi: grid search  |  denenen kombinasyon: {total_combinations}")
    print(f"    t_cb ∈ [{t_cb_list[0]}, {t_cb_list[-1]}]  t_blend ∈ [{t_blend_list[0]}, {t_blend_list[-1]}]")
    print(f"    Kriter: FPR < {GATE_FPR_MAX} VE TPR >= {GATE_TPR_MIN}")
    print(f"    {'t_cb':>6} | {'t_blend':>8} | {'TPR':>8} | {'FPR':>8} | {'WF1':>8} | {'phc':>6} | karar")
    print(f"    {'─'*72}")

    blend_preds   = np.argmax(blend_oof, axis=1)
    missed_mask   = (y == 2) & (blend_preds != 2)
    non_high_mask = (y != 2)

    best_threshold = None
    best_result    = None
    best_tpr       = 0.0   # tiebreaker
    tried_count    = 0

    for t_cb in t_cb_list:
        for t_blend in t_blend_list:
            tried_count += 1
            # [V13 BUG FIX] Proba-tabanlı gate
            gate = (cb_oof[:, 2] > t_cb) & (blend_oof[:, 2] > t_blend)

            tpr = float(gate[missed_mask].mean())   if missed_mask.any() else 0.0
            fpr = float(gate[non_high_mask].mean()) if non_high_mask.any() else 1.0

            gated_blend = blend_oof.copy()
            gated_blend[gate, 2] = np.maximum(gated_blend[gate, 2], 0.50)
            gated_blend /= gated_blend.sum(axis=1, keepdims=True)
            preds = np.argmax(gated_blend, axis=1)

            res    = faz0_check_guardrails(preds, y, baseline, label="")
            accept = (fpr < GATE_FPR_MAX) and (tpr >= GATE_TPR_MIN) and (res["karar"] == "GEÇTİ")
            tag    = "✓ KABUL" if accept else "✗"

            print(f"    {t_cb:>6.2f} | {t_blend:>8.2f} | {tpr:>8.3f} | {fpr:>8.3f} | "
                  f"{res['wf1']:>8.5f} | {res['phc']:>6} | {tag}")

            # En iyi kabul: WF1 max (tiebreaker: TPR max)
            if accept:
                if best_threshold is None or res["wf1"] > (best_result["wf1"] if best_result else 0):
                    best_threshold = (t_cb, t_blend)
                    best_result    = res
                    best_tpr       = tpr

    if best_threshold:
        print(f"\n  → En iyi: t_cb={best_threshold[0]}, t_blend={best_threshold[1]} "
              f"(WF1={best_result['wf1']:.5f}, TPR={best_tpr:.3f})  → GATED RESCUE KULLAN")
    else:
        print(f"\n  → Gated rescue için uygun threshold bulunamadı.")
    print(f"    Toplam denenen kombinasyon: {tried_count}")

    return {
        "best_threshold"             : best_threshold,
        "best_result"                : best_result,
        "thresholds_tried_count"     : tried_count,
        "search_method"              : "grid",
    }


# =============================================================================
# BÖLÜM 10 — FAZ 4: SEED STABİLİTESİ  [V13-F4]
# =============================================================================

def run_seed_stability(
    X_raw      : pd.DataFrame,
    y          : np.ndarray,
    X_test     : pd.DataFrame,
    folds      : list,
    lgbm_params: dict,
    xgb_params : dict,
    cb_params  : dict,
    seeds      : list = SEED_STABILITY_SEEDS,
) -> dict:
    """
    [V13-F4] Seed stabilitesi: 3 seed × aynı fold freeze → varyans raporu.
    Karar kriteri: std > 0.005 → tek seed kararına güvenme.
    """
    print(f"\n{'═' * 72}")
    print(f"  FAZ 4 — SEED STABİLİTESİ ({seeds})")
    print(f"  Fold freeze: {FOLD_INDEX_PATH}")
    print(f"{'═' * 72}")

    seed_results = []

    for seed in seeds:
        print(f"\n  ▶ Seed={seed} çalıştırılıyor...")

        # Seed'e özgü fold_cache (FE+CE seed ile)
        fold_cache_s = precompute_fold_data(X_raw, y, X_test, folds, seed=seed)

        # Sadece OOF metrikler — test submission gerekmez
        (cb_oof_s, _,
         lgbm_oof_s, _,
         xgb_oof_s,  _,
         cb_sc, lgbm_sc, xgb_sc,
         cb_bi_s) = train_oof_triple(
            X_raw=X_raw, y=y, X_test=X_test,
            lgbm_params=lgbm_params,
            xgb_params=xgb_params,
            cb_params=cb_params,
            seed=seed,
            fold_cache=fold_cache_s,
        )

        blend_s = (cb_oof_s + lgbm_oof_s + xgb_oof_s) / 3
        preds_s = np.argmax(blend_s, 1)
        wf1_s   = f1_score(y, preds_s, average="weighted")
        hrec_s  = recall_score(y, preds_s, labels=[2], average="macro")
        phc_s   = int(np.sum(preds_s == 2))

        seed_results.append({
            "seed": seed, "wf1": wf1_s, "hrec": hrec_s, "phc": phc_s,
            "lgbm_wf1": float(np.mean(lgbm_sc)), "cb_wf1": float(np.mean(cb_sc)),
        })
        print(f"  ✓ Seed={seed}: blend_WF1={wf1_s:.5f}  HRec={hrec_s:.5f}  phc={phc_s}")

    wf1_vals  = [r["wf1"]  for r in seed_results]
    hrec_vals = [r["hrec"] for r in seed_results]
    phc_vals  = [r["phc"]  for r in seed_results]
    report = {
        "seeds"              : seeds,
        "wf1_per_seed"       : wf1_vals,
        "hrec_per_seed"      : hrec_vals,
        "phc_per_seed"       : phc_vals,
        "wf1_mean"           : float(np.mean(wf1_vals)),
        "wf1_std"            : float(np.std(wf1_vals)),
        "hrec_mean"          : float(np.mean(hrec_vals)),
        "hrec_std"           : float(np.std(hrec_vals)),
        "phc_mean"           : float(np.mean(phc_vals)),
        "phc_std"            : float(np.std(phc_vals)),
        "instability_flag"   : False,
    }

    print(f"\n  {'═' * 60}")
    print(f"  SEED STABİLİTESİ RAPORU")
    print(f"  {'═' * 60}")
    print(f"  WF1  : {report['wf1_mean']:.5f} ± {report['wf1_std']:.5f}")
    print(f"  HRec : {report['hrec_mean']:.5f} ± {report['hrec_std']:.5f}")
    print(f"  phc  : {report['phc_mean']:.1f} ± {report['phc_std']:.1f}")
    print(f"  Seed bazlı:")
    for r in seed_results:
        print(f"    seed={r['seed']}: WF1={r['wf1']:.5f}  HRec={r['hrec']:.5f}  phc={r['phc']}")
    if report["wf1_std"] > SEED_STABILITY_STD_WARN:
        report["instability_flag"] = True
        print(f"  ⚠ [UYARI] WF1 std={report['wf1_std']:.5f} > {SEED_STABILITY_STD_WARN} "
              f"→ Model seed'e duyarlı. Tek seed kararına güvenme!")
    else:
        print(f"  ✓ WF1 std={report['wf1_std']:.5f} < {SEED_STABILITY_STD_WARN} → Model kararlı.")
    if report["hrec_std"] > SEED_STABILITY_STD_WARN:
        report["instability_flag"] = True
        print(f"  ⚠ [UYARI] HRec std={report['hrec_std']:.5f} > {SEED_STABILITY_STD_WARN} "
              f"→ High recall seed'e duyarlı.")

    return report


# =============================================================================
# BÖLÜM 11 — RAPORLAMA VE SUBMISSION  ← V12.1 ile ÖZDEŞ
# =============================================================================

def report_oof_comparison(y, cb_oof, lgbm_oof, xgb_oof, optuna_oof,
                            cb_scores, lgbm_scores, xgb_scores, best_info):
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
            "hrec"     : recall_score(y, preds, labels=[2], average="macro"),
            "fold_avg" : np.mean(scores) if scores else float("nan"),
            "fold_std" : np.std(scores)  if scores else float("nan"),
        }

    best_wf1 = max(rows, key=lambda k: rows[k]["wf1"])

    print("\n" + "═" * 76)
    print("  V13 KARŞILAŞTIRMALI OOF PERFORMANS ÖZETİ")
    print("═" * 76)
    print(f"  {'Model':<16} {'WF1':>10}  {'HRec':>10}  {'Fold Ort':>10}  {'Fold Std':>9}")
    print(f"  {'─'*16}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*9}")

    for name, m in rows.items():
        fa  = f"{m['fold_avg']:.5f}" if not np.isnan(m['fold_avg']) else "   —   "
        fs  = f"{m['fold_std']:.5f}" if not np.isnan(m['fold_std']) else "   —   "
        tag = "  ← Zindi" if name == "Optuna Blend" else (
              "  ← BEST"  if name == best_wf1 and name != "Optuna Blend" else "")
        print(f"  {name:<16}  {m['wf1']:>10.5f}  {m['hrec']:>10.5f}  {fa:>10}  {fs:>9}{tag}")

    print("═" * 76)
    print(f"\n  Optuna Ağırlıkları: "
          f"CB={best_info['w_cb']:.4f}  "
          f"LGBM={best_info['w_lgbm']:.4f}  "
          f"XGB={best_info['w_xgb']:.4f}")

    optuna_preds = np.argmax(optuna_oof, 1)
    print("\n  [Optuna Blend] Sınıf Bazlı OOF Raporu:")
    print(classification_report(y, optuna_preds, target_names=CLASS_NAMES, zero_division=0))


def create_submission(test_ids, test_proba, sub_path, output_path, label=""):
    pred_idx    = np.argmax(test_proba, 1)
    pred_labels = np.array([TARGET_INVERSE_MAPPING[i] for i in pred_idx])
    if sub_path is not None and Path(sub_path).exists():
        _p = Path(sub_path)
        sample_sub = pd.read_excel(_p) if _p.suffix.lower() in ['.xls', '.xlsx'] else pd.read_csv(_p)
        submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: pred_labels})
        submission = sample_sub[[ID_COL]].merge(submission, on=ID_COL, how="left")
    else:
        submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: pred_labels})
    submission.to_csv(output_path, index=False)
    tag = f"[{label}] " if label else ""
    print(f"\n  ✓ {tag}Submission → {output_path}")
    print(f"    {submission[TARGET_COL].value_counts().to_string()}")
    return submission


# =============================================================================
# BÖLÜM 12 — ANA AKIŞ  [V13: 4 Fazlı]
# =============================================================================

if __name__ == "__main__":

    print("\n" + "█" * 72)
    print("  V13 DEEP OPTUNA — 4 FAZLI MİMARİ")
    print("  Hedef: WF1 > 0.862 VE HRec >= 0.69")
    print("  Taban: LGBM OOF WF1 = 0.86066")
    print("█" * 72)

    # ── 1. Veri oku ──────────────────────────────────────────────────────────
    X_raw, y, X_test, test_ids = read_data(
        train_path = DATA_DIR / "Train.csv",
        test_path  = DATA_DIR / "Test.csv",
    )

    # ── 2. Fold freeze yükle / oluştur ───────────────────────────────────────
    folds = get_or_create_folds(X_raw, y)

    # ── 3. [V13-PERF] Fold verisi ön-hesaplama (seed=42) ─────────────────────
    print(f"\n  ▶ [V13-PERF] Tüm HP study'ler için fold verisi ön-hesaplanıyor...")
    fold_cache_main = precompute_fold_data(X_raw, y, X_test, folds, seed=SEED)

    # ── 4. FAZ 1: HİPERPARAMETRE OPTİMİZASYONU ──────────────────────────────
    print(f"\n{'█' * 72}")
    print(f"  FAZ 1 — 3 BAĞIMSIZ HİPERPARAMETRE STUDY")
    print(f"{'█' * 72}")

    best_lgbm_params = optuna_hyper_lgbm(
        fold_cache=fold_cache_main, y=y, n_trials=OPTUNA_HP_TRIALS_LGBM
    )
    best_xgb_params = optuna_hyper_xgb(
        fold_cache=fold_cache_main, y=y, n_trials=OPTUNA_HP_TRIALS_XGB
    )
    best_cb_params = optuna_hyper_cb(
        fold_cache=fold_cache_main, y=y, n_trials=OPTUNA_HP_TRIALS_CB
    )

    print(f"\n  FAZ 1 ÖZET — EN İYİ PARAMETRELER:")
    print(f"  LGBM: {best_lgbm_params}")
    print(f"  XGB : {best_xgb_params}")
    print(f"  CB  : {best_cb_params}")

    # ── 5. FAZ 1.5: En iyi parametrelerle tam eğitim (submission için) ────────
    print(f"\n{'█' * 72}")
    print(f"  FAZ 1.5 — EN İYİ PARAMS İLE TAM EĞİTİM")
    print(f"{'█' * 72}")

    (cb_oof,   cb_test,
     lgbm_oof, lgbm_test,
     xgb_oof,  xgb_test,
     cb_scores, lgbm_scores, xgb_scores,
     cb_best_iters) = train_oof_triple(
        X_raw=X_raw, y=y, X_test=X_test,
        lgbm_params=best_lgbm_params,
        xgb_params=best_xgb_params,
        cb_params=best_cb_params,
        seed=SEED,
        fold_cache=fold_cache_main,
    )

    # ── 6. FAZ 2: ENSEMBLE AĞIRLIK (50 trial) ─────────────────────────────────
    print(f"\n{'█' * 72}")
    print(f"  FAZ 2 — ENSEMBLE AĞIRLIK OPTİMİZASYONU")
    print(f"{'█' * 72}")

    best_weights, best_info = optuna_weight_search(
        cb_oof=cb_oof, lgbm_oof=lgbm_oof, xgb_oof=xgb_oof, y=y,
        n_trials=OPTUNA_BLEND_TRIALS,
    )
    w_cb, w_lgbm, w_xgb = best_weights

    optuna_oof  = w_cb * cb_oof   + w_lgbm * lgbm_oof  + w_xgb * xgb_oof
    optuna_test = w_cb * cb_test  + w_lgbm * lgbm_test + w_xgb * xgb_test
    equal_test  = (cb_test + lgbm_test + xgb_test) / 3

    # ── 7. Karşılaştırmalı OOF raporu ────────────────────────────────────────
    report_oof_comparison(
        y=y, cb_oof=cb_oof, lgbm_oof=lgbm_oof, xgb_oof=xgb_oof,
        optuna_oof=optuna_oof, cb_scores=cb_scores,
        lgbm_scores=lgbm_scores, xgb_scores=xgb_scores, best_info=best_info,
    )

    # ── 8. FAZ 3: FAZ 0 ──────────────────────────────────────────────────────
    print(f"\n{'█' * 72}")
    print(f"  FAZ 3 — FAZ 0 (V13: iki bug fix + proba gate)")
    print(f"{'█' * 72}")

    baseline = faz0_medium_baseline(optuna_oof, y)

    # [P0-B] Dinamik eşik
    baseline_wf1     = float(f1_score(y, np.argmax(optuna_oof, 1), average="weighted"))
    ALPHA_ACCEPT_WF1 = baseline_wf1 - ALPHA_WF1_DELTA
    print(f"\n  [P0-B] Dinamik eşik: baseline_wf1={baseline_wf1:.5f} → "
          f"ALPHA_ACCEPT_WF1={ALPHA_ACCEPT_WF1:.5f}")

    # CB kalibrasyon
    calib_result = check_cb_calibration(cb_oof, y)

    # Recoverable High
    rh_result = faz0_recoverable_high(cb_oof, optuna_oof, y)

    # Alpha sweep [V13 BUG-A FIX]
    alpha_result = faz0_alpha_sweep(optuna_oof, y, baseline)

    # Gated sweep [V13 BUG-B FIX] — varsayılan [0.20,0.70] grid kullanılır
    if rh_result["recoverable"] >= 40:
        gate_result = faz0_gated_sweep(cb_oof, optuna_oof, y, baseline)
    else:
        print(f"\n  [FAZ0][GATED_SWEEP] Atlandı: recoverable={rh_result['recoverable']} < 40")
        gate_result = {
            "best_threshold": None, "best_result": None,
            "thresholds_tried_count": 0, "search_method": "skipped",
        }

    # Faz 3 özeti
    print(f"\n{'═' * 72}")
    print(f"  FAZ 3 KARAR ÖZETİ")
    print(f"{'═' * 72}")
    print(f"  Recoverable High : {rh_result['recoverable']}/{rh_result['n_missed']} → ROTA: {rh_result['rota']}")
    print(f"  En iyi α         : {alpha_result['best_alpha']}  "
          f"({'HRec=' + str(round(alpha_result['best_result']['hrec'], 5)) if alpha_result['best_result'] else 'YOK'})")
    print(f"  Gated threshold  : {gate_result['best_threshold']}")

    if alpha_result["best_alpha"] is not None:
        print(f"\n  ✓ KARAR: α={alpha_result['best_alpha']} ile SUBMISSION AT")
    elif gate_result["best_threshold"] is not None:
        t_cb, t_blend = gate_result["best_threshold"]
        print(f"\n  ✓ KARAR: Gated rescue (t_cb={t_cb}, t_blend={t_blend}) uygula")
    else:
        print(f"\n  → Faz 0 geçemedi. Optuna blend gönder.")

    # ── 9. FAZ 4: SEED STABİLİTESİ ───────────────────────────────────────────
    print(f"\n{'█' * 72}")
    print(f"  FAZ 4 — SEED STABİLİTESİ")
    print(f"{'█' * 72}")

    seed_report = run_seed_stability(
        X_raw=X_raw, y=y, X_test=X_test,
        folds=folds,
        lgbm_params=best_lgbm_params,
        xgb_params=best_xgb_params,
        cb_params=best_cb_params,
        seeds=SEED_STABILITY_SEEDS,
    )

    # ── 10. Submission dosyaları ──────────────────────────────────────────────
    sub_path = DATA_DIR / "SampleSubmission.csv"
    if not sub_path.exists():
        print(f"  ⚠ SampleSubmission.csv bulunamadı → doğrudan ID+Target üretiliyor")
        sub_path = None

    create_submission(test_ids, cb_test,     sub_path, OUTPUT_DIR / "submission_v13_catboost.csv",    "CatBoost")
    create_submission(test_ids, lgbm_test,   sub_path, OUTPUT_DIR / "submission_v13_lgbm.csv",        "LightGBM")
    create_submission(test_ids, xgb_test,    sub_path, OUTPUT_DIR / "submission_v13_xgboost.csv",     "XGBoost")
    create_submission(test_ids, equal_test,  sub_path, OUTPUT_DIR / "submission_v13_equal_blend.csv", "1/3 Eşit")
    create_submission(test_ids, optuna_test, sub_path, OUTPUT_DIR / "submission_v13_optuna_blend.csv","Optuna Blend ← ANA")

    # ── 11. Artefakt kaydet ───────────────────────────────────────────────────
    for name, arr in {
        "oof_cb_v13"     : cb_oof,    "oof_lgbm_v13" : lgbm_oof,
        "oof_xgb_v13"    : xgb_oof,   "oof_optuna_v13": optuna_oof,
        "test_cb_v13"    : cb_test,   "test_lgbm_v13": lgbm_test,
        "test_xgb_v13"   : xgb_test,  "test_optuna_v13": optuna_test,
        "y_true_v13"     : y,
    }.items():
        np.save(OUTPUT_DIR / f"{name}.npy", arr)

    # cb_early_stop_alert: best_cb_params içine gömdük, çıkar
    cb_early_stop_alert = best_cb_params.pop("_cb_early_stop_alert", False)
    cb_band_hrec_trial  = best_cb_params.pop("_band_hrec_trial", None)
    cb_abs_best_trial   = best_cb_params.pop("_abs_best_wf1_trial", None)

    # Final recommendation note
    if alpha_result["best_alpha"] is not None:
        final_rec = (f"α={alpha_result['best_alpha']} ile alpha-swept submission. "
                     f"HRec={alpha_result['best_result']['hrec']:.5f}")
    elif gate_result["best_threshold"] is not None:
        t_cb_f, t_bl_f = gate_result["best_threshold"]
        final_rec = (f"Gated rescue (t_cb={t_cb_f}, t_blend={t_bl_f}). "
                     f"WF1={gate_result['best_result']['wf1']:.5f}")
    else:
        final_rec = "Faz 0 geçemedi. Optuna blend gönder."
    if seed_report["instability_flag"]:
        final_rec += " ⚠ SEED INSTABILITY DETECTED — sonuçları doğrula."

    evidence = {
        "version"               : "v13_nihaisürum",
        "architecture"          : "4-phase decoupled Optuna",
        "baseline_wf1"          : round(baseline_wf1, 5),
        "alpha_accept_wf1"      : round(ALPHA_ACCEPT_WF1, 5),
        # FAZ 1
        "faz1_best_params"      : {
            "lgbm": best_lgbm_params,
            "xgb" : best_xgb_params,
            "cb"  : best_cb_params,
        },
        "faz1_cb_abs_best_trial"   : cb_abs_best_trial,
        "faz1_cb_band_hrec_trial"  : cb_band_hrec_trial,
        "cb_early_stop_alert"      : cb_early_stop_alert,
        # FAZ 2
        "faz2_ensemble"         : {
            "w_cb"              : float(w_cb),
            "w_lgbm"            : float(w_lgbm),
            "w_xgb"             : float(w_xgb),
            "dominant_model"    : best_info.get("dominant_model", "?"),
            "dominant_note"     : best_info.get("dominant_note", ""),
            "band_candidates_n" : best_info.get("band_candidates_n", -1),
            **{k: float(v) if isinstance(v, (float, np.floating)) else v
               for k, v in best_info.items()
               if k not in ("dominant_model", "dominant_note", "band_candidates_n")},
        },
        # FAZ 3 OOF
        "faz3_oof_metrics"      : {
            "cb_wf1"       : round(f1_score(y, np.argmax(cb_oof, 1),    average="weighted"), 5),
            "lgbm_wf1"     : round(f1_score(y, np.argmax(lgbm_oof, 1),  average="weighted"), 5),
            "xgb_wf1"      : round(f1_score(y, np.argmax(xgb_oof, 1),   average="weighted"), 5),
            "blend_wf1"    : round(f1_score(y, np.argmax(optuna_oof, 1), average="weighted"), 5),
            "blend_hrec"   : round(recall_score(y, np.argmax(optuna_oof, 1), labels=[2], average="macro"), 5),
            "cb_fold_std"  : round(float(np.std(cb_scores)),   5),
            "lgbm_fold_std": round(float(np.std(lgbm_scores)), 5),
            "xgb_fold_std" : round(float(np.std(xgb_scores)),  5),
        },
        "cb_best_iters"             : cb_best_iters,
        "cb_calibration"            : calib_result,
        "recoverable_high"          : rh_result,
        # FAZ 3 threshold kararları
        "phase3_best_alpha"         : alpha_result["best_alpha"],
        "phase3_alpha_passing_count": alpha_result.get("passing_count", 0),
        "phase3_best_gate_threshold": gate_result["best_threshold"],
        "phase3_thresholds_tried_count": gate_result.get("thresholds_tried_count", -1),
        "phase3_gate_search_method" : gate_result.get("search_method", "grid"),
        # FAZ 4
        "faz4_seed_report"          : seed_report,
        "instability_flag"          : seed_report["instability_flag"],
        # Baseline / guardrail
        "medium_baseline"           : {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                       for k, v in baseline.items()},
        "fold_index_path"           : str(FOLD_INDEX_PATH),
        # Final
        "final_recommendation_note" : final_rec,
        "v13_bug_fixes"             : [
            "BUG-A: alpha selection → MAX HRec from band + phc tiebreak (not first passing)",
            "BUG-B: gated gate → proba-based (cb[:,2]>t_cb AND blend[:,2]>t_blend)",
            "n_warmup_steps 1→10 in MedianPruner",
            "Dual report: abs_best_WF1 + band_best_HRec per HP study",
            "FAZ-2: tiebreak by phc_dist + weight entropy",
            "Gate search: [0.20,0.70] full grid, tried_count logged",
            "FAZ-4: phc_mean±std + instability_flag added",
        ],
    }
    with open(OUTPUT_DIR / "evidence_pack_v13.json", "w") as f:
        json.dump(evidence, f, indent=2, default=str)

    print(f"\n{'█' * 72}")
    print(f"  V13 TAMAMLANDI")
    print(f"{'█' * 72}")
    print(f"\n  OOF Sonuçları:")
    print(f"    CB       WF1 = {evidence['faz3_oof_metrics']['cb_wf1']:.5f}")
    print(f"    LGBM     WF1 = {evidence['faz3_oof_metrics']['lgbm_wf1']:.5f}")
    print(f"    XGB      WF1 = {evidence['faz3_oof_metrics']['xgb_wf1']:.5f}")
    print(f"    Blend    WF1 = {evidence['faz3_oof_metrics']['blend_wf1']:.5f}")
    print(f"\n  Seed Stabilitesi: WF1 {seed_report['wf1_mean']:.5f} ± {seed_report['wf1_std']:.5f}")
    print(f"  Faz 3 Kararı   : α={alpha_result['best_alpha']}  gate={gate_result['best_threshold']}")
    print(f"\n  Evidence Pack  → {OUTPUT_DIR}/evidence_pack_v13.json")
    print(f"  ANA Submission → {OUTPUT_DIR}/submission_v13_optuna_blend.csv")
    print(f"  LGBM Backup    → {OUTPUT_DIR}/submission_v13_lgbm.csv")
