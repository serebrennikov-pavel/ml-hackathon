from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from hackaton.eval.metric import calculate_target_metric

LOGGER = logging.getLogger(__name__)

NEW_YEAR_HOLIDAYS = {(1, d) for d in range(1, 9)}

REQUIRED_USER_COLUMNS = ["location_id", "is_strict_location", "id", "has_mk"]
REQUIRED_SHIFT_COLUMNS = [
    "id",
    "start_at",
    "location_id",
    "task_type",
    "employer_id",
    "workplace_id",
    "need_mk",
    "id_differential",
    "hours",
    "reward",
    "capacity",
]
REQUIRED_EVENT_COLUMNS = ["id", "shift_id", "user_id", "interaction", "ts"]
VALID_INTERACTIONS = {"VIEW", "APPLY", "FINISHED", "USER_CANCEL", "SYSTEM_CANCEL"}

# Финальный набор признаков (37 шт) — v2 из EDA-ноутбука.
# Базовые 31 признак после ablation (убраны month, is_weekend, day_of_month).
# Добавлено 6 новых признаков v2 для улучшения качества модели.
FEATURE_COLUMNS = [
    "hours",
    "reward",
    "reward_per_hour",
    "capacity",
    "hour",
    "weekday",
    "is_holiday",
    "view_cnt",
    "user_hist_views",
    "user_hist_applies",
    "user_hist_finished",
    "user_hist_cancels",
    "user_apply_rate",
    "user_finish_rate",
    "user_cancel_rate",
    "cum_shifts_viewed",
    "recency_days",
    "worked_long_recently",
    "same_location",
    "mk_ok",
    "is_strict_location",
    "worked_employer_before",
    "worked_workplace_before",
    "task_match",
    "emp_ctr",
    "fill_rate",
    "days_to_shift",
    "need_mk",
    "id_differential",
    "has_mk",
    "task_type",
    # Новые признаки v2
    "user_reliability_score",
    "system_cancel_cnt",
    "user_finished_employer",
    "user_finished_workplace",
    "user_task_affinity",
    "reward_delta",
]
CATEGORICAL_FEATURES = ["task_type"]
BOOL_AS_FLOAT_FEATURES = ["need_mk", "id_differential", "has_mk", "is_strict_location"]


@dataclass(frozen=True, slots=True)
class TrainConfig:
    user_path: str
    shift_path: str
    event_path: str
    output_dir: str
    random_state: int = 42
    test_ratio: float = 0.2
    skip_shap: bool = False
    shap_sample_size: int = 1000


def _to_bool(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(mapping)


def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> dict[str, object]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")
    null_counts = {c: int(df[c].isna().sum()) for c in required}
    return {
        "rows": int(len(df)),
        "required_columns_ok": True,
        "null_counts": null_counts,
    }


def _load_and_validate_data(
    cfg: TrainConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    users = pd.read_csv(cfg.user_path)
    shifts = pd.read_csv(cfg.shift_path)
    events = pd.read_csv(cfg.event_path)

    checks = {
        "user": _validate_columns(users, REQUIRED_USER_COLUMNS, "user.csv"),
        "shift": _validate_columns(shifts, REQUIRED_SHIFT_COLUMNS, "shift.csv"),
        "event": _validate_columns(events, REQUIRED_EVENT_COLUMNS, "event.csv"),
    }

    users = users[REQUIRED_USER_COLUMNS].copy()
    shifts = shifts[REQUIRED_SHIFT_COLUMNS].copy()
    events = events[REQUIRED_EVENT_COLUMNS].copy()

    users["id"] = users["id"].astype(str)
    users["location_id"] = users["location_id"].astype(str)
    users["has_mk"] = _to_bool(users["has_mk"])
    users["is_strict_location"] = _to_bool(users["is_strict_location"])

    shifts["id"] = shifts["id"].astype(str)
    shifts["location_id"] = shifts["location_id"].astype(str)
    shifts["task_type"] = shifts["task_type"].astype(str)
    shifts["employer_id"] = shifts["employer_id"].astype(str)
    shifts["workplace_id"] = shifts["workplace_id"].astype(str)
    shifts["need_mk"] = _to_bool(shifts["need_mk"])
    shifts["id_differential"] = _to_bool(shifts["id_differential"])
    shifts["hours"] = pd.to_numeric(shifts["hours"], errors="coerce")
    shifts["reward"] = pd.to_numeric(shifts["reward"], errors="coerce")
    shifts["capacity"] = pd.to_numeric(shifts["capacity"], errors="coerce")
    shifts["start_at"] = pd.to_datetime(shifts["start_at"], utc=True, errors="coerce")

    events["id"] = events["id"].astype(str)
    events["shift_id"] = events["shift_id"].astype(str)
    events["user_id"] = events["user_id"].astype(str)
    events["interaction"] = events["interaction"].astype(str).str.upper()
    events["ts"] = pd.to_datetime(events["ts"], utc=True, errors="coerce")
    events = events[events["interaction"].isin(VALID_INTERACTIONS)]

    critical = {
        "users": ["id", "location_id", "has_mk", "is_strict_location"],
        "shifts": [
            "id",
            "start_at",
            "location_id",
            "need_mk",
            "id_differential",
            "hours",
            "reward",
            "capacity",
        ],
        "events": ["id", "shift_id", "user_id", "interaction", "ts"],
    }
    users = users.dropna(subset=critical["users"]).drop_duplicates(subset=["id"])
    shifts = shifts.dropna(subset=critical["shifts"]).drop_duplicates(subset=["id"])
    # event.id в train CSV не уникален (один id — много смен); как в ноутбуке — все строки
    events = events.dropna(subset=critical["events"])

    checks["post_clean_rows"] = {
        "user": int(len(users)),
        "shift": int(len(shifts)),
        "event": int(len(events)),
    }
    return users, shifts, events, checks


def _build_training_frame(
    users: pd.DataFrame, shifts: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    shifts_info = shifts.rename(columns={"id": "shift_id"})[
        ["shift_id", "start_at", "employer_id", "workplace_id", "task_type", "hours"]
    ].copy()
    ev = events.merge(shifts_info, on="shift_id", how="inner")
    ev_before = ev[ev["ts"] < ev["start_at"]].copy()

    # Целевая переменная: из ВСЕХ событий (включая поздние APPLY/FINISHED после start_at)
    target = ev[ev["interaction"].isin(["APPLY", "FINISHED"])][["user_id", "shift_id"]].drop_duplicates()
    target["target"] = 1

    # Базовые пары user-shift из событий ДО start_at (защита от утечки)
    pairs = ev_before.groupby(["user_id", "shift_id"], as_index=False).agg(
        view_cnt=("interaction", lambda s: int((s == "VIEW").sum())),
        apply_cnt=("interaction", lambda s: int((s == "APPLY").sum())),
        finished_cnt=("interaction", lambda s: int((s == "FINISHED").sum())),
        user_cancel_cnt=("interaction", lambda s: int((s == "USER_CANCEL").sum())),
    )
    pairs = pairs.merge(target, on=["user_id", "shift_id"], how="left")
    pairs["target"] = pairs["target"].fillna(0).astype(int)
    pairs = pairs.merge(shifts_info[["shift_id", "start_at"]], on="shift_id", how="left")
    pairs = pairs.sort_values(["user_id", "start_at"]).reset_index(drop=True)

    # История пользователя (cumsum без утечки будущего)
    pairs["user_hist_views"] = pairs.groupby("user_id")["view_cnt"].cumsum() - pairs["view_cnt"]
    pairs["user_hist_applies"] = pairs.groupby("user_id")["apply_cnt"].cumsum() - pairs["apply_cnt"]
    pairs["user_hist_finished"] = pairs.groupby("user_id")["finished_cnt"].cumsum() - pairs["finished_cnt"]
    pairs["user_hist_cancels"] = pairs.groupby("user_id")["user_cancel_cnt"].cumsum() - pairs["user_cancel_cnt"]
    pairs["user_apply_rate"] = pairs["user_hist_applies"] / pairs["user_hist_views"].clip(lower=1)
    pairs["user_finish_rate"] = pairs["user_hist_finished"] / pairs["user_hist_applies"].clip(lower=1)
    pairs["user_cancel_rate"] = pairs["user_hist_cancels"] / pairs["user_hist_applies"].clip(lower=1)
    pairs["cum_shifts_viewed"] = pairs.groupby("user_id").cumcount()

    # Давность последней активности
    last_active = ev_before.groupby(["user_id", "shift_id"])["ts"].max().reset_index(name="last_active_ts")
    pairs = pairs.merge(last_active, on=["user_id", "shift_id"], how="left")
    pairs["recency_days"] = ((pairs["start_at"] - pairs["last_active_ts"]).dt.total_seconds() / 86400).clip(lower=0)

    # Усталость: работал 8+ч за последние 2 дня
    shift_hrs = shifts.rename(columns={"id": "shift_id"})[["shift_id", "hours"]].rename(columns={"hours": "shift_hours"})
    fin_long = ev_before[ev_before["interaction"] == "FINISHED"].merge(shift_hrs, on="shift_id", how="left")
    fatigue = pairs[["user_id", "shift_id", "start_at"]].merge(
        fin_long[["user_id", "shift_id", "ts", "shift_hours"]].rename(columns={"shift_id": "fin_shift", "ts": "fin_ts"}),
        on="user_id",
        how="left",
    )
    fatigue = fatigue[
        (fatigue["fin_ts"] < fatigue["start_at"])
        & ((fatigue["start_at"] - fatigue["fin_ts"]).dt.total_seconds() < 2 * 86400)
        & (fatigue["shift_hours"] >= 8)
    ][["user_id", "shift_id"]].drop_duplicates()
    fatigue["worked_long_recently"] = 1
    pairs = pairs.merge(fatigue, on=["user_id", "shift_id"], how="left")
    pairs["worked_long_recently"] = pairs["worked_long_recently"].fillna(0).astype(int)

    # Объединение со сменами и пользователями (is_strict_location — обязательный по спецификации)
    shifts_r = shifts.rename(columns={"id": "shift_id"})
    pairs = pairs.drop(columns=["start_at"])
    df = pairs.merge(shifts_r, on="shift_id", how="inner")
    df = df.merge(
        users.rename(columns={"id": "user_id", "location_id": "user_location_id"})[
            ["user_id", "user_location_id", "has_mk", "is_strict_location"]
        ],
        on="user_id",
        how="left",
    )

    df["same_location"] = (df["location_id"].astype(str) == df["user_location_id"].astype(str)).astype(int)
    df["mk_ok"] = (df["has_mk"].astype(float) >= df["need_mk"].astype(float)).astype(int)
    df["is_strict_location"] = df["is_strict_location"].astype(float)

    # Работал раньше у работодателя
    fin_emp = ev_before[ev_before["interaction"] == "FINISHED"][["user_id", "employer_id", "start_at"]].rename(
        columns={"start_at": "fin_at"}
    )
    emp_check = df[["user_id", "shift_id", "employer_id", "start_at"]].merge(fin_emp, on=["user_id", "employer_id"], how="left")
    emp_check = emp_check[emp_check["fin_at"] < emp_check["start_at"]][["user_id", "shift_id"]].drop_duplicates()
    emp_check["worked_employer_before"] = 1
    df = df.merge(emp_check, on=["user_id", "shift_id"], how="left")
    df["worked_employer_before"] = df["worked_employer_before"].fillna(0).astype(int)

    # Работал раньше на точке
    fin_wp = ev_before[ev_before["interaction"] == "FINISHED"][["user_id", "workplace_id", "start_at"]].rename(
        columns={"start_at": "fin_at"}
    )
    wp_check = df[["user_id", "shift_id", "workplace_id", "start_at"]].merge(fin_wp, on=["user_id", "workplace_id"], how="left")
    wp_check = wp_check[wp_check["fin_at"] < wp_check["start_at"]][["user_id", "shift_id"]].drop_duplicates()
    wp_check["worked_workplace_before"] = 1
    df = df.merge(wp_check, on=["user_id", "shift_id"], how="left")
    df["worked_workplace_before"] = df["worked_workplace_before"].fillna(0).astype(int)

    # Совпадение типа задачи с любимым
    apply_fin_tt = ev_before[ev_before["interaction"].isin(["APPLY", "FINISHED"])][["user_id", "task_type", "start_at"]].rename(
        columns={"start_at": "ev_start"}
    )
    fav_check = df[["user_id", "shift_id", "start_at", "task_type"]].merge(apply_fin_tt, on="user_id", how="left")
    fav_check = fav_check[fav_check["ev_start"] < fav_check["start_at"]]
    fav_cnt = fav_check.groupby(["user_id", "shift_id", "task_type_x"]).size().reset_index(name="cnt")
    fav_task = fav_cnt.loc[fav_cnt.groupby(["user_id", "shift_id"])["cnt"].idxmax()][
        ["user_id", "shift_id", "task_type_x"]
    ].rename(columns={"task_type_x": "fav_task"})
    df = df.merge(fav_task, on=["user_id", "shift_id"], how="left")
    df["task_match"] = (df["task_type"] == df["fav_task"]).astype(int)

    # CTR работодателя
    emp_agg = ev_before.groupby("employer_id").agg(
        emp_views=("interaction", lambda x: (x == "VIEW").sum()),
        emp_applies=("interaction", lambda x: x.isin(["APPLY", "FINISHED"]).sum()),
    ).reset_index()
    emp_agg["emp_ctr"] = emp_agg["emp_applies"] / emp_agg["emp_views"].clip(lower=1)
    df = df.merge(emp_agg[["employer_id", "emp_ctr"]], on="employer_id", how="left")
    df["emp_ctr"] = df["emp_ctr"].fillna(0)

    # Заполненность смены
    fill = ev_before[ev_before["interaction"].isin(["APPLY", "FINISHED"])].groupby("shift_id").size().reset_index(name="n_applied")
    df = df.merge(fill, on="shift_id", how="left")
    df["n_applied"] = df["n_applied"].fillna(0)
    df["fill_rate"] = (df["n_applied"] / df["capacity"].clip(lower=1)).clip(upper=5)

    # Дней до смены с момента первого просмотра
    first_view = ev_before[ev_before["interaction"] == "VIEW"].groupby(["user_id", "shift_id"])["ts"].min().reset_index(name="first_view_ts")
    df = df.merge(first_view, on=["user_id", "shift_id"], how="left")
    df["days_to_shift"] = ((df["start_at"] - df["first_view_ts"]).dt.total_seconds() / 86400).clip(lower=0)

    # Временные признаки (month, is_weekend, day_of_month убраны после ablation)
    df["reward_per_hour"] = df["reward"] / df["hours"].clip(lower=1)
    df["weekday"] = df["start_at"].dt.dayofweek
    df["hour"] = df["start_at"].dt.hour
    df["is_holiday"] = df["start_at"].apply(lambda x: int((x.month, x.day) in NEW_YEAR_HOLIDAYS))

    # Клиппинг выбросов
    df["hours"] = df["hours"].clip(upper=df["hours"].quantile(0.99))
    df["reward"] = df["reward"].clip(upper=df["reward"].quantile(0.98))
    df["reward_per_hour"] = df["reward_per_hour"].clip(upper=500)
    df["capacity"] = df["capacity"].clip(upper=df["capacity"].quantile(0.99))

    # === НОВЫЕ ПРИЗНАКИ V2 ===

    # 1. system_cancel_cnt - количество системных отмен
    sys_cnt = (
        ev_before[ev_before["interaction"] == "SYSTEM_CANCEL"]
        .groupby(["user_id", "shift_id"])
        .size()
        .reset_index(name="system_cancel_cnt")
    )
    df = df.merge(sys_cnt, on=["user_id", "shift_id"], how="left")
    df["system_cancel_cnt"] = df["system_cancel_cnt"].fillna(0).astype(int)

    # 2. user_reliability_score - взвешенный балл надежности
    INTERACTION_WEIGHTS = {"FINISHED": 2.0, "APPLY": 1.0, "USER_CANCEL": -1.5, "SYSTEM_CANCEL": -0.5}

    pairs_hist = ev_before.groupby(["user_id", "shift_id"], as_index=False).agg(
        _fin=("interaction", lambda s: (s == "FINISHED").sum()),
        _app=("interaction", lambda s: (s == "APPLY").sum()),
        _uc=("interaction", lambda s: (s == "USER_CANCEL").sum()),
        _sc=("interaction", lambda s: (s == "SYSTEM_CANCEL").sum()),
    )
    pairs_hist = pairs_hist.merge(shifts_info[["shift_id", "start_at"]], on="shift_id", how="left")
    pairs_hist = pairs_hist.sort_values(["user_id", "start_at"]).reset_index(drop=True)

    for col, raw in [("h_fin", "_fin"), ("h_app", "_app"), ("h_uc", "_uc"), ("h_sc", "_sc")]:
        pairs_hist[col] = pairs_hist.groupby("user_id")[raw].cumsum() - pairs_hist[raw]

    pairs_hist["user_reliability_score"] = (
        pairs_hist["h_fin"] * INTERACTION_WEIGHTS["FINISHED"]
        + pairs_hist["h_app"] * INTERACTION_WEIGHTS["APPLY"]
        + pairs_hist["h_uc"] * INTERACTION_WEIGHTS["USER_CANCEL"]
        + pairs_hist["h_sc"] * INTERACTION_WEIGHTS["SYSTEM_CANCEL"]
    )

    df = df.merge(
        pairs_hist[["user_id", "shift_id", "user_reliability_score"]],
        on=["user_id", "shift_id"],
        how="left",
    )
    df["user_reliability_score"] = df["user_reliability_score"].fillna(0)

    fin_before = ev_before[ev_before["interaction"] == "FINISHED"]

    # 3. user_finished_employer - количество завершенных смен у работодателя
    fin_emp = fin_before[["user_id", "employer_id", "start_at"]].rename(columns={"start_at": "fin_at"})
    emp_cnt = df[["user_id", "shift_id", "employer_id", "start_at"]].merge(fin_emp, on=["user_id", "employer_id"], how="left")
    emp_cnt = (
        emp_cnt[emp_cnt["fin_at"] < emp_cnt["start_at"]]
        .groupby(["user_id", "shift_id"])
        .size()
        .reset_index(name="user_finished_employer")
    )
    df = df.merge(emp_cnt, on=["user_id", "shift_id"], how="left")
    df["user_finished_employer"] = df["user_finished_employer"].fillna(0).astype(int)

    # 4. user_finished_workplace - количество завершенных смен на точке
    fin_wp = fin_before[["user_id", "workplace_id", "start_at"]].rename(columns={"start_at": "fin_at"})
    wp_cnt = df[["user_id", "shift_id", "workplace_id", "start_at"]].merge(fin_wp, on=["user_id", "workplace_id"], how="left")
    wp_cnt = (
        wp_cnt[wp_cnt["fin_at"] < wp_cnt["start_at"]]
        .groupby(["user_id", "shift_id"])
        .size()
        .reset_index(name="user_finished_workplace")
    )
    df = df.merge(wp_cnt, on=["user_id", "shift_id"], how="left")
    df["user_finished_workplace"] = df["user_finished_workplace"].fillna(0).astype(int)

    # 5. user_task_affinity - аффинити к типу задачи
    task_aff = ev_before[ev_before["interaction"].isin(["APPLY", "FINISHED"])].merge(
        shifts.rename(columns={"id": "shift_id"})[["shift_id", "task_type", "start_at"]].rename(
            columns={"start_at": "sh_start", "task_type": "tt"}
        ),
        on="shift_id",
        how="inner",
    )
    task_aff2 = df[["user_id", "shift_id", "task_type", "start_at"]].merge(
        task_aff[["user_id", "tt", "sh_start"]], on="user_id", how="left"
    )
    task_aff2 = task_aff2[
        (task_aff2["sh_start"] < task_aff2["start_at"]) & (task_aff2["tt"] == task_aff2["task_type"])
    ]
    task_aff_cnt = task_aff2.groupby(["user_id", "shift_id"]).size().reset_index(name="user_task_affinity")
    df = df.merge(task_aff_cnt, on=["user_id", "shift_id"], how="left")
    df["user_task_affinity"] = df["user_task_affinity"].fillna(0).astype(int)

    # 6. reward_delta - отклонение RPH смены от среднего пользователя по завершённым сменам
    shift_pay = shifts.rename(columns={"id": "shift_id"})[["shift_id", "reward", "hours", "start_at"]].rename(
        columns={"start_at": "sh_start", "reward": "fin_reward", "hours": "fin_hours"}
    )
    fin_rph = fin_before[["user_id", "shift_id"]].merge(shift_pay, on="shift_id", how="inner")
    fin_rph["rph"] = fin_rph["fin_reward"] / fin_rph["fin_hours"].clip(lower=1)
    rph_check = df[["user_id", "shift_id", "start_at"]].merge(fin_rph[["user_id", "rph", "sh_start"]], on="user_id", how="left")
    rph_check = rph_check[rph_check["sh_start"] < rph_check["start_at"]]
    user_avg_rph = rph_check.groupby(["user_id", "shift_id"])["rph"].mean().reset_index(name="user_avg_rph")
    df = df.merge(user_avg_rph, on=["user_id", "shift_id"], how="left")
    df["user_avg_rph"] = df["user_avg_rph"].fillna(df["reward_per_hour"].median())
    df["reward_delta"] = df["reward_per_hour"] - df["user_avg_rph"]
    df = df.drop(columns=["user_avg_rph"])

    return df


def _time_split(frame: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        raise ValueError("Training frame is empty after preprocessing.")
    unique_dates = np.array(sorted(frame["start_at"].dropna().dt.date.unique()))
    if unique_dates.size < 2:
        raise ValueError("Not enough temporal points for split.")
    split_idx = max(1, int(unique_dates.size * (1 - test_ratio)))
    split_idx = min(split_idx, unique_dates.size - 1)
    split_border = pd.Timestamp(unique_dates[split_idx], tz="UTC")
    train = frame[frame["start_at"] < split_border].copy()
    test = frame[frame["start_at"] >= split_border].copy()
    if train.empty or test.empty:
        raise ValueError("Time split produced empty train or test set.")
    return train, test


def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Приводит фрейм к виду, ожидаемому CatBoost."""
    x = frame[FEATURE_COLUMNS].copy()
    for col in BOOL_AS_FLOAT_FEATURES:
        x[col] = x[col].astype(float)
    for col in CATEGORICAL_FEATURES:
        x[col] = x[col].astype(str)
    return x


def _generate_importance_plot(
    model: CatBoostClassifier, output_dir: Path
) -> dict[str, str]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=True)
    importance_top = importance.tail(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(importance_top)), importance_top.values, color="#4878CF")
    ax.set_yticks(range(len(importance_top)))
    ax.set_yticklabels(importance_top.index)
    ax.set_title("CatBoost feature importance (top-20)")
    ax.set_xlabel("importance")
    plt.tight_layout()

    path = plots_dir / "feature_importance.png"
    plt.savefig(path, dpi=140)
    plt.close()
    return {"feature_importance": str(path)}


def run_training(cfg: TrainConfig) -> dict[str, object]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Stage 1/8: Загрузка и валидация train CSV")
    users, shifts, events, checks = _load_and_validate_data(cfg)
    LOGGER.info(
        "Loaded rows after cleanup: users=%s shifts=%s events=%s",
        len(users),
        len(shifts),
        len(events),
    )

    LOGGER.info("Stage 2/8: Построение обучающего фрейма и таргета")
    frame = _build_training_frame(users, shifts, events)
    LOGGER.info("Built training frame rows=%s", len(frame))

    LOGGER.info("Stage 3/8: Временной сплит (~80/20) без утечки")
    train_frame, test_frame = _time_split(frame, cfg.test_ratio)
    LOGGER.info("Split rows: train=%s test=%s", len(train_frame), len(test_frame))

    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns after preprocessing: {missing}")

    x_train = _prepare_features(train_frame)
    x_test = _prepare_features(test_frame)

    y_train = train_frame["target"].astype(int)
    y_test = test_frame["target"].astype(int)

    LOGGER.info("Stage 4/8: Список признаков и превью")
    LOGGER.info("Feature columns: %s", ", ".join(FEATURE_COLUMNS))
    LOGGER.info("Feature sample:\n%s", x_train.head(5).to_string(index=False))

    LOGGER.info("Stage 5/8: Обучение CatBoost с early stopping")
    pool_train = Pool(x_train, y_train, cat_features=CATEGORICAL_FEATURES)
    pool_test = Pool(x_test, y_test, cat_features=CATEGORICAL_FEATURES)

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        scale_pos_weight=19,
        random_seed=cfg.random_state,
        thread_count=-1,
        early_stopping_rounds=50,
        verbose=100,
    )
    model.fit(pool_train, eval_set=pool_test)
    LOGGER.info("Best iteration: %s", model.best_iteration_)

    LOGGER.info("Stage 6/8: Инференс и расчёт целевой метрики")
    proba = model.predict_proba(pool_test)[:, 1]
    metric_df = test_frame[["shift_id", "start_at", "capacity", "target"]].copy()
    metric_df["score"] = proba
    metric_result = calculate_target_metric(metric_df)
    metrics = {
        "target_metric": metric_result.target_metric,
        "evaluated_days": metric_result.evaluated_days,
        "evaluated_groups": metric_result.evaluated_groups,
        "evaluated_shifts": metric_result.evaluated_shifts,
        "day_metrics": metric_result.day_metrics,
        "test_rows": int(len(test_frame)),
        "train_rows": int(len(train_frame)),
        "best_iteration": int(model.best_iteration_) if model.best_iteration_ else 0,
    }

    LOGGER.info("Stage 7/8: Сохранение модели и артефактов в %s", output_dir)
    with (output_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "feature_columns": FEATURE_COLUMNS,
                "categorical_features": CATEGORICAL_FEATURES,
                "bool_as_float_features": BOOL_AS_FLOAT_FEATURES,
                "examples": x_train.head(5).to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (output_dir / "train_config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "data_contract_check.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# Train Report",
        "",
        "## Model",
        "",
        "- algorithm: CatBoostClassifier",
        f"- features: {len(FEATURE_COLUMNS)} (final set after ablation)",
        "",
        "## Data",
        "",
        f"- train_rows: {len(train_frame):,}",
        f"- test_rows: {len(test_frame):,}",
        "",
        "## Target metric (by regulation)",
        "",
        f"- target_metric: {metrics['target_metric']}",
        f"- evaluated_days: {metrics['evaluated_days']}",
        f"- evaluated_groups: {metrics['evaluated_groups']}",
        f"- evaluated_shifts: {metrics['evaluated_shifts']}",
        f"- best_iteration: {metrics['best_iteration']}",
    ]

    importance_result: dict[str, str] = {}
    if cfg.skip_shap:
        report_lines.extend(["", "## Feature importance", "", "- skipped by config (--skip-shap)."])
    else:
        try:
            importance_result = _generate_importance_plot(model, output_dir)
            report_lines.extend(
                [
                    "",
                    "## Feature importance",
                    "",
                    f"- plot: {importance_result['feature_importance']}",
                ]
            )
        except Exception as exc:  # noqa: BLE001
            skip_path = output_dir / "plots" / "importance_skipped.txt"
            skip_path.parent.mkdir(parents=True, exist_ok=True)
            skip_path.write_text(f"Importance plot failed: {exc}", encoding="utf-8")
            report_lines.extend(["", "## Feature importance", "", f"- failed: {exc}"])

    (output_dir / "train_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    LOGGER.info("Stage 8/8: Training pipeline finished successfully")
    return {"metrics": metrics, "plots": importance_result}
