from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from hackaton.service.dto import ShiftDTO

NEW_YEAR_HOLIDAYS = {(1, d) for d in range(1, 9)}


class Predictor:
    """Предсказатель для ранжирования кандидатов на смену."""

    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.model = None
        self.feature_columns = [
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
        ]

    def load_model(self) -> None:
        """Загрузка модели из pickle."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        with self.model_path.open("rb") as f:
            self.model = pickle.load(f)

    async def build_features(
        self,
        shift: ShiftDTO,
        candidate_ids: list[str],
        db_path: str,
    ) -> pd.DataFrame:
        """Построение признаков для пар user-shift."""
        import aiosqlite

        async with aiosqlite.connect(db_path) as db:
            # Получаем данные пользователей
            user_query = f"""
            SELECT id, location_id, has_mk, is_strict_location
            FROM users
            WHERE id IN ({','.join('?' * len(candidate_ids))})
            """
            cursor = await db.execute(user_query, candidate_ids)
            user_rows = await cursor.fetchall()
            users_df = pd.DataFrame(
                user_rows, columns=["user_id", "user_location_id", "has_mk", "is_strict_location"]
            )

            # Получаем события до start_at смены
            event_query = """
            SELECT user_id, interaction, ts
            FROM events
            WHERE user_id IN ({})
              AND shift_id = ?
              AND ts < ?
            """.format(",".join("?" * len(candidate_ids)))
            cursor = await db.execute(event_query, [*candidate_ids, shift.id, shift.start_at.isoformat()])
            event_rows = await cursor.fetchall()
            events_df = pd.DataFrame(event_rows, columns=["user_id", "interaction", "ts"])
            events_df["ts"] = pd.to_datetime(events_df["ts"], utc=True)

            # История пользователя (все события до этой смены)
            hist_query = """
            SELECT e.user_id, e.interaction, e.ts, s.start_at
            FROM events e
            JOIN shifts s ON e.shift_id = s.id
            WHERE e.user_id IN ({})
              AND s.start_at < ?
            """.format(",".join("?" * len(candidate_ids)))
            cursor = await db.execute(hist_query, [*candidate_ids, shift.start_at.isoformat()])
            hist_rows = await cursor.fetchall()
            hist_df = pd.DataFrame(hist_rows, columns=["user_id", "interaction", "ts", "shift_start"])
            hist_df["ts"] = pd.to_datetime(hist_df["ts"], utc=True)
            hist_df["shift_start"] = pd.to_datetime(hist_df["shift_start"], utc=True)

            # Работал раньше у работодателя
            emp_query = """
            SELECT DISTINCT e.user_id
            FROM events e
            JOIN shifts s ON e.shift_id = s.id
            WHERE e.user_id IN ({})
              AND e.interaction = 'FINISHED'
              AND s.employer_id = ?
              AND s.start_at < ?
            """.format(",".join("?" * len(candidate_ids)))
            cursor = await db.execute(emp_query, [*candidate_ids, shift.employer_id, shift.start_at.isoformat()])
            emp_rows = await cursor.fetchall()
            worked_emp = {row[0] for row in emp_rows}

            # Работал раньше на точке
            wp_query = """
            SELECT DISTINCT e.user_id
            FROM events e
            JOIN shifts s ON e.shift_id = s.id
            WHERE e.user_id IN ({})
              AND e.interaction = 'FINISHED'
              AND s.workplace_id = ?
              AND s.start_at < ?
            """.format(",".join("?" * len(candidate_ids)))
            cursor = await db.execute(wp_query, [*candidate_ids, shift.workplace_id, shift.start_at.isoformat()])
            wp_rows = await cursor.fetchall()
            worked_wp = {row[0] for row in wp_rows}

            # Любимый тип задачи
            task_query = """
            SELECT e.user_id, s.task_type, COUNT(*) as cnt
            FROM events e
            JOIN shifts s ON e.shift_id = s.id
            WHERE e.user_id IN ({})
              AND e.interaction IN ('APPLY', 'FINISHED')
              AND s.start_at < ?
            GROUP BY e.user_id, s.task_type
            """.format(",".join("?" * len(candidate_ids)))
            cursor = await db.execute(task_query, [*candidate_ids, shift.start_at.isoformat()])
            task_rows = await cursor.fetchall()
            task_df = pd.DataFrame(task_rows, columns=["user_id", "task_type", "cnt"])
            fav_tasks = task_df.loc[task_df.groupby("user_id")["cnt"].idxmax()][["user_id", "task_type"]]
            fav_tasks_dict = dict(zip(fav_tasks["user_id"], fav_tasks["task_type"]))

            # CTR работодателя
            emp_ctr_query = """
            SELECT
                COUNT(CASE WHEN e.interaction = 'VIEW' THEN 1 END) as views,
                COUNT(CASE WHEN e.interaction IN ('APPLY', 'FINISHED') THEN 1 END) as applies
            FROM events e
            JOIN shifts s ON e.shift_id = s.id
            WHERE s.employer_id = ?
              AND s.start_at < ?
            """
            cursor = await db.execute(emp_ctr_query, [shift.employer_id, shift.start_at.isoformat()])
            emp_ctr_row = await cursor.fetchone()
            emp_ctr = emp_ctr_row[1] / max(emp_ctr_row[0], 1) if emp_ctr_row else 0

            # Заполненность смены
            fill_query = """
            SELECT COUNT(DISTINCT user_id)
            FROM events
            WHERE shift_id = ?
              AND interaction IN ('APPLY', 'FINISHED')
              AND ts < ?
            """
            cursor = await db.execute(fill_query, [shift.id, shift.start_at.isoformat()])
            fill_row = await cursor.fetchone()
            n_applied = fill_row[0] if fill_row else 0

        # Построение фрейма признаков
        features = []
        for user_id in candidate_ids:
            user_data = users_df[users_df["user_id"] == user_id].iloc[0] if len(users_df[users_df["user_id"] == user_id]) > 0 else None
            if user_data is None:
                continue

            # События текущей смены
            user_events = events_df[events_df["user_id"] == user_id]
            view_cnt = int((user_events["interaction"] == "VIEW").sum())

            # История пользователя
            user_hist = hist_df[hist_df["user_id"] == user_id]
            user_hist_views = int((user_hist["interaction"] == "VIEW").sum())
            user_hist_applies = int((user_hist["interaction"] == "APPLY").sum())
            user_hist_finished = int((user_hist["interaction"] == "FINISHED").sum())
            user_hist_cancels = int((user_hist["interaction"] == "USER_CANCEL").sum())

            user_apply_rate = user_hist_applies / max(user_hist_views, 1)
            user_finish_rate = user_hist_finished / max(user_hist_applies, 1)
            user_cancel_rate = user_hist_cancels / max(user_hist_applies, 1)

            # Давность последней активности
            if len(user_events) > 0:
                last_ts = user_events["ts"].max()
                recency_days = (shift.start_at - last_ts).total_seconds() / 86400
            else:
                recency_days = 999

            # Дней до смены с момента первого просмотра
            first_view = user_events[user_events["interaction"] == "VIEW"]["ts"].min() if len(user_events) > 0 else None
            days_to_shift = (shift.start_at - first_view).total_seconds() / 86400 if pd.notna(first_view) else 0

            # Усталость
            long_shifts = user_hist[
                (user_hist["interaction"] == "FINISHED")
                & ((shift.start_at - user_hist["shift_start"]).dt.total_seconds() < 2 * 86400)
            ]
            worked_long_recently = 0  # Упрощение: нужна информация о длительности смен

            # Совпадения
            same_location = int(shift.location_id == user_data["user_location_id"])
            mk_ok = int(user_data["has_mk"] >= shift.need_mk)
            is_strict_location = float(user_data["is_strict_location"])
            worked_employer_before = int(user_id in worked_emp)
            worked_workplace_before = int(user_id in worked_wp)
            task_match = int(fav_tasks_dict.get(user_id) == shift.task_type)

            # Временные признаки
            reward_per_hour = shift.reward / max(shift.hours, 1)
            hour = shift.start_at.hour
            weekday = shift.start_at.weekday()
            is_holiday = int((shift.start_at.month, shift.start_at.day) in NEW_YEAR_HOLIDAYS)

            fill_rate = min(n_applied / max(shift.capacity, 1), 5)

            features.append(
                {
                    "user_id": user_id,
                    "hours": min(shift.hours, 22),
                    "reward": min(shift.reward, 9500),
                    "reward_per_hour": min(reward_per_hour, 500),
                    "capacity": min(shift.capacity, 154),
                    "hour": hour,
                    "weekday": weekday,
                    "is_holiday": is_holiday,
                    "view_cnt": view_cnt,
                    "user_hist_views": user_hist_views,
                    "user_hist_applies": user_hist_applies,
                    "user_hist_finished": user_hist_finished,
                    "user_hist_cancels": user_hist_cancels,
                    "user_apply_rate": user_apply_rate,
                    "user_finish_rate": user_finish_rate,
                    "user_cancel_rate": user_cancel_rate,
                    "cum_shifts_viewed": user_hist_views,
                    "recency_days": max(recency_days, 0),
                    "worked_long_recently": worked_long_recently,
                    "same_location": same_location,
                    "mk_ok": mk_ok,
                    "is_strict_location": is_strict_location,
                    "worked_employer_before": worked_employer_before,
                    "worked_workplace_before": worked_workplace_before,
                    "task_match": task_match,
                    "emp_ctr": emp_ctr,
                    "fill_rate": fill_rate,
                    "days_to_shift": max(days_to_shift, 0),
                    "need_mk": float(shift.need_mk),
                    "id_differential": float(shift.id_differential),
                    "has_mk": float(user_data["has_mk"]),
                    "task_type": shift.task_type,
                }
            )

        df = pd.DataFrame(features)
        if len(df) == 0:
            return df

        # task_type как категория
        df["task_type"] = df["task_type"].astype("category")
        return df

    async def predict(
        self,
        shift: ShiftDTO,
        candidate_ids: list[str],
        db_path: str,
        limit: int,
    ) -> list[str]:
        """Предсказание топ-K кандидатов для смены."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not candidate_ids:
            return []

        # Построение признаков
        features_df = await self.build_features(shift, candidate_ids, db_path)
        if len(features_df) == 0:
            return []

        # Предсказание
        x = features_df[self.feature_columns]
        proba = self.model.predict_proba(x)[:, 1]
        features_df["score"] = proba

        # Сортировка по score и возврат топ-K
        top_k = features_df.nlargest(limit, "score")
        return top_k["user_id"].tolist()
