"""
玩家数据统计与分析系统 - 核心模块（修复版）

功能：
- record_match:        完整比赛记录（击杀/死亡/伤害/命中率），去重，持久化
- calculate_stats:     统计计算（考虑样本数量、时间衰减，精确舍入）
- trend_analysis:      趋势分析（时间范围、方向判断、玩家对比）
- achievement_tracking: 成就追踪（解锁、进度、稀有度）
- stats_export:        统计导出（JSON / CSV / 文本，社交分享）
"""
import csv
import io
import json
import math
import os
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional


@dataclass
class Config:
    pass


# 时间衰减半衰期（天）：越久远的比赛权重越低
DEFAULT_HALF_LIFE_DAYS = 7.0
# 贝叶斯平滑的先验场次（先验胜率 50%），用于小样本修正
DEFAULT_PRIOR_MATCHES = 10.0

RARITY_ORDER = ("common", "rare", "epic", "legendary")


def _round_pct(value: float, digits: int = 1) -> float:
    """使用 Decimal ROUND_HALF_UP 精确舍入，避免浮点/银行家舍入误差。"""
    quant = Decimal(1).scaleb(-digits)
    return float(Decimal(repr(value)).quantize(quant, rounding=ROUND_HALF_UP))


class PlayerStats:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self.storage_path = self.config.get(
            "storage_path",
            os.path.join(os.getcwd(), "player_stats_data.json"),
        )
        self.half_life_days = float(
            self.config.get("half_life_days", DEFAULT_HALF_LIFE_DAYS)
        )
        self.prior_matches = float(
            self.config.get("prior_matches", DEFAULT_PRIOR_MATCHES)
        )
        self._matches: List[Dict] = []
        self._match_ids = set()
        self._load()

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def _load(self):
        if self.storage_path and os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._matches = list(data.get("matches", []))
                self._match_ids = {m["match_id"] for m in self._matches}
            except (json.JSONDecodeError, OSError, KeyError):
                self._matches = []
                self._match_ids = set()

    def _save(self):
        if not self.storage_path:
            return
        directory = os.path.dirname(os.path.abspath(self.storage_path))
        os.makedirs(directory, exist_ok=True)
        tmp_path = self.storage_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"matches": self._matches}, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.storage_path)

    # ------------------------------------------------------------------
    # 比赛记录
    # ------------------------------------------------------------------
    def record_match(
        self,
        match_id: str,
        won: bool,
        kills: int = 0,
        deaths: int = 0,
        damage: float = 0.0,
        shots_fired: int = 0,
        shots_hit: int = 0,
        timestamp: Optional[float] = None,
        **extra,
    ) -> bool:
        """记录一场比赛。重复 match_id 返回 False 且不记录。"""
        if match_id in self._match_ids:
            return False
        match = {
            "match_id": match_id,
            "won": bool(won),
            "kills": int(kills),
            "deaths": int(deaths),
            "damage": float(damage),
            "shots_fired": int(shots_fired),
            "shots_hit": int(shots_hit),
            "timestamp": float(timestamp if timestamp is not None else time.time()),
        }
        match.update(extra)
        self._matches.append(match)
        self._match_ids.add(match_id)
        self._save()
        return True

    # ------------------------------------------------------------------
    # 统计计算
    # ------------------------------------------------------------------
    def _time_weight(self, timestamp: float, now: float) -> float:
        age_days = max(0.0, (now - timestamp) / 86400.0)
        return 0.5 ** (age_days / self.half_life_days)

    def calculate_stats(
        self, now: Optional[float] = None, matches: Optional[List[Dict]] = None
    ) -> Dict:
        """计算统计数据。

        - 胜率/命中率使用精确舍入（ROUND_HALF_UP）
        - smoothed_win_rate 使用贝叶斯平滑，小样本不会被极端化
        - weighted_win_rate 使用时间衰减，近期比赛权重更高
        """
        now = now if now is not None else time.time()
        matches = self._matches if matches is None else matches
        total = len(matches)
        if total == 0:
            return {
                "total_matches": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "smoothed_win_rate": 0.0,
                "weighted_win_rate": 0.0,
                "total_kills": 0,
                "total_deaths": 0,
                "kd_ratio": 0.0,
                "avg_damage": 0.0,
                "shots_fired": 0,
                "shots_hit": 0,
                "accuracy": 0.0,
                "weighted_accuracy": 0.0,
            }

        wins = sum(1 for m in matches if m["won"])
        kills = sum(m["kills"] for m in matches)
        deaths = sum(m["deaths"] for m in matches)
        damage = sum(m["damage"] for m in matches)
        fired = sum(m["shots_fired"] for m in matches)
        hit = sum(m["shots_hit"] for m in matches)

        weights = [self._time_weight(m["timestamp"], now) for m in matches]
        weight_sum = sum(weights)
        weighted_wins = sum(w for w, m in zip(weights, matches) if m["won"])
        weighted_fired = sum(w * m["shots_fired"] for w, m in zip(weights, matches))
        weighted_hit = sum(w * m["shots_hit"] for w, m in zip(weights, matches))

        # 贝叶斯平滑：先验 prior_matches 场、胜率 50%，样本越少修正越强
        prior = self.prior_matches
        smoothed = (wins + prior * 0.5) / (total + prior) * 100.0

        return {
            "total_matches": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": _round_pct(wins / total * 100.0),
            "smoothed_win_rate": _round_pct(smoothed),
            "weighted_win_rate": _round_pct(weighted_wins / weight_sum * 100.0),
            "total_kills": kills,
            "total_deaths": deaths,
            "kd_ratio": _round_pct(kills / deaths, 2) if deaths > 0 else float(kills),
            "avg_damage": _round_pct(damage / total, 1),
            "shots_fired": fired,
            "shots_hit": hit,
            "accuracy": _round_pct(hit / fired * 100.0) if fired > 0 else 0.0,
            "weighted_accuracy": (
                _round_pct(weighted_hit / weighted_fired * 100.0)
                if weighted_fired > 0
                else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # 趋势分析
    # ------------------------------------------------------------------
    @staticmethod
    def _metric_value(matches: List[Dict], metric: str) -> Optional[float]:
        if not matches:
            return None
        if metric == "win_rate":
            return sum(1 for m in matches if m["won"]) / len(matches) * 100.0
        if metric == "accuracy":
            fired = sum(m["shots_fired"] for m in matches)
            if fired == 0:
                return None
            return sum(m["shots_hit"] for m in matches) / fired * 100.0
        if metric == "kd_ratio":
            deaths = sum(m["deaths"] for m in matches)
            kills = sum(m["kills"] for m in matches)
            return kills / deaths if deaths > 0 else float(kills)
        if metric == "avg_damage":
            return sum(m["damage"] for m in matches) / len(matches)
        raise ValueError(f"未知指标: {metric}")

    def trend_analysis(
        self,
        metric: str = "win_rate",
        days: int = 30,
        compare_with: Optional[List] = None,
        now: Optional[float] = None,
    ) -> Dict:
        """趋势分析。

        - 仅统计最近 days 天的比赛（时间范围）
        - 将窗口按时间均分为前后两半，比较指标变化方向
        - compare_with: 其他玩家的 PlayerStats 实例或 calculate_stats() 结果，
          用于计算与他人的差距和百分位排名
        """
        now = now if now is not None else time.time()
        cutoff = now - days * 86400.0
        window = [m for m in self._matches if m["timestamp"] >= cutoff]
        window.sort(key=lambda m: m["timestamp"])

        result = {
            "metric": metric,
            "days": days,
            "matches_in_window": len(window),
            "direction": "insufficient_data",
            "previous": None,
            "recent": None,
            "change": None,
            "comparison": None,
        }
        if len(window) < 2:
            return result

        mid = len(window) // 2
        first_half, second_half = window[:mid], window[mid:]
        previous = self._metric_value(first_half, metric)
        recent = self._metric_value(second_half, metric)
        if previous is None or recent is None:
            return result

        change = _round_pct(recent - previous, 2)
        threshold = 1.0 if metric in ("win_rate", "accuracy") else 0.05
        if change > threshold:
            direction = "improving"
        elif change < -threshold:
            direction = "declining"
        else:
            direction = "stable"

        result.update(
            direction=direction,
            previous=_round_pct(previous, 2),
            recent=_round_pct(recent, 2),
            change=change,
        )

        if compare_with:
            others = []
            for other in compare_with:
                if isinstance(other, PlayerStats):
                    others.append(other.calculate_stats(now=now))
                else:
                    others.append(other)
            key = metric if metric in ("win_rate", "accuracy") else None
            values = [o[key] for o in others if key and o.get(key) is not None]
            if values:
                own = self.calculate_stats(now=now)[key]
                below = sum(1 for v in values if own > v)
                result["comparison"] = {
                    "own": own,
                    "others_average": _round_pct(sum(values) / len(values), 2),
                    "diff_vs_average": _round_pct(
                        own - sum(values) / len(values), 2
                    ),
                    "percentile": _round_pct(below / len(values) * 100.0),
                    "better_than_average": own > sum(values) / len(values),
                }
        return result

    # ------------------------------------------------------------------
    # 成就追踪
    # ------------------------------------------------------------------
    def _best_win_streak(self) -> int:
        best = current = 0
        for m in sorted(self._matches, key=lambda x: x["timestamp"]):
            current = current + 1 if m["won"] else 0
            best = max(best, current)
        return best

    def _achievement_defs(self, stats: Dict) -> List[Dict]:
        return [
            {
                "id": "first_win",
                "name": "首胜",
                "description": "赢得第 1 场比赛",
                "rarity": "common",
                "current": stats["wins"],
                "target": 1,
            },
            {
                "id": "winner_10",
                "name": "常胜将军",
                "description": "累计赢得 10 场比赛",
                "rarity": "common",
                "current": stats["wins"],
                "target": 10,
            },
            {
                "id": "veteran_100",
                "name": "百战老兵",
                "description": "完成 100 场比赛",
                "rarity": "rare",
                "current": stats["total_matches"],
                "target": 100,
            },
            {
                "id": "streak_5",
                "name": "势不可挡",
                "description": "取得 5 连胜",
                "rarity": "rare",
                "current": self._best_win_streak(),
                "target": 5,
            },
            {
                "id": "sharpshooter",
                "name": "神枪手",
                "description": "至少 20 场比赛且命中率达到 70%",
                "rarity": "epic",
                "current": (
                    round(stats["accuracy"])
                    if stats["total_matches"] >= 20
                    else round(stats["total_matches"] / 20 * 70)
                ),
                "target": 70,
            },
            {
                "id": "legend_1000",
                "name": "传奇",
                "description": "完成 1000 场比赛",
                "rarity": "legendary",
                "current": stats["total_matches"],
                "target": 1000,
            },
        ]

    def achievement_tracking(self, now: Optional[float] = None) -> List[Dict]:
        """返回成就列表：解锁状态、进度（current/target/百分比）、稀有度。"""
        stats = self.calculate_stats(now=now)
        achievements = []
        for ach in self._achievement_defs(stats):
            current = min(ach["current"], ach["target"])
            unlocked = ach["current"] >= ach["target"]
            achievements.append(
                {
                    "id": ach["id"],
                    "name": ach["name"],
                    "description": ach["description"],
                    "rarity": ach["rarity"],
                    "unlocked": unlocked,
                    "progress": current,
                    "target": ach["target"],
                    "progress_pct": _round_pct(current / ach["target"] * 100.0),
                }
            )
        return achievements

    # ------------------------------------------------------------------
    # 统计导出与分享
    # ------------------------------------------------------------------
    def stats_export(self, format: str = "json", now: Optional[float] = None) -> str:
        """导出统计数据，支持 json / csv / text 三种格式。"""
        stats = self.calculate_stats(now=now)
        if format == "json":
            return json.dumps(
                {"stats": stats, "matches": self._matches},
                ensure_ascii=False,
                indent=2,
            )
        if format == "csv":
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(
                [
                    "match_id",
                    "won",
                    "kills",
                    "deaths",
                    "damage",
                    "shots_fired",
                    "shots_hit",
                    "timestamp",
                ]
            )
            for m in self._matches:
                writer.writerow(
                    [
                        m["match_id"],
                        m["won"],
                        m["kills"],
                        m["deaths"],
                        m["damage"],
                        m["shots_fired"],
                        m["shots_hit"],
                        m["timestamp"],
                    ]
                )
            return buf.getvalue()
        if format == "text":
            return (
                "===== 玩家数据统计 =====\n"
                f"总场次: {stats['total_matches']}\n"
                f"胜/负: {stats['wins']}/{stats['losses']}\n"
                f"胜率: {stats['win_rate']}%\n"
                f"K/D: {stats['kd_ratio']}\n"
                f"场均伤害: {stats['avg_damage']}\n"
                f"命中率: {stats['accuracy']}%\n"
            )
        raise ValueError(f"不支持的导出格式: {format}")

    def share(self, platform: str = "generic", now: Optional[float] = None) -> Dict:
        """生成社交平台分享内容（文案 + 分享链接）。"""
        stats = self.calculate_stats(now=now)
        text = (
            f"我已经打了 {stats['total_matches']} 场比赛，"
            f"胜率 {stats['win_rate']}%，命中率 {stats['accuracy']}%，"
            f"K/D {stats['kd_ratio']}！快来挑战我吧！"
        )
        urls = {
            "twitter": "https://twitter.com/intent/tweet?text=",
            "weibo": "https://service.weibo.com/share/share.php?title=",
            "generic": "https://share.example.com/post?text=",
        }
        from urllib.parse import quote

        base = urls.get(platform, urls["generic"])
        return {
            "platform": platform,
            "text": text,
            "url": base + quote(text),
        }

    # ------------------------------------------------------------------
    # 原有接口
    # ------------------------------------------------------------------
    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []
        self._matches = []
        self._match_ids = set()
        self._save()
