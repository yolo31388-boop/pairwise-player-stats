"""
玩家数据统计与分析系统

功能：
- record_match:        完整记录比赛数据（击杀/死亡/伤害/命中率），去重并持久化
- calculate_stats:     统计计算，考虑样本数量（贝叶斯收缩）与时间衰减，精确计算
- trend_analysis:      趋势分析，支持时间范围与和其他玩家对比
- achievement_tracking: 成就追踪，带进度与稀有度
- stats_export:        统计导出，支持 JSON/CSV/文本及社交分享
"""
import csv
import io
import json
import math
import os
import time
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Config:
    pass


# 稀有度排序（用于成就展示）
RARITY_ORDER = {"common": 0, "rare": 1, "epic": 2, "legendary": 3}

# 成就定义：id, 名称, 描述, 稀有度, 统计字段, 目标值, 最少比赛场次
ACHIEVEMENTS = [
    {"id": "first_win",   "name": "首胜",     "description": "赢得第 1 场比赛",          "rarity": "common",    "metric": "wins",       "target": 1,    "min_matches": 0},
    {"id": "ten_wins",    "name": "十连胜者", "description": "累计赢得 10 场比赛",        "rarity": "rare",      "metric": "wins",       "target": 10,   "min_matches": 0},
    {"id": "century",     "name": "身经百战", "description": "完成 100 场比赛",           "rarity": "rare",      "metric": "matches",    "target": 100,  "min_matches": 0},
    {"id": "killer",      "name": "杀手",     "description": "累计击杀 100 人",           "rarity": "common",    "metric": "kills",      "target": 100,  "min_matches": 0},
    {"id": "slayer",      "name": "屠戮者",   "description": "累计击杀 1000 人",          "rarity": "epic",      "metric": "kills",      "target": 1000, "min_matches": 0},
    {"id": "sharpshooter","name": "神枪手",   "description": "命中率达到 60%（至少 10 场）","rarity": "epic",      "metric": "accuracy_pct","target": 60,  "min_matches": 10},
    {"id": "legend",      "name": "传奇",     "description": "胜率达到 70%（至少 50 场）", "rarity": "legendary", "metric": "winrate_pct","target": 70,   "min_matches": 50},
]


class PlayerStats:
    """玩家数据统计与分析系统。"""

    DEFAULT_DATA_FILE = "player_stats_data.json"

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history: List[Dict] = []
        self._data_file = self.config.get("data_file", self.DEFAULT_DATA_FILE)
        # 时间衰减半衰期（秒），默认 30 天；<=0 表示不做时间衰减
        self._half_life = float(self.config.get("half_life_days", 30)) * 86400.0
        # 样本量收缩强度：先验等效场次 / 先验射击数
        self._prior_matches = float(self.config.get("prior_matches", 10))
        self._prior_shots = float(self.config.get("prior_shots", 20))
        self._load()

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def _load(self):
        try:
            with open(self._data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._history = list(data.get("matches", []))
            self._state = dict(data.get("state", {}))
        except (OSError, ValueError):
            self._history = []
            self._state = {}

    def _save(self):
        directory = os.path.dirname(os.path.abspath(self._data_file))
        os.makedirs(directory, exist_ok=True)
        tmp_file = self._data_file + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump({"matches": self._history, "state": self._state}, f,
                      ensure_ascii=False)
        os.replace(tmp_file, self._data_file)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []
        self._save()

    # ------------------------------------------------------------------
    # 比赛记录
    # ------------------------------------------------------------------
    def record_match(self, match: Dict) -> bool:
        """记录一场比赛。

        必填字段: match_id（用于去重）。
        可选字段: win/result, kills, deaths, damage, hits, shots, timestamp。
        返回 True 表示记录成功；重复 match_id 返回 False。
        """
        if not isinstance(match, dict):
            raise TypeError("match 必须是 dict")
        match_id = match.get("match_id")
        if match_id is None:
            raise ValueError("match 缺少必填字段 match_id")

        # 去重：同一场比赛只允许记录一次
        if any(m["match_id"] == match_id for m in self._history):
            return False

        record = {
            "match_id": match_id,
            "win": bool(match.get("win", match.get("result") == "win")),
            "kills": int(match.get("kills", 0)),
            "deaths": int(match.get("deaths", 0)),
            "damage": float(match.get("damage", 0.0)),
            "hits": int(match.get("hits", 0)),
            "shots": int(match.get("shots", 0)),
            "timestamp": float(match.get("timestamp", time.time())),
        }
        self._history.append(record)
        self._save()
        return True

    # ------------------------------------------------------------------
    # 统计计算
    # ------------------------------------------------------------------
    def _time_weight(self, timestamp: float, now: float) -> float:
        if self._half_life <= 0:
            return 1.0
        age = max(0.0, now - timestamp)
        return 0.5 ** (age / self._half_life)

    @staticmethod
    def _match_accuracy(match: Dict) -> float:
        return match["hits"] / match["shots"] if match["shots"] > 0 else 0.0

    def calculate_stats(self, now: Optional[float] = None) -> Dict:
        """计算统计数据。

        - 胜率/命中率由真实计数精确计算（先除后四舍五入到 4 位，不做中间取整）。
        - adjusted_* 使用贝叶斯收缩，样本越少越向 50% 收敛，样本量被显式考虑。
        - weighted_* 按时间衰减加权，最近的比赛权重更高。
        """
        if now is None:
            now = time.time()
        matches = self._history
        total = len(matches)
        wins = sum(1 for m in matches if m["win"])
        kills = sum(m["kills"] for m in matches)
        deaths = sum(m["deaths"] for m in matches)
        damage = sum(m["damage"] for m in matches)
        hits = sum(m["hits"] for m in matches)
        shots = sum(m["shots"] for m in matches)

        win_rate = wins / total if total else 0.0
        accuracy = hits / shots if shots else 0.0
        kd_ratio = kills / deaths if deaths else float(kills)
        avg_damage = damage / total if total else 0.0

        # 样本量收缩（贝叶斯）：小样本向先验均值 0.5 收敛
        prior_m = self._prior_matches
        prior_s = self._prior_shots
        adjusted_win_rate = (wins + prior_m * 0.5) / (total + prior_m) if total or prior_m else 0.0
        adjusted_accuracy = (hits + prior_s * 0.5) / (shots + prior_s) if shots or prior_s else 0.0
        confidence = total / (total + prior_m) if total or prior_m else 0.0

        # 时间衰减加权：最近的比赛权重更高
        weights = [self._time_weight(m["timestamp"], now) for m in matches]
        weight_sum = sum(weights)
        if weight_sum > 0:
            weighted_win_rate = sum(w * (1.0 if m["win"] else 0.0)
                                    for w, m in zip(weights, matches)) / weight_sum
            weighted_accuracy = sum(w * self._match_accuracy(m)
                                    for w, m in zip(weights, matches)) / weight_sum
        else:
            weighted_win_rate = 0.0
            weighted_accuracy = 0.0

        return {
            "matches": total,
            "wins": wins,
            "losses": total - wins,
            "kills": kills,
            "deaths": deaths,
            "damage": damage,
            "hits": hits,
            "shots": shots,
            "win_rate": round(win_rate, 4),
            "accuracy": round(accuracy, 4),
            "kd_ratio": round(kd_ratio, 4),
            "avg_damage": round(avg_damage, 4),
            "adjusted_win_rate": round(adjusted_win_rate, 4),
            "adjusted_accuracy": round(adjusted_accuracy, 4),
            "weighted_win_rate": round(weighted_win_rate, 4),
            "weighted_accuracy": round(weighted_accuracy, 4),
            "confidence": round(confidence, 4),
        }

    # ------------------------------------------------------------------
    # 趋势分析
    # ------------------------------------------------------------------
    @staticmethod
    def _metric_value(metric: str, matches: List[Dict]) -> float:
        if not matches:
            return 0.0
        if metric == "win_rate":
            return sum(1 for m in matches if m["win"]) / len(matches)
        if metric == "accuracy":
            shots = sum(m["shots"] for m in matches)
            return sum(m["hits"] for m in matches) / shots if shots else 0.0
        if metric == "kills":
            return sum(m["kills"] for m in matches) / len(matches)
        if metric == "damage":
            return sum(m["damage"] for m in matches) / len(matches)
        if metric == "kd_ratio":
            deaths = sum(m["deaths"] for m in matches)
            kills = sum(m["kills"] for m in matches)
            return kills / deaths if deaths else float(kills)
        raise ValueError(f"未知的趋势指标: {metric}")

    def _matches_since(self, cutoff: float) -> List[Dict]:
        return [m for m in self._history if m["timestamp"] >= cutoff]

    def trend_analysis(self, metric: str = "accuracy", days: int = 30,
                       compare_with=None, now: Optional[float] = None) -> Dict:
        """趋势分析。

        - days: 只统计最近 days 天内的比赛（7/30 等时间范围）。
        - 将时间范围内的比赛按时间一分为二，比较前后两段判断走势。
        - compare_with: 另一个 PlayerStats 实例或其列表，用于玩家间对比。
        """
        if now is None:
            now = time.time()
        cutoff = now - days * 86400.0
        recent = sorted(self._matches_since(cutoff), key=lambda m: m["timestamp"])

        half = len(recent) // 2
        first_half, second_half = recent[:half], recent[half:]
        first_value = self._metric_value(metric, first_half)
        second_value = self._metric_value(metric, second_half)
        change = second_value - first_value

        base = abs(first_value)
        relative = change / base if base > 1e-9 else (1.0 if change > 0 else (-1.0 if change < 0 else 0.0))
        if relative > 0.05:
            direction = "improving"
        elif relative < -0.05:
            direction = "declining"
        else:
            direction = "stable"

        result = {
            "metric": metric,
            "days": days,
            "matches": len(recent),
            "direction": direction,
            "first_half": round(first_value, 4),
            "second_half": round(second_value, 4),
            "change": round(change, 4),
            "series": [
                {"timestamp": m["timestamp"], "value": round(self._metric_value(metric, [m]), 4)}
                for m in recent
            ],
        }

        if compare_with is not None:
            others = [compare_with] if isinstance(compare_with, PlayerStats) else list(compare_with)
            my_value = self._metric_value(metric, recent)
            other_values = [self._metric_value(metric, o._matches_since(cutoff)) for o in others]
            players = len(other_values) + 1
            better = sum(1 for v in other_values if my_value > v)
            result["comparison"] = {
                "my_value": round(my_value, 4),
                "others_average": round(sum(other_values) / len(other_values), 4) if other_values else 0.0,
                "rank": 1 + sum(1 for v in other_values if v > my_value),
                "players": players,
                "better_than_percent": round(better / len(other_values) * 100, 2) if other_values else 0.0,
            }
        return result

    # ------------------------------------------------------------------
    # 成就追踪
    # ------------------------------------------------------------------
    def achievement_tracking(self) -> List[Dict]:
        """成就追踪：返回每个成就的解锁状态、进度与稀有度。"""
        stats = self.calculate_stats()
        metric_values = {
            "matches": stats["matches"],
            "wins": stats["wins"],
            "kills": stats["kills"],
            "accuracy_pct": stats["accuracy"] * 100.0,
            "winrate_pct": stats["win_rate"] * 100.0,
        }
        unlocked_count = 0
        achievements = []
        for ach in ACHIEVEMENTS:
            current = metric_values[ach["metric"]]
            target = ach["target"]
            progress = min(1.0, current / target) if target > 0 else 1.0
            min_matches = ach["min_matches"]
            if min_matches > 0:
                # 样本不足时进度受比赛场次限制
                progress = min(progress, stats["matches"] / min_matches)
            unlocked = progress >= 1.0
            if unlocked:
                unlocked_count += 1
            achievements.append({
                "id": ach["id"],
                "name": ach["name"],
                "description": ach["description"],
                "rarity": ach["rarity"],
                "unlocked": unlocked,
                "progress": round(progress, 4),
                "current": round(current, 4),
                "target": target,
            })
        achievements.sort(key=lambda a: (not a["unlocked"], -RARITY_ORDER[a["rarity"]]))
        return achievements

    # ------------------------------------------------------------------
    # 统计导出
    # ------------------------------------------------------------------
    EXPORT_FORMATS = ("json", "csv", "text")

    def stats_export(self, format: str = "json", share: bool = False) -> Dict:
        """导出统计数据。

        format: "json" / "csv" / "text"。
        share:  True 时附带适合社交平台分享的文本与链接。
        """
        if format not in self.EXPORT_FORMATS:
            raise ValueError(f"不支持的导出格式: {format}，支持 {self.EXPORT_FORMATS}")

        stats = self.calculate_stats()
        achievements = self.achievement_tracking()

        if format == "json":
            data = json.dumps(
                {"stats": stats, "achievements": achievements, "matches": self._history},
                ensure_ascii=False, indent=2)
        elif format == "csv":
            buffer = io.StringIO()
            fieldnames = ["match_id", "win", "kills", "deaths", "damage",
                          "hits", "shots", "timestamp"]
            writer = csv.DictWriter(buffer, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._history)
            data = buffer.getvalue()
        else:  # text
            unlocked = [a for a in achievements if a["unlocked"]]
            lines = [
                "===== 玩家数据统计 =====",
                f"比赛场次: {stats['matches']}  胜: {stats['wins']}  负: {stats['losses']}",
                f"胜率: {stats['win_rate']:.2%}  K/D: {stats['kd_ratio']}",
                f"命中率: {stats['accuracy']:.2%}  场均伤害: {stats['avg_damage']}",
                f"成就: {len(unlocked)}/{len(achievements)} 已解锁",
            ]
            data = "\n".join(lines)

        result = {"format": format, "data": data}
        if share:
            unlocked = sum(1 for a in achievements if a["unlocked"])
            share_text = (
                f"我打了 {stats['matches']} 场比赛，胜率 {stats['win_rate']:.1%}，"
                f"命中率 {stats['accuracy']:.1%}，"
                f"已解锁 {unlocked}/{len(achievements)} 个成就！"
            )
            result["share_text"] = share_text
            result["share_url"] = ("https://social.example.com/share?text="
                                   + urllib.parse.quote(share_text))
        return result
