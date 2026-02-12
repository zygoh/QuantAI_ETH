# -*- coding: utf-8 -*-
"""
AI 图表分析器

调用 Claude API 分析 K 线图表，返回交易信号。
"""

import base64
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import httpx

from app.trading.models import TradeSignal, SignalAction


logger = logging.getLogger(__name__)


# Claude API 配置
ANTHROPIC_BASE_URL = "https://www.aiproxies.cc"
ANTHROPIC_AUTH_TOKEN = "sk-7ce7f8c3c1355bd2f868b6e5b95a43839272390d444cccee7343ae704a921dcc"


class AIAnalyzer:
    """
    AI 图表分析器
    
    使用 Claude API 分析 K 线图表，返回交易信号。
    """
    
    def __init__(self) -> None:
        """初始化分析器"""
        self.base_url = ANTHROPIC_BASE_URL
        self.api_key = ANTHROPIC_AUTH_TOKEN
        # self.model = "claude-opus-4-5-20251101"
        self.model = "claude-opus-4-6"
        self.timeout = 120.0
        self.chat_history: List[Dict] = []  # AI 对话记录
        self._max_history: int = 50  # 最多保留 50 条

        logger.info("🤖 AI 分析器初始化完成")
    
    async def analyze_charts(
        self,
        symbol: str,
        chart_5m_path: str,
        chart_15m_path: str,
        current_price: float,
        position_info: Optional[dict] = None
    ) -> Optional[TradeSignal]:
        """
        分析图表并返回交易信号
        
        Args:
            symbol: 交易对
            chart_5m_path: 5分钟图表路径
            chart_15m_path: 15分钟图表路径
            current_price: 当前价格
            
        Returns:
            交易信号
        """
        try:
            # 读取图表文件
            images = []
            for path in [chart_5m_path, chart_15m_path]:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        img_data = base64.standard_b64encode(f.read()).decode("utf-8")
                        images.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_data
                            }
                        })
            
            if len(images) < 2:
                logger.warning(f"⚠️ {symbol} 图表文件不完整")
                return None
            
            # 构建提示词
            prompt = self._build_prompt(symbol, current_price, position_info)
            
            logger.info(f"🤖 AI 信号: {prompt}")

            # 记录请求时间
            request_time = datetime.now()
            
            # 调用 Claude API
            response = await self._call_claude_api(images, prompt)

            if response:
                # 计算耗时
                response_duration = round((datetime.now() - request_time).total_seconds(), 1)
                signal = self._parse_response(symbol, response)
                # 记录对话（含请求时间和回复时间）
                self._record_chat(symbol, current_price, prompt, response, signal, position_info, request_time)
                if signal:
                    logger.info(
                        f"🤖 AI 信号: {symbol} -> {signal.action.value}, "
                        f"置信度: {signal.confidence}%, "
                        f"耗时: {response_duration}s, "
                        f"原因: {signal.reasoning}"
                    )
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ AI 分析失败: {e}", exc_info=True)
            return None

    def _build_prompt(self, symbol: str, current_price: float, position_info: Optional[dict] = None) -> str:
        """构建分析提示词"""
        # 持仓上下文
        position_context = ""
        if position_info:
            side_cn = "做多" if position_info["side"] == "long" else "做空"
            pnl = position_info.get("unrealized_pnl", 0)
            pnl_pct = position_info.get("unrealized_pnl_pct", 0)
            position_context = f"""
当前持仓信息:
- 方向: {side_cn}
- 入场价格: {position_info['entry_price']}
- 仓位大小: ${position_info['position_size_usd']:.2f}
- 杠杆: {position_info['leverage']}x
- 当前止损: {position_info['stop_loss']}
- 当前止盈: {position_info['take_profit']}
- 浮动盈亏: ${pnl:+.4f} ({pnl_pct:+.2f}%)
"""
        else:
            position_context = "\n当前无持仓\n"

        return f"""你是一个专业的加密货币短线交易分析师。目标：提高胜率，宁可少做不可做错。请分析这两张 K 线图表（5 分钟 + 15 分钟）。

交易对: {symbol}
当前价格: {current_price}
{position_context}

分析框架（严格遵守）：
- 15 分钟图：定方向（上升/下降/震荡）。仅当 15m 有明确趋势时才考虑开仓。
- 5 分钟图：定入场时机。开仓必须满足「15m 与 5m 方向一致」；若 15m 震荡或双周期方向矛盾，一律选择 wait。
- 默认倾向观望：不确定、多空均衡、或置信度不足时，必须选 wait；只有多周期共振、结构清晰时才开仓。

{"=== 当前有持仓，你必须从以下动作中选一个 ===" if position_info else "=== 当前无持仓，你必须从以下动作中选一个 ==="}

{'''1. close_position - 主动平仓（趋势反转、动能衰竭、达到预期、或风险大于收益时果断平仓）
2. hold - 继续持有（趋势延续良好、无反转信号时保持不动）
3. adjust_stops - 调整止盈止损（仅当明确需要移动止损保护利润时使用，禁止频繁调整）
4. open_long / open_short - 反向开仓（仅当明确趋势反转且需要反向操作时）

原则：盈利且动能减弱优先 close_position；趋势良好选 hold；浮亏扩大且趋势不利果断 close_position。

止损止盈方向（必须遵守）：
- 做多：止损 < 入场价，止盈 > 入场价；保护利润 = 止损移到入场价之上
- 做空：止损 > 入场价，止盈 < 入场价；保护利润 = 止损移到入场价之下''' if position_info else '''1. wait - 观望（默认首选。震荡、双周期不一致、或置信度 < 70 时必须选此项）
2. open_long - 开多（仅当 15m 与 5m 均支持做多且置信度 ≥ 70）
3. open_short - 开空（仅当 15m 与 5m 均支持做空且置信度 ≥ 70）

开仓硬性条件：15m 有明确趋势 + 5m 与 15m 同向 + 置信度 ≥ 70；否则必须 wait。
止损止盈：止损距离不宜过大（相对当前价约 1%~3% 内可接受）；止盈建议至少 1.2 倍止损距离，保证盈亏比。'''}

JSON 格式（只返回一个 JSON 对象）：

{'''{{"symbol": "''' + symbol + '''", "action": "close_position", "reasoning": "平仓原因"}}
{{"symbol": "''' + symbol + '''", "action": "hold", "reasoning": "继续持有原因"}}
{{"symbol": "''' + symbol + '''", "action": "adjust_stops", "stop_loss": 新止损价, "take_profit": 新止盈价, "reasoning": "调整原因"}}
{{"symbol": "''' + symbol + '''", "action": "open_long", "stop_loss": 止损价, "take_profit": 止盈价, "confidence": 置信度0-100, "reasoning": "反向开多原因"}}
{{"symbol": "''' + symbol + '''", "action": "open_short", "stop_loss": 止损价, "take_profit": 止盈价, "confidence": 置信度0-100, "reasoning": "反向开空原因"}}''' if position_info else '''{{"symbol": "''' + symbol + '''", "action": "wait", "reasoning": "观望原因"}}
{{"symbol": "''' + symbol + '''", "action": "open_long", "stop_loss": 止损价, "take_profit": 止盈价, "confidence": 70-100, "reasoning": "开多原因"}}
{{"symbol": "''' + symbol + '''", "action": "open_short", "stop_loss": 止损价, "take_profit": 止盈价, "confidence": 70-100, "reasoning": "开空原因"}}'''}

注意：
1. 止损/止盈为数字，且符合方向规则；开仓时 confidence 低于 70 应选 wait。
2. 杠杆与仓位由系统管理，无需在 JSON 中指定。
3. 只返回一段 JSON，不要 markdown、不要其他文字。"""
    
    async def _call_claude_api(
        self,
        images: list,
        prompt: str
    ) -> Optional[str]:
        """调用 Claude API"""
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            # 构建消息内容
            content = images + [{"type": "text", "text": prompt}]
            
            payload = {
                "model": self.model,
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": content}
                ]
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/messages",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("content") and len(data["content"]) > 0:
                        return data["content"][0].get("text", "")
                else:
                    error_text = response.text[:200] if len(response.text) > 200 else response.text
                    logger.error(f"❌ Claude API 错误: {response.status_code} - {error_text}")
                    
        except Exception as e:
            logger.error(f"❌ Claude API 调用失败: {e}")
        
        return None
    
    def _parse_response(self, symbol: str, response: str) -> Optional[TradeSignal]:
        """解析 AI 响应"""
        try:
            # 提取 JSON
            response = response.strip()

            # 尝试找到 JSON 部分
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                logger.warning(f"⚠️ 无法从响应中提取 JSON: {response[:100]}")
                return None

            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            # 解析动作
            action_str = data.get("action", "wait").lower()
            action_map = {
                "wait": SignalAction.WAIT,
                "open_long": SignalAction.OPEN_LONG,
                "open_short": SignalAction.OPEN_SHORT,
                "close_position": SignalAction.CLOSE_POSITION,
                "hold": SignalAction.HOLD,
                "adjust_stops": SignalAction.ADJUST_STOPS
            }
            action = action_map.get(action_str, SignalAction.WAIT)
            confidence = data.get("confidence", 0)

            # 开仓置信度门槛：低于 70 强制视为观望，提高胜率
            if action in (SignalAction.OPEN_LONG, SignalAction.OPEN_SHORT) and confidence < 70:
                action = SignalAction.WAIT
                logger.info(f"🤖 置信度 {confidence}% < 70，已强制改为 wait")

            return TradeSignal(
                symbol=data.get("symbol", symbol),
                action=action,
                reasoning=data.get("reasoning", ""),
                leverage=data.get("leverage", 10),
                position_size_usd=data.get("position_size_usd", 200.0),
                stop_loss=data.get("stop_loss", 0.0),
                take_profit=data.get("take_profit", 0.0),
                confidence=confidence,
                risk_usd=data.get("risk_usd", 0.0)
            )

        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON 解析失败: {e}, 响应: {response[:200]}")
        except Exception as e:
            logger.warning(f"⚠️ 响应解析失败: {e}")

        return None

    def _record_chat(
        self,
        symbol: str,
        current_price: float,
        prompt: str,
        response: str,
        signal: Optional[TradeSignal],
        position_info: Optional[dict],
        request_time: Optional[datetime] = None,
    ) -> None:
        """记录 AI 对话"""
        record: Dict = {
            "request_time": (request_time or datetime.now()).isoformat(),
            "response_time": datetime.now().isoformat(),
            "response_duration": round((datetime.now() - (request_time or datetime.now())).total_seconds(), 1),
            "symbol": symbol,
            "current_price": current_price,
            "has_position": position_info is not None,
            "prompt_summary": f"分析 {symbol} @ ${current_price}",
            "response": response,
            "signal": None
        }
        if signal:
            record["signal"] = {
                "action": signal.action.value,
                "confidence": signal.confidence,
                "reasoning": signal.reasoning,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit
            }
        if position_info:
            record["position"] = {
                "side": position_info["side"],
                "entry_price": position_info["entry_price"],
                "unrealized_pnl": position_info.get("unrealized_pnl", 0)
            }

        self.chat_history.append(record)
        # 限制历史长度
        if len(self.chat_history) > self._max_history:
            self.chat_history = self.chat_history[-self._max_history:]

    def get_chat_history(self, limit: int = 20) -> List[Dict]:
        """获取最近的对话记录"""
        return list(reversed(self.chat_history))[:limit]


# 全局 AI 分析器实例
ai_analyzer = AIAnalyzer()
