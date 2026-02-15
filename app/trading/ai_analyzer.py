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
        position_info: Optional[dict] = None,
        market_context: Optional[str] = None,
    ) -> Optional[TradeSignal]:
        """
        分析图表并返回交易信号

        Args:
            symbol: 交易对
            chart_5m_path: 5分钟图表路径
            chart_15m_path: 15分钟图表路径
            current_price: 当前价格
            position_info: 持仓信息（可选）
            market_context: 订单簿/成交分布等市场微观文本（可选）

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
            
            # 构建提示词（含可选市场微观数据）
            prompt = self._build_prompt(symbol, current_price, position_info, market_context)
            
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

    def _build_prompt(
        self,
        symbol: str,
        current_price: float,
        position_info: Optional[dict] = None,
        market_context: Optional[str] = None,
    ) -> str:
        """构建分析提示词（位置+形态的结构化交易逻辑，高盈亏比导向）"""
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

        # 市场微观：订单簿、成交分布、逐笔买卖压力（可选）
        micro_section = ""
        if market_context and market_context.strip():
            micro_section = f"""
{market_context}
"""

        return f"""你是一名专业的加密货币剥头皮交易员（Scalper）。你的目标是寻找【高盈亏比（High R:R）】的交易机会，核心优势是识别【关键供需区】和【价格行为（Price Action）】。

当前市场数据:
- 交易对: {symbol}
- 当前价格: {current_price}
{position_context}
{micro_section}

请综合分析 5m 和 15m 两张 K 线图。

### 第一步：市场结构分析（必须在 reasoning 中体现）
1. **关键位置**：识别强支撑位（Support）和强阻力位（Resistance）。
2. **趋势状态**：趋势行情中寻找回调（Pullback）入场；震荡行情中寻找高抛低吸（Range）机会。
3. **K 线形态**：5m 级别的反转或中继信号（吞没、针、双底/顶、突破回踩等）。

### 第二步：交易决策
{"=== 现有持仓管理 ===" if position_info else "=== 寻找开仓机会 ==="}

{'''- close_position：价格遇强阻力、动能衰竭（如背离）、或跌破关键支撑结构时。
- hold：价格仍在健康趋势中，尚未触及止损/止盈目标。
- adjust_stops：仅在价格突破关键前高/前低后，移动止损以锁定利润。
- open_long/open_short：极少使用，仅在发生明确结构性破坏（趋势反转）时反向开仓。

止损止盈方向（必须遵守）：
- 做多：止损 < 入场价，止盈 > 入场价；保护利润 = 止损移到入场价之上。
- 做空：止损 > 入场价，止盈 < 入场价；保护利润 = 止损移到入场价之下。''' if position_info else '''- wait：无明显形态、价格在“真空地带”（不在关键支撑/阻力附近）、或多空不明朗。

**重要**：当价格已进入关键支撑/阻力附近（与关键位距离约 1% 以内）且 5m 有企稳或止跌迹象时，即可考虑开仓，不必等待“完美触碰+标准反转 K 线”。给出具体 stop_loss、take_profit 和 confidence（60+）即可。'''}

### 第三步：风控规则（严格执行）
- **止损**：基于图表结构（前低之下/前高之上），不要用固定百分比。
- **盈亏比 R:R**：止盈空间至少为止损空间的 1.5 倍；不划算则选 wait。
- **置信度**：60–70 一般机会（如接近关键位+企稳）；70–85 优质结构；85+ 完美共振（大周期支撑 + 小周期突破）。

请只输出一个 JSON 对象，不要 markdown 或其它文字。开仓时必须带 stop_loss、take_profit、confidence；wait/close_position/hold 可不带止损止盈。

{'''{{"symbol": "''' + symbol + '''", "action": "close_position", "reasoning": "..."}}
{{"symbol": "''' + symbol + '''", "action": "hold", "reasoning": "..."}}
{{"symbol": "''' + symbol + '''", "action": "adjust_stops", "stop_loss": 新止损价, "take_profit": 新止盈价, "reasoning": "..."}}
{{"symbol": "''' + symbol + '''", "action": "open_long", "stop_loss": 止损价, "take_profit": 止盈价, "confidence": 60-100, "reasoning": "..."}}
{{"symbol": "''' + symbol + '''", "action": "open_short", "stop_loss": 止损价, "take_profit": 止盈价, "confidence": 60-100, "reasoning": "..."}}''' if position_info else '''{{"symbol": "''' + symbol + '''", "action": "wait", "reasoning": "..."}}
{{"symbol": "''' + symbol + '''", "action": "open_long", "stop_loss": 止损价, "take_profit": 止盈价, "confidence": 60-100, "reasoning": "趋势+关键位置+形态+盈亏比"}}
{{"symbol": "''' + symbol + '''", "action": "open_short", "stop_loss": 止损价, "take_profit": 止盈价, "confidence": 60-100, "reasoning": "趋势+关键位置+形态+盈亏比"}}'''}"""
    
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
            # 清洗：去掉可能的 Markdown 代码块标记（防止 Claude 输出 ```json ... ```）
            response = response.replace("```json", "").replace("```", "").strip()

            # 提取 JSON
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

            # 开仓置信度门槛：低于 60 强制视为观望；60+ 交给仓位/风控层处理
            if action in (SignalAction.OPEN_LONG, SignalAction.OPEN_SHORT) and confidence < 60:
                action = SignalAction.WAIT
                logger.info(f"🤖 置信度 {confidence}% < 60，已强制改为 wait")

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
