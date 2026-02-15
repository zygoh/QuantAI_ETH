/**
 * AI 模拟交易系统前端
 *
 * 功能：
 * - 账户状态展示
 * - 持仓信息实时更新
 * - AI 对话记录展示
 * - 交易历史
 * - K线图表展示（5m / 15m）
 * - 10秒自动刷新
 */

const CONFIG = {
    API_BASE: '/api',
    REFRESH_INTERVAL: 2000,
    IMAGE_BASE: '/image'
};

let state = {
    currentSymbol: null,
    selectedInterval: '5m',
    isLoading: false
};

// ========== API ==========

async function fetchJSON(url) {
    try {
        const res = await fetch(url);
        const data = await res.json();
        return data.success ? data : null;
    } catch (e) {
        console.error('API 请求失败:', url, e);
        return null;
    }
}

// ========== 渲染 ==========

function renderAccount(data) {
    if (!data) return;
    const d = data.data;

    document.getElementById('balance').textContent = '$' + d.balance.toFixed(2);

    const pnlEl = document.getElementById('total-pnl');
    pnlEl.textContent = '$' + (d.total_pnl >= 0 ? '+' : '') + d.total_pnl.toFixed(2);
    pnlEl.className = 'stat-value ' + (d.total_pnl >= 0 ? 'positive' : 'negative');

    const retEl = document.getElementById('return-pct');
    retEl.textContent = (d.return_pct >= 0 ? '+' : '') + d.return_pct.toFixed(2) + '%';
    retEl.className = 'stat-value ' + (d.return_pct >= 0 ? 'positive' : 'negative');

    document.getElementById('win-rate').textContent = d.win_rate.toFixed(1) + '%';
    document.getElementById('total-trades').textContent = d.total_trades;
    document.getElementById('total-fees').textContent = '$' + d.total_fees.toFixed(4);
}

function renderStatus(data) {
    if (!data) {
        document.getElementById('status-indicator').className = 'status-dot';
        document.getElementById('status-text').textContent = '连接失败';
        return;
    }
    var d = data.data;
    var indicator = document.getElementById('status-indicator');
    var text = document.getElementById('status-text');
    var symbolEl = document.getElementById('current-symbol');

    if (d.is_running) {
        indicator.className = 'status-dot running';
        text.textContent = '运行中';
    } else {
        indicator.className = 'status-dot';
        text.textContent = '启动中...';
    }

    var priceEl = document.getElementById('current-price');

    if (d.current_symbol) {
        symbolEl.textContent = '当前: ' + d.current_symbol;
        if (d.current_price && d.current_price > 0) {
            priceEl.textContent = '$' + formatPrice(d.current_price);
        } else {
            priceEl.textContent = '';
        }
        if (state.currentSymbol !== d.current_symbol) {
            state.currentSymbol = d.current_symbol;
            loadChart(d.current_symbol, state.selectedInterval);
        }
    } else {
        symbolEl.textContent = '';
        priceEl.textContent = '';
    }
}

function renderPosition(data) {
    var emptyEl = document.getElementById('position-empty');
    var infoEl = document.getElementById('position-info');

    if (!data || !data.data) {
        emptyEl.classList.remove('hidden');
        infoEl.classList.add('hidden');
        return;
    }

    emptyEl.classList.add('hidden');
    infoEl.classList.remove('hidden');

    var p = data.data;
    document.getElementById('pos-symbol').textContent = p.symbol;

    var sideEl = document.getElementById('pos-side');
    sideEl.textContent = p.side === 'long' ? '做多' : '做空';
    sideEl.className = 'pos-side ' + (p.side === 'long' ? 'positive' : 'negative');

    document.getElementById('pos-leverage').textContent = p.leverage + 'x';
    document.getElementById('pos-entry').textContent = '$' + formatPrice(p.entry_price);
    document.getElementById('pos-size').textContent = '$' + p.position_size_usd.toFixed(2);
    document.getElementById('pos-sl').textContent = '$' + formatPrice(p.stop_loss);
    document.getElementById('pos-tp').textContent = '$' + formatPrice(p.take_profit);
    document.getElementById('pos-time').textContent = formatTime(p.entry_time);
    // 计算持仓时长
    var entryDate = new Date(p.entry_time);
    var now = new Date();
    var diffSec = Math.floor((now - entryDate) / 1000);
    var hours = Math.floor(diffSec / 3600);
    var minutes = Math.floor((diffSec % 3600) / 60);
    var seconds = diffSec % 60;
    var durationStr = '';
    if (hours > 0) {
        durationStr = hours + 'h ' + minutes + 'm ' + seconds + 's';
    } else if (minutes > 0) {
        durationStr = minutes + 'm ' + seconds + 's';
    } else {
        durationStr = seconds + 's';
    }
    document.getElementById('pos-duration').textContent = durationStr;
    document.getElementById('pos-fee').textContent = '$' + p.entry_fee.toFixed(6);

    // 浮动盈亏
    var pnlEl = document.getElementById('pos-pnl');
    var pnl = p.unrealized_pnl || 0;
    var pnlPct = p.unrealized_pnl_pct || 0;
    pnlEl.textContent = (pnl >= 0 ? '+' : '') + pnl.toFixed(4) + ' (' + (pnlPct >= 0 ? '+' : '') + pnlPct.toFixed(2) + '%)';
    pnlEl.className = 'pos-val ' + (pnl >= 0 ? 'positive' : 'negative');

    // 爆仓价
    if (p.liquidation_price) {
        document.getElementById('pos-liq').textContent = '$' + formatPrice(p.liquidation_price);
    }
}

function renderTrades(data) {
    var emptyEl = document.getElementById('trades-empty');
    var tableEl = document.getElementById('trades-table');
    var tbody = document.getElementById('trades-tbody');

    if (!data || !data.data || data.data.length === 0) {
        if (emptyEl) emptyEl.classList.remove('hidden');
        if (tableEl) tableEl.classList.add('hidden');
        return;
    }

    if (emptyEl) emptyEl.classList.add('hidden');
    if (tableEl) tableEl.classList.remove('hidden');

    var reasonMap = {
        'stop_loss': '止损',
        'take_profit': '止盈',
        'ai_reverse': 'AI反转',
        'ai_close': 'AI平仓',
        'liquidation': '爆仓',
        'new_signal': '新信号',
        'signal': '信号'
    };

    tbody.innerHTML = data.data.map(function(t) {
        var pnlClass = t.pnl >= 0 ? 'positive' : 'negative';
        var sideText = t.side === 'long' ? '多' : '空';
        var sideClass = t.side === 'long' ? 'positive' : 'negative';
        return '<tr>' +
            '<td>' + t.symbol + '</td>' +
            '<td class="' + sideClass + '">' + sideText + '</td>' +
            '<td>$' + formatPrice(t.entry_price) + '</td>' +
            '<td>$' + formatPrice(t.exit_price) + '</td>' +
            '<td class="' + pnlClass + '">' + (t.pnl >= 0 ? '+' : '') + t.pnl.toFixed(4) + ' (' + (t.pnl_pct >= 0 ? '+' : '') + t.pnl_pct.toFixed(2) + '%)</td>' +
            '<td>' + (reasonMap[t.exit_reason] || t.exit_reason) + '</td>' +
            '<td>' + formatTime(t.entry_time) + '</td>' +
            '<td>' + formatTime(t.exit_time) + '</td>' +
            '</tr>';
    }).join('');
}

function renderChat(data) {
    var container = document.getElementById('chat-container');
    var emptyEl = document.getElementById('chat-empty');

    if (!data || !data.data || data.data.length === 0) {
        if (emptyEl) emptyEl.classList.remove('hidden');
        return;
    }

    if (emptyEl) emptyEl.classList.add('hidden');

    var actionMap = {
        'wait': { text: '观望', cls: 'signal-wait' },
        'open_long': { text: '开多', cls: 'signal-long' },
        'open_short': { text: '开空', cls: 'signal-short' },
        'close_position': { text: '平仓', cls: 'signal-short' },
        'hold': { text: '持有', cls: 'signal-long' },
        'adjust_stops': { text: '调整止损', cls: 'signal-adjust' }
    };

    var html = data.data.map(function(chat) {
        var reqTime = formatTime(chat.request_time || chat.timestamp);
        var resTime = formatTime(chat.response_time || chat.timestamp);
        var posTag = chat.has_position ? '<span class="chat-tag tag-pos">持仓中</span>' : '';

        // 用户消息（请求摘要）
        var userBubble =
            '<div class="chat-msg user-msg">' +
                '<div class="chat-meta">' +
                    '<span class="chat-role">📤 请求</span>' +
                    '<span class="chat-time">' + reqTime + '</span>' +
                    posTag +
                '</div>' +
                '<div class="chat-bubble user-bubble">' + escapeHtml(chat.prompt_summary) + '</div>' +
            '</div>';

        // AI 回复
        var signalBadge = '';
        var detail = '';
        if (chat.signal) {
            var s = chat.signal;
            var info = actionMap[s.action] || { text: s.action, cls: '' };
            signalBadge = '<span class="chat-signal ' + info.cls + '">' + info.text + (s.confidence ? ' ' + s.confidence + '%' : '') + '</span>';
            if (s.stop_loss > 0 || s.take_profit > 0) {
                detail = '<div class="chat-detail">止损: $' + formatPrice(s.stop_loss) + ' | 止盈: $' + formatPrice(s.take_profit) + '</div>';
            }
        }

        // AI reasoning
        var reasoning = '';
        if (chat.signal && chat.signal.reasoning) {
            reasoning = chat.signal.reasoning;
        } else if (chat.response) {
            reasoning = chat.response.length > 200 ? chat.response.substring(0, 200) + '...' : chat.response;
        }

        var durationTag = '';
        if (chat.response_duration !== undefined && chat.response_duration !== null) {
            durationTag = '<span class="chat-tag tag-duration">' + chat.response_duration + 's</span>';
        }

        var aiBubble =
            '<div class="chat-msg ai-msg">' +
                '<div class="chat-meta">' +
                    '<span class="chat-role">🤖 AI</span>' +
                    '<span class="chat-time">' + resTime + '</span>' +
                    durationTag +
                    signalBadge +
                '</div>' +
                '<div class="chat-bubble ai-bubble">' + escapeHtml(reasoning) + '</div>' +
                detail +
            '</div>';

        return '<div class="chat-round">' + userBubble + aiBubble + '</div>';
    }).join('');

    container.innerHTML = html;
    container.scrollTop = 0;
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ========== 图表 ==========

function loadChart(symbol, interval) {
    var img = document.getElementById('chart-image');
    var placeholder = document.getElementById('chart-placeholder');

    if (!symbol) {
        img.style.display = 'none';
        placeholder.classList.remove('hidden');
        return;
    }

    var url = CONFIG.IMAGE_BASE + '/' + symbol + '/' + symbol + '_' + interval + '.png?t=' + Date.now();
    var testImg = new Image();

    testImg.onload = function() {
        img.src = url;
        img.style.display = 'block';
        placeholder.classList.add('hidden');
    };

    testImg.onerror = function() {
        img.style.display = 'none';
        placeholder.textContent = symbol + ' ' + interval + ' 图表暂未生成';
        placeholder.classList.remove('hidden');
    };

    testImg.src = url;
}

// ========== 工具函数 ==========

function formatPrice(price) {
    if (price >= 1000) return price.toFixed(2);
    if (price >= 1) return price.toFixed(4);
    if (price >= 0.01) return price.toFixed(6);
    return price.toFixed(8);
}

function formatTime(isoStr) {
    var d = new Date(isoStr);
    var month = (d.getMonth() + 1).toString().padStart(2, '0');
    var day = d.getDate().toString().padStart(2, '0');
    var hour = d.getHours().toString().padStart(2, '0');
    var min = d.getMinutes().toString().padStart(2, '0');
    var sec = d.getSeconds().toString().padStart(2, '0');
    return month + '-' + day + ' ' + hour + ':' + min + ':' + sec;
}

// ========== 主循环 ==========

async function refresh() {
    if (state.isLoading) return;
    state.isLoading = true;

    try {
        var results = await Promise.allSettled([
            fetchJSON(CONFIG.API_BASE + '/status'),
            fetchJSON(CONFIG.API_BASE + '/account'),
            fetchJSON(CONFIG.API_BASE + '/position'),
            fetchJSON(CONFIG.API_BASE + '/trades?limit=20'),
            fetchJSON(CONFIG.API_BASE + '/chat?limit=20')
        ]);

        var vals = results.map(function(r) { return r.status === 'fulfilled' ? r.value : null; });

        renderStatus(vals[0]);
        renderAccount(vals[1]);
        renderPosition(vals[2]);
        renderTrades(vals[3]);
        renderChat(vals[4]);

        if (state.currentSymbol) {
            loadChart(state.currentSymbol, state.selectedInterval);
        }
    } catch (e) {
        console.error('刷新失败:', e);
    } finally {
        state.isLoading = false;
    }
}

function updateLayout() {
    var mainContent = document.querySelector('.main-content');
    var midPanel = document.querySelector('.mid-panel');
    var rightPanel = document.querySelector('.right-panel');
    var notesPanel = document.querySelector('.notes-panel');
    var midVisible = !midPanel.classList.contains('hidden');
    var rightVisible = !rightPanel.classList.contains('hidden');
    var notesVisible = !notesPanel.classList.contains('hidden');
    var cols = 1 + (midVisible ? 1 : 0) + (rightVisible ? 1 : 0) + (notesVisible ? 1 : 0);

    mainContent.classList.remove('cols-1', 'cols-2', 'cols-3', 'cols-4');
    mainContent.classList.add('cols-' + cols);
}

// ========== 留言板 ==========

var NOTES_KEY = 'quantai_notes';

function loadNotes() {
    try {
        var raw = localStorage.getItem(NOTES_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch (e) {
        return [];
    }
}

function saveNotes(notes) {
    localStorage.setItem(NOTES_KEY, JSON.stringify(notes));
}

function renderNotes() {
    var list = document.getElementById('notes-list');
    var notes = loadNotes();

    if (notes.length === 0) {
        list.innerHTML = '<div class="empty-state">暂无留言</div>';
        return;
    }

    var html = notes.map(function(n, i) {
        return '<div class="note-item">' +
            '<div class="note-text">' + escapeHtml(n.text) + '</div>' +
            '<div class="note-meta">' +
                '<span class="note-time">' + n.time + '</span>' +
                '<button class="note-delete" data-index="' + i + '">删除</button>' +
            '</div>' +
        '</div>';
    }).join('');
    list.innerHTML = html;

    // 绑定删除事件
    list.querySelectorAll('.note-delete').forEach(function(btn) {
        btn.addEventListener('click', function() {
            var idx = parseInt(btn.dataset.index);
            var notes = loadNotes();
            notes.splice(idx, 1);
            saveNotes(notes);
            renderNotes();
        });
    });
}

function addNote() {
    var input = document.getElementById('note-input');
    var text = input.value.trim();
    if (!text) return;

    var notes = loadNotes();
    var now = new Date();
    var month = (now.getMonth() + 1).toString().padStart(2, '0');
    var day = now.getDate().toString().padStart(2, '0');
    var hour = now.getHours().toString().padStart(2, '0');
    var min = now.getMinutes().toString().padStart(2, '0');
    var sec = now.getSeconds().toString().padStart(2, '0');
    var timeStr = month + '-' + day + ' ' + hour + ':' + min + ':' + sec;

    notes.unshift({ text: text, time: timeStr });
    saveNotes(notes);
    input.value = '';
    renderNotes();
}

function initNotes() {
    document.getElementById('add-note').addEventListener('click', addNote);

    // 回车发布（Shift+Enter 换行）
    document.getElementById('note-input').addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            addNote();
        }
    });

    document.getElementById('clear-notes').addEventListener('click', function() {
        if (confirm('确定清空所有留言？')) {
            saveNotes([]);
            renderNotes();
        }
    });

    renderNotes();
}

function init() {
    var toggleChartBtn = document.getElementById('toggle-chart');
    var toggleTradesBtn = document.getElementById('toggle-trades');
    var toggleNotesBtn = document.getElementById('toggle-notes');
    var rightPanel = document.querySelector('.right-panel');
    var midPanel = document.querySelector('.mid-panel');
    var notesPanel = document.querySelector('.notes-panel');

    // 默认都隐藏
    updateLayout();

    toggleTradesBtn.addEventListener('click', function() {
        midPanel.classList.toggle('hidden');
        toggleTradesBtn.classList.toggle('active');
        updateLayout();
    });

    toggleChartBtn.addEventListener('click', function() {
        rightPanel.classList.toggle('hidden');
        toggleChartBtn.classList.toggle('active');
        if (!rightPanel.classList.contains('hidden') && state.currentSymbol) {
            loadChart(state.currentSymbol, state.selectedInterval);
        }
        updateLayout();
    });

    toggleNotesBtn.addEventListener('click', function() {
        notesPanel.classList.toggle('hidden');
        toggleNotesBtn.classList.toggle('active');
        updateLayout();
    });

    // 初始化留言板
    initNotes();

    document.querySelectorAll('.tab-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.tab-btn').forEach(function(b) {
                b.classList.remove('active');
            });
            btn.classList.add('active');
            state.selectedInterval = btn.dataset.interval;
            if (state.currentSymbol) {
                loadChart(state.currentSymbol, state.selectedInterval);
            }
        });
    });

    refresh();
    setInterval(refresh, CONFIG.REFRESH_INTERVAL);
}

document.addEventListener('DOMContentLoaded', init);
