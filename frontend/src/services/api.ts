import axios from 'axios';

// 创建axios实例
const api = axios.create({
  baseURL: '/api',
  timeout: 10000,
});

// 响应拦截器
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    console.error('API请求失败:', error);
    return Promise.reject(error);
  }
);

// 账户相关API
export const getAccountStatus = () => api.get('/account/status');
export const getAccountBalance = () => api.get('/account/balance');
export const getAccountSummary = () => api.get('/account/summary');

// 持仓相关API
export const getPositions = (symbol?: string) => 
  api.get('/positions/', { params: { symbol } });
export const getPositionsSummary = () => api.get('/positions/summary');
export const getPositionRisk = () => api.get('/positions/risk');
export const getPositionBySymbol = (symbol: string) => 
  api.get(`/positions/${symbol}`);

// 信号相关API
export const getSignals = (params?: { symbol?: string; hours?: number; limit?: number }) => 
  api.get('/signals/', { params });
export const generateSignal = (symbol: string, force = false) => 
  api.post('/signals/generate', { symbol, force });
export const getLatestSignal = (symbol?: string) => 
  api.get('/signals/latest', { params: { symbol } });
export const getSignalPerformance = (symbol?: string, days = 7) => 
  api.get('/signals/performance', { params: { symbol, days } });

// 交易相关API
export const executeTrade = (symbol: string, action: string, quantity?: number) => 
  api.post('/trading/execute', { symbol, action, quantity });
export const setTradingMode = (mode: string) => 
  api.post('/trading/mode', { mode });
export const getTradingStatus = () => api.get('/trading/status');
export const getTradingPerformance = () => api.get('/trading/performance');
export const closePosition = (symbol: string) => 
  api.post(`/trading/close/${symbol}`);

// 训练相关API
export const startTraining = (force_retrain = false) => 
  api.post('/training/start', { force_retrain });
export const getTrainingStatus = () => api.get('/training/status');
export const getTrainingMetrics = () => api.get('/training/metrics');
export const runTrainingTask = () => api.post('/training/schedule/run');

// 绩效相关API
export const getPerformance = (symbol?: string, days = 30) => 
  api.get('/performance/', { params: { symbol, days } });
export const getRiskMetrics = (symbol?: string) => 
  api.get('/performance/risk', { params: { symbol } });
export const getDrawdownMetrics = (symbol?: string) => 
  api.get('/performance/drawdown', { params: { symbol } });
export const getPerformanceRatios = (symbol?: string) => 
  api.get('/performance/ratios', { params: { symbol } });

// 系统相关API
export const getSystemStatus = () => api.get('/system/status');
export const controlSystem = (action: string) => 
  api.post('/system/control', { action });
export const getSystemHealth = () => api.get('/system/health');
export const getSystemInfo = () => api.get('/system/info');
export const getSystemTasks = () => api.get('/system/tasks');
export const runSystemTask = (taskName: string) => 
  api.post(`/system/tasks/${taskName}/run`);

// WebSocket统计
export const getWebSocketStats = () => api.get('/ws/stats');