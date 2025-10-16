import React, { createContext, useContext, useState, useEffect } from 'react';
import { useWebSocket } from './WebSocketContext';
import * as api from '../services/api';

interface AccountInfo {
  total_wallet_balance: number;
  total_unrealized_pnl: number;
  available_balance: number;
  can_trade: boolean;
}

interface PositionInfo {
  symbol: string;
  side: string;
  size: number;
  entry_price: number;
  mark_price: number;
  unrealized_pnl: number;
  percentage: number;
}

interface TradingSignal {
  timestamp: string;
  symbol: string;
  signal_type: string;
  confidence: number;
  entry_price: number;
}

interface SystemStatus {
  system_state: string;
  auto_trading_enabled: boolean;
  trading_mode: string;
  services: Record<string, boolean>;
}

interface DataContextType {
  accountInfo: AccountInfo | null;
  positions: PositionInfo[];
  signals: TradingSignal[];
  systemStatus: SystemStatus | null;
  currentPrice: number;
  loading: boolean;
  error: string | null;
  refreshData: () => void;
}

const DataContext = createContext<DataContextType | undefined>(undefined);

export const useData = () => {
  const context = useContext(DataContext);
  if (!context) {
    throw new Error('useData must be used within a DataProvider');
  }
  return context;
};

interface DataProviderProps {
  children: React.ReactNode;
}

export const DataProvider: React.FC<DataProviderProps> = ({ children }) => {
  const { lastMessage, subscribe } = useWebSocket();
  
  const [accountInfo, setAccountInfo] = useState<AccountInfo | null>(null);
  const [positions, setPositions] = useState<PositionInfo[]>([]);
  const [signals, setSignals] = useState<TradingSignal[]>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // 处理WebSocket消息
  useEffect(() => {
    if (lastMessage) {
      switch (lastMessage.type) {
        case 'price_update':
          setCurrentPrice(lastMessage.data.price);
          break;
        case 'signal_update':
          setSignals(prev => [lastMessage.data.signal, ...prev.slice(0, 49)]);
          break;
        case 'system_status':
          setSystemStatus(lastMessage.data);
          break;
        default:
          break;
      }
    }
  }, [lastMessage]);

  // 订阅WebSocket频道
  useEffect(() => {
    subscribe('price');
    subscribe('signals');
    subscribe('system');
  }, [subscribe]);

  // 获取数据
  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      // 并行获取所有数据
      const [
        accountResponse,
        positionsResponse,
        signalsResponse,
        systemResponse,
      ] = await Promise.all([
        api.getAccountStatus(),
        api.getPositions(),
        api.getSignals(),
        api.getSystemStatus(),
      ]);

      if (accountResponse.success) {
        setAccountInfo(accountResponse.data);
      }

      if (positionsResponse.success) {
        setPositions(positionsResponse.data);
      }

      if (signalsResponse.success) {
        setSignals(signalsResponse.data);
      }

      if (systemResponse.success) {
        setSystemStatus(systemResponse.data);
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : '获取数据失败');
    } finally {
      setLoading(false);
    }
  };

  // 初始化数据
  useEffect(() => {
    fetchData();
    
    // 定期刷新数据
    const interval = setInterval(fetchData, 30000); // 30秒刷新一次
    
    return () => clearInterval(interval);
  }, []);

  const refreshData = () => {
    fetchData();
  };

  const value: DataContextType = {
    accountInfo,
    positions,
    signals,
    systemStatus,
    currentPrice,
    loading,
    error,
    refreshData,
  };

  return (
    <DataContext.Provider value={value}>
      {children}
    </DataContext.Provider>
  );
};