import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';

interface WebSocketMessage {
  type: string;
  data: any;
  timestamp?: string;
}

interface WebSocketContextType {
  isConnected: boolean;
  lastMessage: WebSocketMessage | null;
  sendMessage: (message: any) => void;
  subscribe: (channel: string) => void;
  unsubscribe: (channel: string) => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

interface WebSocketProviderProps {
  children: React.ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  const maxReconnectAttempts = 5;
  const reconnectInterval = 3000;

  const connect = useCallback(() => {
    try {
      const wsUrl = `ws://${window.location.host}/api/ws/connect`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('WebSocket连接已建立');
        setIsConnected(true);
        setReconnectAttempts(0);
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          setLastMessage(message);
        } catch (error) {
          console.error('解析WebSocket消息失败:', error);
        }
      };

      ws.onclose = () => {
        console.log('WebSocket连接已关闭');
        setIsConnected(false);
        
        // 自动重连
        if (reconnectAttempts < maxReconnectAttempts) {
          setTimeout(() => {
            setReconnectAttempts(prev => prev + 1);
            connect();
          }, reconnectInterval);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket错误:', error);
      };

      setSocket(ws);
    } catch (error) {
      console.error('WebSocket连接失败:', error);
    }
  }, [reconnectAttempts]);

  useEffect(() => {
    connect();

    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, [connect]);

  const sendMessage = useCallback((message: any) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket未连接，无法发送消息');
    }
  }, [socket]);

  const subscribe = useCallback((channel: string) => {
    sendMessage({
      type: 'subscribe',
      channel: channel,
    });
  }, [sendMessage]);

  const unsubscribe = useCallback((channel: string) => {
    sendMessage({
      type: 'unsubscribe',
      channel: channel,
    });
  }, [sendMessage]);

  const value: WebSocketContextType = {
    isConnected,
    lastMessage,
    sendMessage,
    subscribe,
    unsubscribe,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};