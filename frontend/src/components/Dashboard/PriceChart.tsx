import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Box, Typography } from '@mui/material';
import { useData } from '../../contexts/DataContext';

interface PriceData {
  timestamp: string;
  price: number;
}

const PriceChart: React.FC = () => {
  const { currentPrice } = useData();
  const [priceHistory, setPriceHistory] = useState<PriceData[]>([]);

  // 模拟价格历史数据
  useEffect(() => {
    const generateMockData = () => {
      const data: PriceData[] = [];
      const now = new Date();
      let basePrice = currentPrice || 2000;
      
      for (let i = 23; i >= 0; i--) {
        const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000);
        // 添加一些随机波动
        const variation = (Math.random() - 0.5) * 100;
        basePrice += variation;
        
        data.push({
          timestamp: timestamp.toISOString(),
          price: Math.max(basePrice, 1000), // 确保价格不会太低
        });
      }
      
      return data;
    };

    if (currentPrice > 0) {
      setPriceHistory(generateMockData());
    }
  }, [currentPrice]);

  // 实时更新最新价格
  useEffect(() => {
    if (currentPrice > 0 && priceHistory.length > 0) {
      setPriceHistory(prev => {
        const newData = [...prev];
        const now = new Date();
        
        // 添加新的价格点
        newData.push({
          timestamp: now.toISOString(),
          price: currentPrice,
        });
        
        // 保持最近24小时的数据
        return newData.slice(-24);
      });
    }
  }, [currentPrice]);

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('zh-CN', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatPrice = (price: number) => {
    return `$${price.toFixed(2)}`;
  };

  if (priceHistory.length === 0) {
    return (
      <Box sx={{ 
        height: '100%', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center' 
      }}>
        <Typography variant="body2" color="text.secondary">
          加载价格数据中...
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', height: '100%' }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={priceHistory}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="timestamp"
            tickFormatter={formatTime}
            tick={{ fontSize: 12 }}
          />
          <YAxis 
            tickFormatter={formatPrice}
            tick={{ fontSize: 12 }}
            domain={['dataMin - 50', 'dataMax + 50']}
          />
          <Tooltip
            labelFormatter={(label) => `时间: ${formatTime(label)}`}
            formatter={(value: number) => [formatPrice(value), '价格']}
          />
          <Line
            type="monotone"
            dataKey="price"
            stroke="#00d4aa"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default PriceChart;