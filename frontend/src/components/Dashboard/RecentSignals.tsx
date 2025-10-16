import React from 'react';
import {
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  Chip,
  Box,
  Skeleton,
} from '@mui/material';
import { formatDistanceToNow } from 'date-fns';
import { zhCN } from 'date-fns/locale';

interface TradingSignal {
  timestamp: string;
  symbol: string;
  signal_type: string;
  confidence: number;
  entry_price: number;
}

interface RecentSignalsProps {
  signals: TradingSignal[];
  loading: boolean;
}

const RecentSignals: React.FC<RecentSignalsProps> = ({ signals, loading }) => {
  if (loading) {
    return (
      <Paper sx={{ p: 2, height: 400 }}>
        <Skeleton variant="text" width="40%" height={32} />
        <Box sx={{ mt: 2 }}>
          {[...Array(5)].map((_, index) => (
            <Skeleton key={index} variant="rectangular" height={60} sx={{ mb: 1 }} />
          ))}
        </Box>
      </Paper>
    );
  }

  const getSignalColor = (signalType: string) => {
    switch (signalType) {
      case 'LONG':
        return 'success';
      case 'SHORT':
        return 'error';
      default:
        return 'default';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  return (
    <Paper sx={{ p: 2, height: 400, overflow: 'auto' }}>
      <Typography variant="h6" gutterBottom>
        最近信号
      </Typography>
      
      {signals.length === 0 ? (
        <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
          暂无信号
        </Typography>
      ) : (
        <List dense>
          {signals.slice(0, 10).map((signal, index) => (
            <ListItem key={index} divider>
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Chip
                      label={signal.signal_type}
                      color={getSignalColor(signal.signal_type) as any}
                      size="small"
                    />
                    <Chip
                      label={`${(signal.confidence * 100).toFixed(0)}%`}
                      color={getConfidenceColor(signal.confidence) as any}
                      size="small"
                      variant="outlined"
                    />
                  </Box>
                }
                secondary={
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      {signal.symbol} @ ${signal.entry_price.toFixed(2)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {formatDistanceToNow(new Date(signal.timestamp), {
                        addSuffix: true,
                        locale: zhCN,
                      })}
                    </Typography>
                  </Box>
                }
              />
            </ListItem>
          ))}
        </List>
      )}
    </Paper>
  );
};

export default RecentSignals;