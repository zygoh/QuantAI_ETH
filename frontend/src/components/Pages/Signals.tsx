import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
} from '@mui/material';
import { useData } from '../../contexts/DataContext';
import * as api from '../../services/api';

const Signals: React.FC = () => {
  const { signals } = useData();
  const [loading, setLoading] = useState(false);

  const generateSignal = async () => {
    setLoading(true);
    try {
      await api.generateSignal('ETHUSDT', true);
    } catch (error) {
      console.error('生成信号失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSignalColor = (signalType: string) => {
    switch (signalType) {
      case 'LONG': return 'success';
      case 'SHORT': return 'error';
      default: return 'default';
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        交易信号
      </Typography>

      <Paper sx={{ p: 2, mb: 3 }}>
        <Button
          variant="contained"
          onClick={generateSignal}
          disabled={loading}
        >
          生成信号
        </Button>
      </Paper>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>时间</TableCell>
              <TableCell>交易对</TableCell>
              <TableCell>信号类型</TableCell>
              <TableCell>置信度</TableCell>
              <TableCell>入场价格</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {signals.map((signal, index) => (
              <TableRow key={index}>
                <TableCell>
                  {new Date(signal.timestamp).toLocaleString()}
                </TableCell>
                <TableCell>{signal.symbol}</TableCell>
                <TableCell>
                  <Chip
                    label={signal.signal_type}
                    color={getSignalColor(signal.signal_type) as any}
                    size="small"
                  />
                </TableCell>
                <TableCell>{(signal.confidence * 100).toFixed(1)}%</TableCell>
                <TableCell>${signal.entry_price.toFixed(2)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default Signals;