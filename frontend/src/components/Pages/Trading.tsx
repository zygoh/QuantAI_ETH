import React, { useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Card,
  CardContent,
  Switch,
  FormControlLabel,
} from '@mui/material';
import { useData } from '../../contexts/DataContext';
import * as api from '../../services/api';

const Trading: React.FC = () => {
  const { systemStatus, currentPrice } = useData();
  const [quantity, setQuantity] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  const handleTrade = async (action: string) => {
    if (!quantity && action !== 'CLOSE') {
      setMessage('请输入交易数量');
      return;
    }

    setLoading(true);
    try {
      const response = await api.executeTrade('ETHUSDT', action, parseFloat(quantity));
      setMessage(response.message);
    } catch (error) {
      setMessage('交易失败');
    } finally {
      setLoading(false);
    }
  };

  const handleModeChange = async (mode: string) => {
    try {
      const response = await api.setTradingMode(mode);
      setMessage(response.message);
    } catch (error) {
      setMessage('模式切换失败');
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        交易控制
      </Typography>

      <Grid container spacing={3}>
        {/* 交易模式控制 */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              交易模式
            </Typography>
            <FormControlLabel
              control={
                <Switch
                  checked={systemStatus?.auto_trading_enabled || false}
                  onChange={(e) => handleModeChange(e.target.checked ? 'AUTO' : 'SIGNAL_ONLY')}
                />
              }
              label={systemStatus?.auto_trading_enabled ? '自动交易' : '信号模式'}
            />
          </Paper>
        </Grid>

        {/* 手动交易 */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              手动交易
            </Typography>
            <TextField
              fullWidth
              label="交易数量"
              value={quantity}
              onChange={(e) => setQuantity(e.target.value)}
              type="number"
              sx={{ mb: 2 }}
            />
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="contained"
                color="success"
                onClick={() => handleTrade('LONG')}
                disabled={loading}
              >
                做多
              </Button>
              <Button
                variant="contained"
                color="error"
                onClick={() => handleTrade('SHORT')}
                disabled={loading}
              >
                做空
              </Button>
              <Button
                variant="outlined"
                onClick={() => handleTrade('CLOSE')}
                disabled={loading}
              >
                平仓
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* 当前价格 */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6">ETH/USDT</Typography>
              <Typography variant="h4" color="primary">
                ${currentPrice.toFixed(2)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {message && (
          <Grid item xs={12}>
            <Alert severity="info">{message}</Alert>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default Trading;