import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  Chip,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AccountBalance,
  ShowChart,
} from '@mui/icons-material';
import { useData } from '../../contexts/DataContext';
import { useWebSocket } from '../../contexts/WebSocketContext';
import AccountSummary from '../Dashboard/AccountSummary';
import PositionsSummary from '../Dashboard/PositionsSummary';
import RecentSignals from '../Dashboard/RecentSignals';
import SystemStatus from '../Dashboard/SystemStatus';
import PriceChart from '../Dashboard/PriceChart';

const Dashboard: React.FC = () => {
  const { 
    accountInfo, 
    positions, 
    signals, 
    systemStatus, 
    currentPrice, 
    loading 
  } = useData();
  const { isConnected } = useWebSocket();

  // 计算总盈亏
  const totalPnL = positions.reduce((sum, pos) => sum + pos.unrealized_pnl, 0);
  const totalPositionValue = positions.reduce((sum, pos) => 
    sum + (pos.size * pos.mark_price), 0
  );

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom>
        交易仪表板
      </Typography>

      {/* 连接状态指示器 */}
      <Box sx={{ mb: 2 }}>
        <Chip
          label={isConnected ? '实时连接' : '连接断开'}
          color={isConnected ? 'success' : 'error'}
          variant="outlined"
          size="small"
        />
      </Box>

      <Grid container spacing={3}>
        {/* 关键指标卡片 */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <ShowChart color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">当前价格</Typography>
              </Box>
              <Typography variant="h4" color="primary">
                ${currentPrice.toFixed(2)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                ETH/USDT
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <AccountBalance color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">账户余额</Typography>
              </Box>
              <Typography variant="h4">
                ${accountInfo?.total_wallet_balance.toFixed(2) || '0.00'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                可用: ${accountInfo?.available_balance.toFixed(2) || '0.00'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                {totalPnL >= 0 ? (
                  <TrendingUp color="success" sx={{ mr: 1 }} />
                ) : (
                  <TrendingDown color="error" sx={{ mr: 1 }} />
                )}
                <Typography variant="h6">未实现盈亏</Typography>
              </Box>
              <Typography 
                variant="h4" 
                color={totalPnL >= 0 ? 'success.main' : 'error.main'}
              >
                ${totalPnL.toFixed(2)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                持仓价值: ${totalPositionValue.toFixed(2)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Typography variant="h6">活跃持仓</Typography>
              </Box>
              <Typography variant="h4">
                {positions.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                个合约
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* 系统状态 */}
        <Grid item xs={12} md={6}>
          <SystemStatus status={systemStatus} loading={loading} />
        </Grid>

        {/* 账户摘要 */}
        <Grid item xs={12} md={6}>
          <AccountSummary accountInfo={accountInfo} loading={loading} />
        </Grid>

        {/* 价格图表 */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              ETH/USDT 价格走势
            </Typography>
            <PriceChart />
          </Paper>
        </Grid>

        {/* 最近信号 */}
        <Grid item xs={12} lg={4}>
          <RecentSignals signals={signals} loading={loading} />
        </Grid>

        {/* 持仓摘要 */}
        <Grid item xs={12}>
          <PositionsSummary positions={positions} loading={loading} />
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;