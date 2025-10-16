import React from 'react';
import {
  Paper,
  Typography,
  Box,
  Chip,
  Grid,
  Skeleton,
} from '@mui/material';

interface SystemStatus {
  system_state: string;
  auto_trading_enabled: boolean;
  trading_mode: string;
  services: Record<string, boolean>;
}

interface SystemStatusProps {
  status: SystemStatus | null;
  loading: boolean;
}

const SystemStatus: React.FC<SystemStatusProps> = ({ status, loading }) => {
  if (loading) {
    return (
      <Paper sx={{ p: 2, height: 300 }}>
        <Skeleton variant="text" width="40%" height={32} />
        <Box sx={{ mt: 2 }}>
          {[...Array(3)].map((_, index) => (
            <Box key={index} sx={{ mb: 2 }}>
              <Skeleton variant="text" width="30%" height={24} />
              <Skeleton variant="rectangular" width="60%" height={32} sx={{ mt: 1 }} />
            </Box>
          ))}
        </Box>
      </Paper>
    );
  }

  const getStatusColor = (state: string) => {
    switch (state) {
      case 'RUNNING':
        return 'success';
      case 'STOPPED':
        return 'error';
      case 'PAUSED':
        return 'warning';
      default:
        return 'default';
    }
  };

  const getServiceStatus = (isRunning: boolean) => ({
    label: isRunning ? '运行中' : '已停止',
    color: isRunning ? 'success' : 'error',
  });

  return (
    <Paper sx={{ p: 2, height: 300 }}>
      <Typography variant="h6" gutterBottom>
        系统状态
      </Typography>
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          系统状态
        </Typography>
        <Chip
          label={status?.system_state || 'UNKNOWN'}
          color={getStatusColor(status?.system_state || '')}
          variant="outlined"
        />
      </Box>

      <Box sx={{ mb: 3 }}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          交易模式
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Chip
            label={status?.trading_mode || 'UNKNOWN'}
            color="primary"
            variant="outlined"
            size="small"
          />
          <Chip
            label={status?.auto_trading_enabled ? '自动交易' : '信号模式'}
            color={status?.auto_trading_enabled ? 'success' : 'warning'}
            variant="outlined"
            size="small"
          />
        </Box>
      </Box>

      <Box>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          服务状态
        </Typography>
        <Grid container spacing={1}>
          {Object.entries(status?.services || {}).map(([service, isRunning]) => {
            const serviceStatus = getServiceStatus(isRunning);
            return (
              <Grid item key={service}>
                <Chip
                  label={`${service}: ${serviceStatus.label}`}
                  color={serviceStatus.color as any}
                  size="small"
                  variant="outlined"
                />
              </Grid>
            );
          })}
        </Grid>
      </Box>
    </Paper>
  );
};

export default SystemStatus;