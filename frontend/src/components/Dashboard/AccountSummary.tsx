import React from 'react';
import {
  Paper,
  Typography,
  Box,
  List,
  ListItem,
  ListItemText,
  Skeleton,
  Chip,
} from '@mui/material';

interface AccountInfo {
  total_wallet_balance: number;
  total_unrealized_pnl: number;
  available_balance: number;
  can_trade: boolean;
}

interface AccountSummaryProps {
  accountInfo: AccountInfo | null;
  loading: boolean;
}

const AccountSummary: React.FC<AccountSummaryProps> = ({ accountInfo, loading }) => {
  if (loading) {
    return (
      <Paper sx={{ p: 2, height: 300 }}>
        <Skeleton variant="text" width="40%" height={32} />
        <Box sx={{ mt: 2 }}>
          {[...Array(4)].map((_, index) => (
            <Skeleton key={index} variant="text" height={48} sx={{ mb: 1 }} />
          ))}
        </Box>
      </Paper>
    );
  }

  return (
    <Paper sx={{ p: 2, height: 300 }}>
      <Typography variant="h6" gutterBottom>
        账户摘要
      </Typography>
      
      <List dense>
        <ListItem>
          <ListItemText
            primary="总余额"
            secondary={`$${accountInfo?.total_wallet_balance.toFixed(2) || '0.00'}`}
          />
        </ListItem>
        
        <ListItem>
          <ListItemText
            primary="可用余额"
            secondary={`$${accountInfo?.available_balance.toFixed(2) || '0.00'}`}
          />
        </ListItem>
        
        <ListItem>
          <ListItemText
            primary="未实现盈亏"
            secondary={
              <Typography
                color={
                  (accountInfo?.total_unrealized_pnl || 0) >= 0 
                    ? 'success.main' 
                    : 'error.main'
                }
              >
                ${accountInfo?.total_unrealized_pnl.toFixed(2) || '0.00'}
              </Typography>
            }
          />
        </ListItem>
        
        <ListItem>
          <ListItemText
            primary="交易状态"
            secondary={
              <Chip
                label={accountInfo?.can_trade ? '可交易' : '禁止交易'}
                color={accountInfo?.can_trade ? 'success' : 'error'}
                size="small"
              />
            }
          />
        </ListItem>
      </List>
    </Paper>
  );
};

export default AccountSummary;