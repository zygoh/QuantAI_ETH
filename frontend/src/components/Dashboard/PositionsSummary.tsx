import React from 'react';
import {
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Skeleton,
  Box,
} from '@mui/material';

interface PositionInfo {
  symbol: string;
  side: string;
  size: number;
  entry_price: number;
  mark_price: number;
  unrealized_pnl: number;
  percentage: number;
}

interface PositionsSummaryProps {
  positions: PositionInfo[];
  loading: boolean;
}

const PositionsSummary: React.FC<PositionsSummaryProps> = ({ positions, loading }) => {
  if (loading) {
    return (
      <Paper sx={{ p: 2 }}>
        <Skeleton variant="text" width="30%" height={32} />
        <Box sx={{ mt: 2 }}>
          {[...Array(3)].map((_, index) => (
            <Skeleton key={index} variant="rectangular" height={60} sx={{ mb: 1 }} />
          ))}
        </Box>
      </Paper>
    );
  }

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        持仓概览
      </Typography>
      
      {positions.length === 0 ? (
        <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
          暂无持仓
        </Typography>
      ) : (
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>交易对</TableCell>
                <TableCell>方向</TableCell>
                <TableCell align="right">数量</TableCell>
                <TableCell align="right">入场价格</TableCell>
                <TableCell align="right">标记价格</TableCell>
                <TableCell align="right">未实现盈亏</TableCell>
                <TableCell align="right">收益率</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {positions.map((position) => (
                <TableRow key={position.symbol}>
                  <TableCell component="th" scope="row">
                    {position.symbol}
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={position.side}
                      color={position.side === 'LONG' ? 'success' : 'error'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell align="right">
                    {position.size.toFixed(4)}
                  </TableCell>
                  <TableCell align="right">
                    ${position.entry_price.toFixed(2)}
                  </TableCell>
                  <TableCell align="right">
                    ${position.mark_price.toFixed(2)}
                  </TableCell>
                  <TableCell 
                    align="right"
                    sx={{ 
                      color: position.unrealized_pnl >= 0 ? 'success.main' : 'error.main' 
                    }}
                  >
                    ${position.unrealized_pnl.toFixed(2)}
                  </TableCell>
                  <TableCell 
                    align="right"
                    sx={{ 
                      color: position.percentage >= 0 ? 'success.main' : 'error.main' 
                    }}
                  >
                    {position.percentage.toFixed(2)}%
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Paper>
  );
};

export default PositionsSummary;