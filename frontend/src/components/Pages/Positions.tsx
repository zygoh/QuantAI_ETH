import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Button,
} from '@mui/material';
import { useData } from '../../contexts/DataContext';
import * as api from '../../services/api';

const Positions: React.FC = () => {
  const { positions } = useData();

  const closePosition = async (symbol: string) => {
    try {
      await api.closePosition(symbol);
    } catch (error) {
      console.error('平仓失败:', error);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        持仓管理
      </Typography>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>交易对</TableCell>
              <TableCell>方向</TableCell>
              <TableCell align="right">数量</TableCell>
              <TableCell align="right">入场价格</TableCell>
              <TableCell align="right">标记价格</TableCell>
              <TableCell align="right">未实现盈亏</TableCell>
              <TableCell align="right">收益率</TableCell>
              <TableCell align="center">操作</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {positions.map((position) => (
              <TableRow key={position.symbol}>
                <TableCell>{position.symbol}</TableCell>
                <TableCell>
                  <Chip
                    label={position.side}
                    color={position.side === 'LONG' ? 'success' : 'error'}
                    size="small"
                  />
                </TableCell>
                <TableCell align="right">{position.size.toFixed(4)}</TableCell>
                <TableCell align="right">${position.entry_price.toFixed(2)}</TableCell>
                <TableCell align="right">${position.mark_price.toFixed(2)}</TableCell>
                <TableCell 
                  align="right"
                  sx={{ color: position.unrealized_pnl >= 0 ? 'success.main' : 'error.main' }}
                >
                  ${position.unrealized_pnl.toFixed(2)}
                </TableCell>
                <TableCell 
                  align="right"
                  sx={{ color: position.percentage >= 0 ? 'success.main' : 'error.main' }}
                >
                  {position.percentage.toFixed(2)}%
                </TableCell>
                <TableCell align="center">
                  <Button
                    size="small"
                    variant="outlined"
                    onClick={() => closePosition(position.symbol)}
                  >
                    平仓
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default Positions;