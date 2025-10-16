import React from 'react';
import { Box, Typography, Grid, Paper, Card, CardContent } from '@mui/material';

const Performance: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        绩效分析
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6">总收益率</Typography>
              <Typography variant="h4" color="success.main">+12.5%</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6">夏普比率</Typography>
              <Typography variant="h4">1.85</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Performance;