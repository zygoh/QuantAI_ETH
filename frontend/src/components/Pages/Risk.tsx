import React from 'react';
import { Box, Typography, Grid, Paper, Card, CardContent } from '@mui/material';

const Risk: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        风险管理
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6">VaR (95%)</Typography>
              <Typography variant="h4" color="warning.main">2.3%</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6">最大回撤</Typography>
              <Typography variant="h4" color="error.main">5.8%</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Risk;