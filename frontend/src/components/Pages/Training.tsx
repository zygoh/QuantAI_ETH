import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Alert,
  LinearProgress,
} from '@mui/material';
import * as api from '../../services/api';

const Training: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  const startTraining = async () => {
    setLoading(true);
    try {
      const response = await api.startTraining(true);
      setMessage(response.message);
    } catch (error) {
      setMessage('训练失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        模型训练
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          训练控制
        </Typography>
        <Button
          variant="contained"
          onClick={startTraining}
          disabled={loading}
          sx={{ mb: 2 }}
        >
          开始训练
        </Button>
        {loading && <LinearProgress />}
      </Paper>

      {message && (
        <Alert severity="info" sx={{ mb: 3 }}>
          {message}
        </Alert>
      )}
    </Box>
  );
};

export default Training;