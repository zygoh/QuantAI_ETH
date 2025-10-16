import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import { Box } from '@mui/material';
import Layout from './components/Layout/Layout';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { DataProvider } from './contexts/DataContext';

function App() {
  return (
    <Router>
      <WebSocketProvider>
        <DataProvider>
          <Box sx={{ display: 'flex', height: '100vh' }}>
            <Layout />
          </Box>
        </DataProvider>
      </WebSocketProvider>
    </Router>
  );
}

export default App;