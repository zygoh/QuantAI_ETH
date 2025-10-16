import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box } from '@mui/material';

// 页面组件
import Dashboard from '../Pages/Dashboard';
import Trading from '../Pages/Trading';
import Signals from '../Pages/Signals';
import Positions from '../Pages/Positions';
import Performance from '../Pages/Performance';
import Risk from '../Pages/Risk';
import Training from '../Pages/Training';
import Settings from '../Pages/Settings';

const MainContent: React.FC = () => {
  return (
    <Box sx={{ height: '100%', overflow: 'auto' }}>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/trading" element={<Trading />} />
        <Route path="/signals" element={<Signals />} />
        <Route path="/positions" element={<Positions />} />
        <Route path="/performance" element={<Performance />} />
        <Route path="/risk" element={<Risk />} />
        <Route path="/training" element={<Training />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Box>
  );
};

export default MainContent;