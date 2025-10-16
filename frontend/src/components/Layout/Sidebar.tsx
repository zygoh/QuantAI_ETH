import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Divider,
  Typography,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  TrendingUp as TradingIcon,
  Assessment as SignalsIcon,
  AccountBalance as PositionsIcon,
  Analytics as PerformanceIcon,
  Settings as SettingsIcon,
  School as TrainingIcon,
  Security as RiskIcon,
} from '@mui/icons-material';

interface MenuItem {
  text: string;
  icon: React.ReactElement;
  path: string;
}

const menuItems: MenuItem[] = [
  { text: '仪表板', icon: <DashboardIcon />, path: '/' },
  { text: '交易控制', icon: <TradingIcon />, path: '/trading' },
  { text: '交易信号', icon: <SignalsIcon />, path: '/signals' },
  { text: '持仓管理', icon: <PositionsIcon />, path: '/positions' },
  { text: '绩效分析', icon: <PerformanceIcon />, path: '/performance' },
  { text: '风险管理', icon: <RiskIcon />, path: '/risk' },
  { text: '模型训练', icon: <TrainingIcon />, path: '/training' },
  { text: '系统设置', icon: <SettingsIcon />, path: '/settings' },
];

const Sidebar: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const handleNavigation = (path: string) => {
    navigate(path);
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Toolbar>
        <Typography variant="h6" noWrap component="div" sx={{ color: 'primary.main' }}>
          ETH交易系统
        </Typography>
      </Toolbar>
      
      <Divider />
      
      <List sx={{ flexGrow: 1 }}>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => handleNavigation(item.path)}
              sx={{
                '&.Mui-selected': {
                  backgroundColor: 'primary.main',
                  color: 'primary.contrastText',
                  '&:hover': {
                    backgroundColor: 'primary.dark',
                  },
                  '& .MuiListItemIcon-root': {
                    color: 'primary.contrastText',
                  },
                },
              }}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      
      <Divider />
      
      <Box sx={{ p: 2 }}>
        <Typography variant="caption" color="text.secondary">
          版本 1.0.0
        </Typography>
      </Box>
    </Box>
  );
};

export default Sidebar;