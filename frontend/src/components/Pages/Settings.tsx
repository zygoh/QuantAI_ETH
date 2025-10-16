import React from 'react';
import {
  Box,
  Typography,
  Paper,
  List,
  ListItem,
  ListItemText,
  Switch,
  Divider,
} from '@mui/material';

const Settings: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        系统设置
      </Typography>

      <Paper>
        <List>
          <ListItem>
            <ListItemText
              primary="自动交易"
              secondary="启用自动交易功能"
            />
            <Switch />
          </ListItem>
          <Divider />
          <ListItem>
            <ListItemText
              primary="风险警报"
              secondary="启用风险警报通知"
            />
            <Switch defaultChecked />
          </ListItem>
          <Divider />
          <ListItem>
            <ListItemText
              primary="实时数据"
              secondary="启用实时数据推送"
            />
            <Switch defaultChecked />
          </ListItem>
        </List>
      </Paper>
    </Box>
  );
};

export default Settings;