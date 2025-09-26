import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.actual_data = None
        self.predicted_data = None
        self.satellites = []
        self.prediction_columns = []
        
    def load_data(self) -> bool:
        """Load both actual and predicted data files"""
        try:
            # Load actual data
            logger.info("Loading actual data...")
            self.actual_data = pd.read_csv('clock_bias_correction_combined_1_7_jan_2024.csv')
            self.actual_data['datetime'] = pd.to_datetime(self.actual_data['datetime'])
            
            # Load predicted data  
            logger.info("Loading predicted data...")
            self.predicted_data = pd.read_csv('suyash_gandu.csv')
            self.predicted_data['datetime'] = pd.to_datetime(self.predicted_data['datetime'])
            
            # Extract satellites and prediction columns
            self.satellites = sorted(self.actual_data['PRN'].unique().tolist())
            self.prediction_columns = [col for col in self.predicted_data.columns if col.startswith('pred_')]
            
            logger.info(f"Loaded actual data: {len(self.actual_data)} rows")
            logger.info(f"Loaded predicted data: {len(self.predicted_data)} rows")
            logger.info(f"Available satellites: {len(self.satellites)}")
            logger.info(f"Prediction horizons: {len(self.prediction_columns)} (30sec to {len(self.prediction_columns)*30}sec)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def get_satellite_actual_data(self, prn: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get actual data for specific satellite"""
        if self.actual_data is None:
            raise ValueError("Data not loaded")
            
        data = self.actual_data[self.actual_data['PRN'] == prn].copy()
        
        if start_date:
            data = data[data['datetime'] >= start_date]
        if end_date:
            data = data[data['datetime'] <= end_date]
            
        return data
    
    def get_satellite_predicted_data(self, prn: str) -> pd.DataFrame:
        """Get predicted data for specific satellite"""
        if self.predicted_data is None:
            raise ValueError("Data not loaded")
            
        return self.predicted_data[self.predicted_data['PRN'] == prn].copy()
    
    def get_prediction_horizon_seconds(self, pred_column: str) -> int:
        """Convert prediction column to seconds (pred_1 = 30sec, pred_2 = 60sec, etc.)"""
        if not pred_column.startswith('pred_'):
            raise ValueError("Invalid prediction column")
        
        pred_num = int(pred_column.split('_')[1])
        return pred_num * 30
    
    def get_prediction_column_by_seconds(self, seconds: int) -> str:
        """Get prediction column name by seconds"""
        if seconds % 30 != 0 or seconds <= 0:
            raise ValueError("Seconds must be positive multiple of 30")
        
        pred_num = seconds // 30
        return f"pred_{pred_num}"
    
    def get_available_satellites(self) -> List[str]:
        """Get list of available satellites"""
        return self.satellites.copy()
    
    def get_data_summary(self) -> Dict:
        """Get summary of loaded data"""
        if self.actual_data is None or self.predicted_data is None:
            return {"status": "Data not loaded"}
        
        return {
            "actual_data_rows": len(self.actual_data),
            "predicted_data_rows": len(self.predicted_data),
            "satellites": self.satellites,
            "date_range_actual": {
                "start": self.actual_data['datetime'].min().strftime('%Y-%m-%d %H:%M:%S'),
                "end": self.actual_data['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')
            },
            "prediction_horizons": {
                "count": len(self.prediction_columns),
                "max_seconds": len(self.prediction_columns) * 30,
                "max_hours": (len(self.prediction_columns) * 30) / 3600
            }
        }

# Global instance
data_loader = DataLoader()