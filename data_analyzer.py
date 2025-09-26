import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from data_loader import data_loader
import logging

logger = logging.getLogger(__name__)

class DataAnalyzer:
    
    def calculate_error_metrics(self, actual: np.array, predicted: np.array) -> Dict[str, float]:
        """Calculate comprehensive error metrics"""
        errors = predicted - actual
        
        return {
            'mae': np.mean(np.abs(errors)),
            'rmse': np.sqrt(np.mean(errors**2)),
            'mbe': np.mean(errors),  # Mean Bias Error
            'std_error': np.std(errors),
            'max_error': np.max(np.abs(errors)),
            'min_error': np.min(np.abs(errors)),
            'error_range': np.max(errors) - np.min(errors)
        }
    
    def analyze_satellite_performance(self, prn: str, prediction_horizon: int = 30) -> Dict:
        """Analyze performance of a single satellite at specific prediction horizon"""
        try:
            # Check if satellite exists
            available_satellites = data_loader.get_available_satellites()
            if prn not in available_satellites:
                return {
                    "error": f"Satellite {prn} not found in dataset",
                    "available_satellites": available_satellites[:10],  # Show first 10
                    "total_available": len(available_satellites)
                }
            
            # Get actual and predicted data
            actual_data = data_loader.get_satellite_actual_data(prn)
            predicted_data = data_loader.get_satellite_predicted_data(prn)
            
            if actual_data.empty or predicted_data.empty:
                return {
                    "error": f"No data available for satellite {prn}",
                    "available_satellites": available_satellites[:10],
                    "total_available": len(available_satellites)
                }
            
            # Get prediction column
            pred_column = data_loader.get_prediction_column_by_seconds(prediction_horizon)
            if pred_column not in predicted_data.columns:
                return {"error": f"Prediction horizon {prediction_horizon}s not available"}
            
            # For clock bias correction analysis (main target variable)
            actual_values = actual_data['clock_bias_correction'].values
            predicted_values = predicted_data[pred_column].values
            
            # Align data (take minimum length)
            min_len = min(len(actual_values), len(predicted_values))
            actual_values = actual_values[:min_len]
            predicted_values = predicted_values[:min_len]
            
            # Calculate metrics
            metrics = self.calculate_error_metrics(actual_values, predicted_values)
            
            # Additional analysis
            error_percentiles = np.percentile(np.abs(predicted_values - actual_values), [25, 50, 75, 90, 95])
            
            return {
                'satellite': prn,
                'prediction_horizon_seconds': prediction_horizon,
                'data_points': min_len,
                'metrics': metrics,
                'error_percentiles': {
                    'p25': error_percentiles[0],
                    'p50': error_percentiles[1], 
                    'p75': error_percentiles[2],
                    'p90': error_percentiles[3],
                    'p95': error_percentiles[4]
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing satellite {prn}: {e}")
            return {"error": str(e)}
    
    def compare_satellites(self, satellite_list: List[str], prediction_horizon: int = 30) -> Dict:
        """Compare performance of multiple satellites"""
        results = {}
        
        for prn in satellite_list:
            results[prn] = self.analyze_satellite_performance(prn, prediction_horizon)
        
        # Rank satellites by RMSE (lower is better)
        valid_results = {k: v for k, v in results.items() if 'metrics' in v}
        if valid_results:
            ranking = sorted(valid_results.items(), key=lambda x: x[1]['metrics']['rmse'])
            
            return {
                'comparison': results,
                'ranking': [{'satellite': sat, 'rmse': data['metrics']['rmse']} for sat, data in ranking],
                'best_satellite': ranking[0][0] if ranking else None,
                'worst_satellite': ranking[-1][0] if ranking else None
            }
        
        return {'comparison': results, 'error': 'No valid data for comparison'}
    
    def analyze_prediction_horizons(self, prn: str, horizons: List[int] = None) -> Dict:
        """Analyze how performance degrades across prediction horizons"""
        if horizons is None:
            horizons = [30, 60, 120, 300, 600, 1800, 3600]  # 30s, 1m, 2m, 5m, 10m, 30m, 1h
        
        results = {}
        
        for horizon in horizons:
            try:
                result = self.analyze_satellite_performance(prn, horizon)
                if 'metrics' in result:
                    results[f"{horizon}s"] = {
                        'rmse': result['metrics']['rmse'],
                        'mae': result['metrics']['mae'],
                        'mbe': result['metrics']['mbe']
                    }
            except:
                continue
        
        return {
            'satellite': prn,
            'horizon_analysis': results,
            'trend': self._analyze_degradation_trend(results) if results else "No data"
        }
    
    def _analyze_degradation_trend(self, horizon_results: Dict) -> str:
        """Analyze if prediction performance degrades over time"""
        rmse_values = [v['rmse'] for v in horizon_results.values()]
        
        if len(rmse_values) < 2:
            return "Insufficient data"
        
        # Check if RMSE generally increases
        increasing = sum(1 for i in range(1, len(rmse_values)) if rmse_values[i] > rmse_values[i-1])
        total_comparisons = len(rmse_values) - 1
        
        if increasing / total_comparisons > 0.7:
            return "Performance degrades with longer prediction horizons"
        elif increasing / total_comparisons < 0.3:
            return "Performance improves with longer prediction horizons (unusual)"
        else:
            return "Mixed performance across prediction horizons"
    
    def get_satellite_statistics(self, prn: str) -> Dict:
        """Get comprehensive statistics for a satellite"""
        try:
            actual_data = data_loader.get_satellite_actual_data(prn)
            
            if actual_data.empty:
                return {"error": f"No actual data for satellite {prn}"}
            
            clock_bias = actual_data['clock_bias_correction']
            
            return {
                'satellite': prn,
                'data_points': len(actual_data),
                'time_range': {
                    'start': actual_data['datetime'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': actual_data['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')
                },
                'clock_bias_stats': {
                    'mean': float(clock_bias.mean()),
                    'std': float(clock_bias.std()),
                    'min': float(clock_bias.min()),
                    'max': float(clock_bias.max()),
                    'median': float(clock_bias.median())
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def find_satellites_with_error_threshold(self, max_error: float, prediction_horizon: int = 30) -> Dict:
        """Find satellites with error below threshold"""
        satellites = data_loader.get_available_satellites()
        results = {}
        qualifying_satellites = []
        
        for prn in satellites:
            analysis = self.analyze_satellite_performance(prn, prediction_horizon)
            if 'metrics' in analysis:
                rmse = analysis['metrics']['rmse']
                results[prn] = rmse
                if rmse <= max_error:
                    qualifying_satellites.append({'satellite': prn, 'rmse': rmse})
        
        # Sort by performance
        qualifying_satellites.sort(key=lambda x: x['rmse'])
        
        return {
            'threshold': max_error,
            'prediction_horizon': prediction_horizon,
            'qualifying_satellites': qualifying_satellites,
            'total_satellites': len(satellites),
            'qualifying_count': len(qualifying_satellites),
            'all_results': results
        }

# Global instance
data_analyzer = DataAnalyzer()