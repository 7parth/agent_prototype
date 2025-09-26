import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from pathlib import Path

from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent , ZeroShotAgent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


class DataManager:
    """Manages GNSS satellite data loading and preprocessing"""
    
    def __init__(self, actual_data_path: str, predicted_data_path: str):
        self.actual_data_path = actual_data_path
        self.predicted_data_path = predicted_data_path
        self.actual_data = None
        self.predicted_data = None
        self.satellites = set()
        
    def load_data(self):
        """Load and preprocess both actual and predicted data"""
        try:
            # Load actual data
            self.actual_data = pd.read_csv(self.actual_data_path)
            self.actual_data['datetime'] = pd.to_datetime(self.actual_data['datetime'])
            
            # Load predicted data
            self.predicted_data = pd.read_csv(self.predicted_data_path)
            self.predicted_data['datetime'] = pd.to_datetime(self.predicted_data['datetime'])
            
            # Get unique satellites
            self.satellites = set(self.actual_data['PRN'].unique()) & set(self.predicted_data['PRN'].unique())
            
            print(f"Loaded data for {len(self.satellites)} satellites: {sorted(self.satellites)}")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_satellite_data(self, satellite_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get actual and predicted data for a specific satellite"""
        actual = self.actual_data[self.actual_data['PRN'] == satellite_id].copy()
        predicted = self.predicted_data[self.predicted_data['PRN'] == satellite_id].copy()
        return actual, predicted
    
    def get_available_satellites(self) -> List[str]:
        """Get list of available satellites"""
        return sorted(list(self.satellites))


class MetricsCalculator:
    """Calculate various error metrics for GNSS predictions"""
    
    @staticmethod
    def calculate_mae(actual: np.array, predicted: np.array) -> float:
        """Mean Absolute Error"""
        return np.mean(np.abs(actual - predicted))
    
    @staticmethod
    def calculate_rmse(actual: np.array, predicted: np.array) -> float:
        """Root Mean Square Error"""
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    @staticmethod
    def calculate_r_squared(actual: np.array, predicted: np.array) -> float:
        """R-squared coefficient of determination"""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    @staticmethod
    def calculate_mape(actual: np.array, predicted: np.array) -> float:
        """Mean Absolute Percentage Error"""
        mask = actual != 0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]) * 100)
    
    @staticmethod
    def calculate_bias(actual: np.array, predicted: np.array) -> float:
        """Bias (mean error)"""
        return np.mean(predicted - actual)


class MetricsTool(BaseTool):
    """LangChain tool for calculating error metrics"""
    
    name: str = "metrics_calculator"
    description: str = """Calculate error metrics (MAE, RMSE, R², MAPE, Bias) for GNSS satellite predictions.
    Input should be a JSON string with:
    - satellite_id: satellite identifier (e.g., 'G01', 'G07')
    - prediction_horizon: prediction step (1-240, where 1=30sec, 2=60sec, etc.)
    - metric_type: 'all' or specific metric name
    """
    
    def __init__(self, data_manager: DataManager):
        super().__init__()
        # Store data_manager as a regular attribute
        object.__setattr__(self, 'data_manager', data_manager)
        object.__setattr__(self, 'calculator', MetricsCalculator())
    
    def _run(self, query: str) -> str:
        try:
            params = json.loads(query)
            satellite_id = params.get('satellite_id')
            prediction_horizon = params.get('prediction_horizon', 1)
            metric_type = params.get('metric_type', 'all')
            
            # Get data for the satellite
            actual_data, predicted_data = self.data_manager.get_satellite_data(satellite_id)
            
            if actual_data.empty or predicted_data.empty:
                return f"No data available for satellite {satellite_id}"
            
            # Align data by datetime
            merged = pd.merge(actual_data, predicted_data, on=['datetime', 'PRN'], how='inner')
            
            if merged.empty:
                return f"No matching timestamps found for satellite {satellite_id}"
            
            # Get actual clock bias correction values
            actual_values = merged['clock_bias_correction'].values
            
            # Get predicted values for the specified horizon
            pred_col = f'pred_{prediction_horizon}'
            if pred_col not in merged.columns:
                return f"Prediction horizon {prediction_horizon} not available (max: 240)"
            
            predicted_values = merged[pred_col].values
            
            # Calculate metrics
            results = {}
            
            if metric_type == 'all' or metric_type == 'mae':
                results['MAE'] = self.calculator.calculate_mae(actual_values, predicted_values)
            
            if metric_type == 'all' or metric_type == 'rmse':
                results['RMSE'] = self.calculator.calculate_rmse(actual_values, predicted_values)
            
            if metric_type == 'all' or metric_type == 'r2':
                results['R²'] = self.calculator.calculate_r_squared(actual_values, predicted_values)
            
            if metric_type == 'all' or metric_type == 'mape':
                results['MAPE'] = self.calculator.calculate_mape(actual_values, predicted_values)
            
            if metric_type == 'all' or metric_type == 'bias':
                results['Bias'] = self.calculator.calculate_bias(actual_values, predicted_values)
            
            # Add metadata
            results['satellite'] = satellite_id
            results['prediction_horizon'] = f"{prediction_horizon * 30} seconds"
            results['data_points'] = len(actual_values)
            
            return json.dumps(results, indent=2)
            
        except Exception as e:
            return f"Error calculating metrics: {str(e)}"


class PlotTool(BaseTool):
    """LangChain tool for generating visualizations"""
    
    name: str = "plot_generator"
    description: str = """Generate visualizations for GNSS satellite data analysis.
    Input should be a JSON string with:
    - plot_type: 'line', 'residual', 'histogram', 'heatmap', 'comparison'
    - satellite_id: satellite identifier
    - prediction_horizon: prediction step (optional, defaults to 1)
    - additional parameters based on plot type
    """
    
    def __init__(self, data_manager: DataManager):
        super().__init__()
        # Store data_manager as a regular attribute
        object.__setattr__(self, 'data_manager', data_manager)
    
    def _run(self, query: str) -> str:
        try:
            params = json.loads(query)
            plot_type = params.get('plot_type')
            satellite_id = params.get('satellite_id')
            
            if plot_type == 'line':
                return self._generate_line_plot(params)
            elif plot_type == 'residual':
                return self._generate_residual_plot(params)
            elif plot_type == 'histogram':
                return self._generate_histogram(params)
            elif plot_type == 'heatmap':
                return self._generate_heatmap(params)
            elif plot_type == 'comparison':
                return self._generate_comparison_plot(params)
            else:
                return f"Unknown plot type: {plot_type}"
                
        except Exception as e:
            return f"Error generating plot: {str(e)}"
    
    def _generate_line_plot(self, params: Dict) -> str:
        """Generate line plot comparing actual vs predicted values"""
        satellite_id = params['satellite_id']
        prediction_horizon = params.get('prediction_horizon', 1)
        
        actual_data, predicted_data = self.data_manager.get_satellite_data(satellite_id)
        merged = pd.merge(actual_data, predicted_data, on=['datetime', 'PRN'], how='inner')
        
        if merged.empty:
            return "No data to plot"
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=merged['datetime'],
            y=merged['clock_bias_correction'],
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        # Add predicted values
        pred_col = f'pred_{prediction_horizon}'
        if pred_col in merged.columns:
            fig.add_trace(go.Scatter(
                x=merged['datetime'],
                y=merged[pred_col],
                mode='lines',
                name=f'Predicted ({prediction_horizon * 30}s)',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=f'Clock Bias Prediction vs Actual - Satellite {satellite_id}',
            xaxis_title='Time',
            yaxis_title='Clock Bias Correction',
            hovermode='x unified',
            height=500
        )
        
        # Convert to HTML
        html_str = fig.to_html(include_plotlyjs='cdn')
        
        return f"Line plot generated for satellite {satellite_id}. Plot shows actual vs predicted clock bias correction over time."
    
    def _generate_residual_plot(self, params: Dict) -> str:
        """Generate residual plot"""
        satellite_id = params['satellite_id']
        prediction_horizon = params.get('prediction_horizon', 1)
        
        actual_data, predicted_data = self.data_manager.get_satellite_data(satellite_id)
        merged = pd.merge(actual_data, predicted_data, on=['datetime', 'PRN'], how='inner')
        
        pred_col = f'pred_{prediction_horizon}'
        residuals = merged['clock_bias_correction'] - merged[pred_col]
        
        fig = go.Figure()
        
        # Residual scatter plot
        fig.add_trace(go.Scatter(
            x=merged[pred_col],
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='red', size=6, opacity=0.6)
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        
        fig.update_layout(
            title=f'Residual Plot - Satellite {satellite_id}',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals (Actual - Predicted)',
            height=500
        )
        
        return f"Residual plot generated for satellite {satellite_id}."
    
    def _generate_histogram(self, params: Dict) -> str:
        """Generate histogram of prediction errors"""
        satellite_id = params['satellite_id']
        prediction_horizon = params.get('prediction_horizon', 1)
        
        actual_data, predicted_data = self.data_manager.get_satellite_data(satellite_id)
        merged = pd.merge(actual_data, predicted_data, on=['datetime', 'PRN'], how='inner')
        
        pred_col = f'pred_{prediction_horizon}'
        errors = merged['clock_bias_correction'] - merged[pred_col]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=errors,
            nbinsx=30,
            name='Error Distribution',
            marker_color='skyblue',
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f'Error Distribution - Satellite {satellite_id}',
            xaxis_title='Prediction Error',
            yaxis_title='Frequency',
            height=500
        )
        
        return f"Error distribution histogram generated for satellite {satellite_id}."
    
    def _generate_heatmap(self, params: Dict) -> str:
        """Generate heatmap of prediction errors across different horizons"""
        satellite_id = params['satellite_id']
        
        actual_data, predicted_data = self.data_manager.get_satellite_data(satellite_id)
        merged = pd.merge(actual_data, predicted_data, on=['datetime', 'PRN'], how='inner')
        
        # Calculate RMSE for different prediction horizons
        horizons = range(1, min(25, 241))  # First 24 horizons (up to 12 minutes)
        rmse_values = []
        
        for horizon in horizons:
            pred_col = f'pred_{horizon}'
            if pred_col in merged.columns:
                errors = merged['clock_bias_correction'] - merged[pred_col]
                rmse = np.sqrt(np.mean(errors**2))
                rmse_values.append(rmse)
            else:
                rmse_values.append(np.nan)
        
        # Create heatmap data
        heatmap_data = np.array(rmse_values).reshape(1, -1)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[f"{h*30}s" for h in horizons],
            y=[satellite_id],
            colorscale='Viridis',
            colorbar=dict(title="RMSE")
        ))
        
        fig.update_layout(
            title=f'RMSE Heatmap Across Prediction Horizons - Satellite {satellite_id}',
            xaxis_title='Prediction Horizon',
            height=400
        )
        
        return f"RMSE heatmap generated for satellite {satellite_id} across different prediction horizons."
    
    def _generate_comparison_plot(self, params: Dict) -> str:
        """Generate comparison plot for multiple satellites"""
        satellite_ids = params.get('satellite_ids', [])
        prediction_horizon = params.get('prediction_horizon', 1)
        
        if not satellite_ids:
            return "No satellites specified for comparison"
        
        fig = go.Figure()
        
        for sat_id in satellite_ids:
            actual_data, predicted_data = self.data_manager.get_satellite_data(sat_id)
            merged = pd.merge(actual_data, predicted_data, on=['datetime', 'PRN'], how='inner')
            
            if not merged.empty:
                pred_col = f'pred_{prediction_horizon}'
                if pred_col in merged.columns:
                    errors = np.abs(merged['clock_bias_correction'] - merged[pred_col])
                    fig.add_trace(go.Box(
                        y=errors,
                        name=sat_id,
                        boxpoints='outliers'
                    ))
        
        fig.update_layout(
            title=f'Prediction Error Comparison Across Satellites',
            yaxis_title='Absolute Error',
            xaxis_title='Satellite',
            height=500
        )
        
        return f"Comparison plot generated for satellites: {', '.join(satellite_ids)}"


class GNSSAIAgent:
    """Main AI Agent for GNSS satellite analysis"""
    
    def __init__(self, actual_data_path: str, predicted_data_path: str, google_api_key: str):
        self.data_manager = DataManager(actual_data_path, predicted_data_path)
        self.google_api_key = google_api_key
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.1
        )
        
        # Initialize tools
        self.metrics_tool = MetricsTool(self.data_manager)
        self.plot_tool = PlotTool(self.data_manager)
        
        self.tools = [self.metrics_tool, self.plot_tool]
        
        # Create agent
        self.agent_executor = None
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup the LangChain agent"""
        prompt_template = """You are a GNSS satellite analysis expert. You help users analyze satellite clock and ephemeris errors by calculating metrics and generating visualizations.

Available tools:{tool_names},{tools}
- metrics_calculator: Calculate error metrics (MAE, RMSE, R², MAPE, Bias) for satellite predictions
- plot_generator: Generate various plots for data analysis

Available satellites: {satellites}

When users ask about satellite performance:
1. Use metrics_calculator to compute relevant error metrics
2. Use plot_generator to create appropriate visualizations
3. Provide clear, quantitative analysis with insights
4. Explain what the metrics mean in practical terms

Current conversation:
{chat_history}

User: {input}

Think step by step:
{agent_scratchpad}"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "chat_history", "agent_scratchpad", "satellites", "tools", "tool_names"],
        )

        
        # Create ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def initialize(self) -> bool:
        """Initialize the agent by loading data"""
        success = self.data_manager.load_data()
        if success:
            # Update agent with available satellites
            satellites = self.data_manager.get_available_satellites()
            print(f"Agent initialized with satellites: {satellites}")
        return success
    
    def query(self, user_input: str, chat_history: str = "") -> str:
        """Process user query and return response"""
        try:
            satellites = self.data_manager.get_available_satellites()
            response = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history,
                "satellites": satellites
            })
            return response["output"]
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def get_satellite_summary(self, satellite_id: str) -> Dict[str, Any]:
        """Get summary statistics for a satellite"""
        actual_data, predicted_data = self.data_manager.get_satellite_data(satellite_id)
        
        if actual_data.empty:
            return {"error": f"No data available for satellite {satellite_id}"}
        
        summary = {
            "satellite_id": satellite_id,
            "data_points": len(actual_data),
            "time_range": {
                "start": actual_data['datetime'].min().isoformat(),
                "end": actual_data['datetime'].max().isoformat()
            },
            "clock_bias_stats": {
                "mean": float(actual_data['clock_bias_correction'].mean()),
                "std": float(actual_data['clock_bias_correction'].std()),
                "min": float(actual_data['clock_bias_correction'].min()),
                "max": float(actual_data['clock_bias_correction'].max())
            }
        }
        
        return summary