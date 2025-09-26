from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime
import asyncio
from config import settings
from contextlib import asynccontextmanager

# Import our GNSS AI Agent
from ai_agent import GNSSAIAgent

# Global agent instance
agent: Optional[GNSSAIAgent] = None

# Pydantic models for API
class AgentQuery(BaseModel):
    query: str
    chat_history: Optional[str] = ""

class AgentResponse(BaseModel):
    response: str
    timestamp: datetime
    satellite_data: Optional[Dict[str, Any]] = None

class MetricsRequest(BaseModel):
    satellite_id: str
    prediction_horizon: Optional[int] = 1
    metric_type: Optional[str] = "all"

class PlotRequest(BaseModel):
    plot_type: str
    satellite_id: str
    prediction_horizon: Optional[int] = 1
    satellite_ids: Optional[List[str]] = None

class SatelliteInfo(BaseModel):
    satellite_id: str
    status: str
    data_points: int
    last_update: datetime

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global agent
    print("Initializing GNSS AI Agent...")
    
    agent = GNSSAIAgent(
        actual_data_path=settings.actual_data_file,
        predicted_data_path=settings.predicted_data_file,
        google_api_key=settings.google_api_key
    )
    
    if agent.initialize():
        print("✅ GNSS AI Agent initialized successfully")
    else:
        print("❌ Failed to initialize GNSS AI Agent")
        raise RuntimeError("Could not initialize AI agent")
    
    yield
    
    # Shutdown
    print("Shutting down GNSS AI Agent...")

# Initialize FastAPI app
app = FastAPI(
    title="GNSS AI Agent API",
    description="AI-powered analysis system for GNSS satellite clock and ephemeris error forecasting",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global agent
    return {
        "status": "healthy" if agent else "unhealthy",
        "timestamp": datetime.now(),
        "agent_initialized": agent is not None
    }

# Get available satellites
@app.get("/api/satellites", response_model=List[str])
async def get_satellites():
    """Get list of available satellites"""
    global agent
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    satellites = agent.data_manager.get_available_satellites()
    return satellites

# Get satellite information
@app.get("/api/satellites/{satellite_id}/info")
async def get_satellite_info(satellite_id: str):
    """Get detailed information about a specific satellite"""
    global agent
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        summary = agent.get_satellite_summary(satellite_id)
        if "error" in summary:
            raise HTTPException(status_code=404, detail=summary["error"])
        
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Calculate metrics endpoint
@app.post("/api/metrics")
async def calculate_metrics(request: MetricsRequest):
    """Calculate error metrics for satellite predictions"""
    global agent
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        query_params = {
            "satellite_id": request.satellite_id,
            "prediction_horizon": request.prediction_horizon,
            "metric_type": request.metric_type
        }
        
        result = agent.metrics_tool._run(json.dumps(query_params))
        
        # Try to parse as JSON, otherwise return as text
        try:
            metrics_data = json.loads(result)
            return {"success": True, "data": metrics_data}
        except json.JSONDecodeError:
            return {"success": False, "message": result}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Generate plot endpoint
@app.post("/api/plots")
async def generate_plot(request: PlotRequest):
    """Generate visualization plots"""
    global agent
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        query_params = {
            "plot_type": request.plot_type,
            "satellite_id": request.satellite_id,
            "prediction_horizon": request.prediction_horizon
        }
        
        if request.satellite_ids:
            query_params["satellite_ids"] = request.satellite_ids
        
        result = agent.plot_tool._run(json.dumps(query_params))
        
        return {"success": True, "message": result, "plot_generated": True}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Main AI agent query endpoint
@app.post("/api/agent/query", response_model=AgentResponse)
async def query_agent(request: AgentQuery):
    """Send query to the AI agent"""
    global agent
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        response = agent.query(request.query, request.chat_history)
        
        return AgentResponse(
            response=response,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Batch analysis endpoint
@app.post("/api/analysis/batch")
async def batch_analysis(satellite_ids: List[str]):
    """Perform batch analysis on multiple satellites"""
    global agent
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        results = {}
        
        for sat_id in satellite_ids:
            # Get basic metrics for first prediction horizon
            metrics_query = {
                "satellite_id": sat_id,
                "prediction_horizon": 1,
                "metric_type": "all"
            }
            
            metrics_result = agent.metrics_tool._run(json.dumps(metrics_query))
            
            try:
                metrics_data = json.loads(metrics_result)
                results[sat_id] = {
                    "success": True,
                    "metrics": metrics_data,
                    "summary": agent.get_satellite_summary(sat_id)
                }
            except json.JSONDecodeError:
                results[sat_id] = {
                    "success": False,
                    "error": metrics_result
                }
        
        return {"results": results, "timestamp": datetime.now()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Performance comparison endpoint
@app.post("/api/analysis/compare")
async def compare_satellites(satellite_ids: List[str], prediction_horizon: int = 1):
    """Compare performance metrics across multiple satellites"""
    global agent
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        comparison_data = []
        
        for sat_id in satellite_ids:
            metrics_query = {
                "satellite_id": sat_id,
                "prediction_horizon": prediction_horizon,
                "metric_type": "all"
            }
            
            metrics_result = agent.metrics_tool._run(json.dumps(metrics_query))
            
            try:
                metrics = json.loads(metrics_result)
                comparison_data.append({
                    "satellite_id": sat_id,
                    "metrics": metrics
                })
            except json.JSONDecodeError:
                comparison_data.append({
                    "satellite_id": sat_id,
                    "error": metrics_result
                })
        
        return {
            "comparison": comparison_data,
            "prediction_horizon": f"{prediction_horizon * 30} seconds",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get prediction horizons performance
@app.get("/api/satellites/{satellite_id}/horizons")
async def get_horizon_performance(satellite_id: str, max_horizon: int = 24):
    """Get performance metrics across different prediction horizons"""
    global agent
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        horizon_data = []
        
        for horizon in range(1, min(max_horizon + 1, 241)):
            metrics_query = {
                "satellite_id": satellite_id,
                "prediction_horizon": horizon,
                "metric_type": "all"
            }
            
            metrics_result = agent.metrics_tool._run(json.dumps(metrics_query))
            
            try:
                metrics = json.loads(metrics_result)
                horizon_data.append({
                    "horizon": horizon,
                    "time_seconds": horizon * 30,
                    "metrics": metrics
                })
            except json.JSONDecodeError:
                horizon_data.append({
                    "horizon": horizon,
                    "time_seconds": horizon * 30,
                    "error": metrics_result
                })
        
        return {
            "satellite_id": satellite_id,
            "horizon_performance": horizon_data,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Anomaly detection endpoint
@app.get("/api/satellites/{satellite_id}/anomalies")
async def detect_anomalies(satellite_id: str, threshold_multiplier: float = 2.0):
    """Detect anomalies in satellite prediction errors"""
    global agent
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        actual_data, predicted_data = agent.data_manager.get_satellite_data(satellite_id)
        
        if actual_data.empty or predicted_data.empty:
            raise HTTPException(status_code=404, detail=f"No data for satellite {satellite_id}")
        
        # Merge data
        import pandas as pd
        merged = pd.merge(actual_data, predicted_data, on=['datetime', 'PRN'], how='inner')
        
        anomalies = []
        
        # Check first few prediction horizons
        for horizon in range(1, min(13, 241)):  # Up to 6 minutes
            pred_col = f'pred_{horizon}'
            if pred_col in merged.columns:
                errors = abs(merged['clock_bias_correction'] - merged[pred_col])
                threshold = errors.mean() + threshold_multiplier * errors.std()
                
                anomaly_indices = errors[errors > threshold].index
                
                for idx in anomaly_indices:
                    anomalies.append({
                        "timestamp": merged.loc[idx, 'datetime'].isoformat(),
                        "prediction_horizon": f"{horizon * 30} seconds",
                        "error_magnitude": float(errors.loc[idx]),
                        "threshold": float(threshold),
                        "actual_value": float(merged.loc[idx, 'clock_bias_correction']),
                        "predicted_value": float(merged.loc[idx, pred_col])
                    })
        
        return {
            "satellite_id": satellite_id,
            "anomalies_detected": len(anomalies),
            "threshold_multiplier": threshold_multiplier,
            "anomalies": anomalies[:50],  # Limit to first 50
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# System statistics endpoint
@app.get("/api/system/stats")
async def get_system_stats():
    """Get overall system statistics"""
    global agent
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        satellites = agent.data_manager.get_available_satellites()
        total_data_points = 0
        satellite_stats = []
        
        for sat_id in satellites:
            summary = agent.get_satellite_summary(sat_id)
            if "error" not in summary:
                total_data_points += summary["data_points"]
                satellite_stats.append({
                    "satellite_id": sat_id,
                    "data_points": summary["data_points"],
                    "time_range": summary["time_range"]
                })
        
        return {
            "total_satellites": len(satellites),
            "total_data_points": total_data_points,
            "satellite_list": satellites,
            "satellite_stats": satellite_stats,
            "agent_status": "operational",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Advanced query endpoint with context
@app.post("/api/agent/advanced_query")
async def advanced_agent_query(request: AgentQuery, background_tasks: BackgroundTasks):
    """Advanced AI agent query with enhanced context and background processing"""
    global agent
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Add system context to the query
        satellites = agent.data_manager.get_available_satellites()
        enhanced_query = f"""
Context: Available satellites: {', '.join(satellites)}
Current system status: Operational with {len(satellites)} active satellites.

User Query: {request.query}

Please provide detailed analysis including:
1. Relevant metrics calculations
2. Appropriate visualizations
3. Practical insights and recommendations
4. Any detected anomalies or patterns
"""
        
        response = agent.query(enhanced_query, request.chat_history)
        
        return AgentResponse(
            response=response,
            timestamp=datetime.now(),
            satellite_data={
                "available_satellites": satellites,
                "total_satellites": len(satellites)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Bulk export endpoint
@app.get("/api/export/{satellite_id}")
async def export_satellite_data(satellite_id: str, format: str = "json"):
    """Export satellite data and analysis results"""
    global agent
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        if format not in ["json", "csv"]:
            raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")
        
        # Get comprehensive satellite data
        summary = agent.get_satellite_summary(satellite_id)
        
        # Get metrics for multiple horizons
        metrics_data = []
        for horizon in [1, 2, 4, 8, 12, 24]:  # Key prediction horizons
            metrics_query = {
                "satellite_id": satellite_id,
                "prediction_horizon": horizon,
                "metric_type": "all"
            }
            
            result = agent.metrics_tool._run(json.dumps(metrics_query))
            try:
                metrics = json.loads(result)
                metrics_data.append(metrics)
            except json.JSONDecodeError:
                pass
        
        export_data = {
            "satellite_id": satellite_id,
            "export_timestamp": datetime.now().isoformat(),
            "summary": summary,
            "metrics_analysis": metrics_data,
            "format": format
        }
        
        if format == "json":
            return export_data
        else:
            # For CSV, we'd need to flatten the data structure
            # This is a simplified version
            return {"message": "CSV export functionality to be implemented", "data": export_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates (if needed)
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/agent")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time agent communication"""
    global agent
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            query_data = json.loads(data)
            
            if agent and "query" in query_data:
                response = agent.query(query_data["query"])
                await manager.send_personal_message(json.dumps({
                    "response": response,
                    "timestamp": datetime.now().isoformat(),
                    "query_id": query_data.get("query_id", "")
                }), websocket)
            else:
                await manager.send_personal_message(json.dumps({
                    "error": "Agent not available or invalid query",
                    "timestamp": datetime.now().isoformat()
                }), websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         app,
#         host="0.0.0.0",
#         port=8000,
#         reload=Config.DEBUG,
#         log_level="info"
#     )