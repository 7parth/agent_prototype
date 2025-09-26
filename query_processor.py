import os
from typing import Dict, Any, List
import re
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from data_loader import data_loader
from data_analyzer import data_analyzer
import logging

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self):
        # Initialize Google GenAI
        self.llm = ChatGoogleGenerativeAI(
            model="gemini=2.5-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # System prompt for the LLM
        self.system_prompt = """You are an expert GNSS satellite data analyst chatbot. 
        Your job is to understand user queries about satellite prediction performance and extract key information.
        
        Available satellites: {satellites}
        Prediction horizons: 30 seconds to {max_hours} hours (in 30-second intervals)
        
        Extract from user queries:
        1. Satellite IDs mentioned (e.g., G01, G14, G20)
        2. Query type (comparison, performance analysis, error analysis, statistics)
        3. Specific metrics requested (RMSE, MAE, error percentage, etc.)
        4. Time horizons mentioned (seconds, minutes, hours)
        5. Error thresholds mentioned (percentages, absolute values)
        
        Return your analysis as JSON with these fields:
        {{
            "query_type": "comparison|performance|error_analysis|statistics|find_satellites",
            "satellites": ["G01", "G02"],
            "prediction_horizon": 30,
            "metrics": ["rmse", "mae"],
            "error_threshold": 0.05,
            "intent": "brief description of what user wants"
        }}
        
        If satellites are not specified, use all available satellites.
        If prediction horizon is not specified, use 30 seconds.
        """
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query and return analysis results"""
        try:
            # Extract intent using LLM
            intent_data = self._extract_intent(user_query)
            
            # Execute the appropriate analysis based on intent
            result = self._execute_analysis(intent_data, user_query)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "error": "Failed to process query",
                "details": str(e)
            }
    
    def _extract_intent(self, user_query: str) -> Dict:
        """Use LLM to extract intent from user query"""
        try:
            # Get data summary for context
            summary = data_loader.get_data_summary()
            satellites = summary.get('satellites', [])
            max_hours = summary.get('prediction_horizons', {}).get('max_hours', 24)
            
            # Format system prompt with available data
            formatted_prompt = self.system_prompt.format(
                satellites=satellites,
                max_hours=max_hours
            )
            
            messages = [
                SystemMessage(content=formatted_prompt),
                HumanMessage(content=f"Extract intent from this query: '{user_query}'")
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            try:
                intent_data = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                intent_data = self._fallback_intent_parsing(user_query)
            
            return intent_data
            
        except Exception as e:
            logger.error(f"Error extracting intent: {e}")
            return self._fallback_intent_parsing(user_query)
    
    def _fallback_intent_parsing(self, user_query: str) -> Dict:
        """Fallback method to parse user intent using regex"""
        query_lower = user_query.lower()
        
        # Extract satellite IDs
        satellite_pattern = r'g\d{2}'
        satellites = re.findall(satellite_pattern, query_lower)
        satellites = [s.upper() for s in satellites]
        
        # Determine query type
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'between']):
            query_type = "comparison"
        elif any(word in query_lower for word in ['error', 'threshold', 'within', 'below']):
            query_type = "find_satellites"
        elif any(word in query_lower for word in ['performance', 'horizon', 'predict']):
            query_type = "performance"
        elif any(word in query_lower for word in ['statistics', 'stats', 'summary']):
            query_type = "statistics"
        else:
            query_type = "performance"
        
        # Extract time horizon
        time_matches = re.findall(r'(\d+)\s*(second|minute|hour|sec|min|hr)s?', query_lower)
        prediction_horizon = 30  # default
        
        if time_matches:
            value, unit = time_matches[0]
            value = int(value)
            if unit in ['minute', 'min']:
                prediction_horizon = value * 60
            elif unit in ['hour', 'hr']:
                prediction_horizon = value * 3600
            else:
                prediction_horizon = value
        
        # Extract error threshold
        error_threshold = None
        threshold_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', query_lower)
        if threshold_matches:
            error_threshold = float(threshold_matches[0]) / 100
        
        return {
            "query_type": query_type,
            "satellites": satellites if satellites else data_loader.get_available_satellites()[:5],  # limit to 5 for fallback
            "prediction_horizon": prediction_horizon,
            "metrics": ["rmse", "mae"],
            "error_threshold": error_threshold,
            "intent": query_lower
        }
    
    def _execute_analysis(self, intent_data: Dict, original_query: str) -> Dict:
        """Execute the appropriate analysis based on extracted intent"""
        query_type = intent_data.get("query_type", "performance")
        satellites = intent_data.get("satellites", [])
        prediction_horizon = intent_data.get("prediction_horizon", 30)
        error_threshold = intent_data.get("error_threshold")
        
        try:
            if query_type == "comparison":
                result = data_analyzer.compare_satellites(satellites, prediction_horizon)
                
            elif query_type == "find_satellites":
                if error_threshold:
                    result = data_analyzer.find_satellites_with_error_threshold(
                        error_threshold, prediction_horizon
                    )
                else:
                    # If no threshold specified, analyze all satellites
                    result = data_analyzer.compare_satellites(satellites, prediction_horizon)
                    
            elif query_type == "statistics":
                if len(satellites) == 1:
                    result = data_analyzer.get_satellite_statistics(satellites[0])
                else:
                    result = {sat: data_analyzer.get_satellite_statistics(sat) for sat in satellites[:3]}
                    
            elif query_type == "performance":
                if len(satellites) == 1:
                    result = data_analyzer.analyze_prediction_horizons(satellites[0])
                else:
                    result = data_analyzer.compare_satellites(satellites, prediction_horizon)
                    
            else:
                # Default to performance analysis
                result = data_analyzer.analyze_satellite_performance(satellites[0], prediction_horizon)
            
            # Add metadata
            result["query_metadata"] = {
                "original_query": original_query,
                "extracted_intent": intent_data,
                "analysis_type": query_type
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing analysis: {e}")
            return {
                "error": "Analysis failed",
                "details": str(e),
                "query_metadata": {
                    "original_query": original_query,
                    "extracted_intent": intent_data
                }
            }

# Global instance
query_processor = QueryProcessor()