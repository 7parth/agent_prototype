from typing import Dict, Any
import json

class ResponseGenerator:
    
    def generate_response(self, analysis_result: Dict[str, Any], format_type: str = "text") -> Dict[str, Any]:
        """Generate formatted response from analysis results"""
        
        if "error" in analysis_result:
            return self._format_error_response(analysis_result, format_type)
        
        # Determine response type based on analysis result structure
        if "comparison" in analysis_result:
            return self._format_comparison_response(analysis_result, format_type)
        elif "horizon_analysis" in analysis_result:
            return self._format_horizon_response(analysis_result, format_type)
        elif "qualifying_satellites" in analysis_result:
            return self._format_threshold_response(analysis_result, format_type)
        elif "metrics" in analysis_result:
            return self._format_performance_response(analysis_result, format_type)
        elif self._is_multi_satellite_stats(analysis_result):
            return self._format_statistics_response(analysis_result, format_type)
        elif "clock_bias_stats" in analysis_result:
            return self._format_statistics_response(analysis_result, format_type)
        else:
            return self._format_generic_response(analysis_result, format_type)
    
    def _is_multi_satellite_stats(self, result: Dict) -> bool:
        """Check if result contains multi-satellite statistics"""
        # Remove metadata to check actual data structure
        data_keys = [k for k in result.keys() if k != 'query_metadata']
        
        # If we have multiple keys that look like satellite IDs, it's multi-satellite stats
        if len(data_keys) >= 2:
            for key in data_keys:
                if isinstance(result[key], dict) and "clock_bias_stats" in result[key]:
                    return True
        return False
    
    def _format_comparison_response(self, result: Dict, format_type: str) -> Dict:
        """Format satellite comparison results"""
        comparison = result.get("comparison", {})
        ranking = result.get("ranking", [])
        best = result.get("best_satellite")
        worst = result.get("worst_satellite")
        
        # Check if query asks for specific ranking position
        original_query = result.get("query_metadata", {}).get("original_query", "").lower()
        ranking_query = None
        
        if "second worst" in original_query:
            ranking_query = ("second_worst", len(ranking) - 1 if len(ranking) > 1 else None)
        elif "second best" in original_query:
            ranking_query = ("second_best", 1 if len(ranking) > 1 else None)
        elif "third" in original_query and "worst" in original_query:
            ranking_query = ("third_worst", len(ranking) - 2 if len(ranking) > 2 else None)
        elif "third" in original_query and "best" in original_query:
            ranking_query = ("third_best", 2 if len(ranking) > 2 else None)
        
        if format_type == "json":
            return {
                "response_type": "comparison",
                "data": result,
                "summary": f"Compared {len(comparison)} satellites. Best: {best}, Worst: {worst}"
            }
        
        # Text format
        text_response = "🛰️ **Satellite Performance Comparison**\n\n"
        
        # If specific ranking requested, highlight it first
        if ranking_query and ranking_query[1] is not None:
            position_name, index = ranking_query
            if 0 <= index < len(ranking):
                satellite = ranking[index]['satellite']
                rmse = ranking[index]['rmse']
                text_response += f"🎯 **Answer: {satellite} is the {position_name.replace('_', ' ')}** (RMSE: {rmse:.6f})\n\n"
        
        if ranking:
            text_response += "**Performance Ranking (by RMSE - lower is better):**\n"
            for i, sat_data in enumerate(ranking, 1):
                sat = sat_data['satellite']
                rmse = sat_data['rmse']
                
                # Highlight specific positions
                prefix = ""
                if ranking_query and ranking_query[1] == i - 1:
                    prefix = "👉 "
                elif i == 1:
                    prefix = "🏆 "
                elif i == len(ranking):
                    prefix = "🔻 "
                    
                text_response += f"{prefix}{i}. {sat}: RMSE = {rmse:.6f}\n"
            
            text_response += f"\n🏆 **Best Performer:** {best} (RMSE: {ranking[0]['rmse']:.6f})\n"
            text_response += f"🔻 **Worst Performer:** {worst} (RMSE: {ranking[-1]['rmse']:.6f})\n\n"
        
        # Add detailed metrics for each satellite
        text_response += "**Detailed Metrics:**\n"
        for sat, data in comparison.items():
            if 'metrics' in data:
                metrics = data['metrics']
                text_response += f"\n**{sat}:**\n"
                text_response += f"  - RMSE: {metrics['rmse']:.6f}\n"
                text_response += f"  - MAE: {metrics['mae']:.6f}\n"
                text_response += f"  - Mean Bias Error: {metrics['mbe']:.6f}\n"
                text_response += f"  - Max Error: {metrics['max_error']:.6f}\n"
        
        return {
            "response_type": "text",
            "text": text_response,
            "data": result
        }
    
    def _format_horizon_response(self, result: Dict, format_type: str) -> Dict:
        """Format prediction horizon analysis"""
        satellite = result.get("satellite")
        horizon_analysis = result.get("horizon_analysis", {})
        trend = result.get("trend", "Unknown")
        
        if format_type == "json":
            return {
                "response_type": "horizon_analysis",
                "data": result,
                "summary": f"Horizon analysis for {satellite}: {trend}"
            }
        
        text_response = f"📈 **Prediction Horizon Analysis for {satellite}**\n\n"
        text_response += f"**Performance Trend:** {trend}\n\n"
        
        if horizon_analysis:
            text_response += "**Performance by Prediction Horizon:**\n"
            for horizon, metrics in horizon_analysis.items():
                text_response += f"\n**{horizon}:**\n"
                text_response += f"  - RMSE: {metrics['rmse']:.6f}\n"
                text_response += f"  - MAE: {metrics['mae']:.6f}\n"
                text_response += f"  - Mean Bias Error: {metrics['mbe']:.6f}\n"
        
        return {
            "response_type": "text",
            "text": text_response,
            "data": result
        }
    
    def _format_threshold_response(self, result: Dict, format_type: str) -> Dict:
        """Format error threshold search results"""
        threshold = result.get("threshold")
        qualifying_satellites = result.get("qualifying_satellites", [])
        total_satellites = result.get("total_satellites", 0)
        qualifying_count = result.get("qualifying_count", 0)
        
        if format_type == "json":
            return {
                "response_type": "threshold_search",
                "data": result,
                "summary": f"Found {qualifying_count}/{total_satellites} satellites below {threshold} error threshold"
            }
        
        text_response = f"🎯 **Satellites with RMSE Below {threshold}**\n\n"
        text_response += f"**Results:** {qualifying_count} out of {total_satellites} satellites qualify\n\n"
        
        if qualifying_count == total_satellites:
            text_response += "✅ **Great News!** All satellites meet the specified threshold!\n\n"
        
        if qualifying_satellites:
            text_response += "**Qualifying Satellites (ranked by performance):**\n"
            for i, sat_data in enumerate(qualifying_satellites, 1):
                sat = sat_data['satellite']
                rmse = sat_data['rmse']
                text_response += f"{i}. {sat}: RMSE = {rmse:.6f}\n"
        else:
            text_response += "❌ No satellites meet the specified error threshold.\n"
        
        return {
            "response_type": "text",
            "text": text_response,
            "data": result
        }
    
    def _format_statistics_response(self, result: Dict, format_type: str) -> Dict:
        """Format satellite statistics"""
        
        # Check if this is a single satellite result
        if "satellite" in result and "clock_bias_stats" in result:
            # Single satellite
            satellite = result.get("satellite")
            stats = result.get("clock_bias_stats", {})
            data_points = result.get("data_points", 0)
            time_range = result.get("time_range", {})
            
            if format_type == "json":
                return {
                    "response_type": "statistics",
                    "data": result,
                    "summary": f"Statistics for {satellite}"
                }
            
            text_response = f"📊 **Statistics for Satellite {satellite}**\n\n"
            text_response += f"**Data Points:** {data_points:,}\n"
            text_response += f"**Time Range:** {time_range.get('start', 'N/A')} to {time_range.get('end', 'N/A')}\n\n"
            text_response += "**Clock Bias Correction Statistics:**\n"
            text_response += f"  - Mean: {stats.get('mean', 0):.6f}\n"
            text_response += f"  - Standard Deviation: {stats.get('std', 0):.6f}\n"
            text_response += f"  - Minimum: {stats.get('min', 0):.6f}\n"
            text_response += f"  - Maximum: {stats.get('max', 0):.6f}\n"
            text_response += f"  - Median: {stats.get('median', 0):.6f}\n"
            
            return {
                "response_type": "text",
                "text": text_response,
                "data": result
            }
            
        else:
            # Multiple satellites - filter out query_metadata
            satellite_data = {k: v for k, v in result.items() if k != 'query_metadata'}
            
            if format_type == "json":
                return {
                    "response_type": "statistics",
                    "data": result,
                    "summary": f"Statistics comparison for {len(satellite_data)} satellites"
                }
            
            text_response = "📊 **Multi-Satellite Statistics Comparison**\n\n"
            
            # Collect stats for comparison
            satellite_stats = []
            
            for sat, stats in satellite_data.items():
                if isinstance(stats, dict) and "clock_bias_stats" in stats:
                    text_response += f"**{sat}:**\n"
                    bias_stats = stats["clock_bias_stats"]
                    data_points = stats.get('data_points', 0)
                    time_range = stats.get('time_range', {})
                    
                    text_response += f"  - Data Points: {data_points:,}\n"
                    text_response += f"  - Time Range: {time_range.get('start', 'N/A')} to {time_range.get('end', 'N/A')}\n"
                    text_response += f"  - Mean Clock Bias: {bias_stats.get('mean', 0):.6f}\n"
                    text_response += f"  - Standard Deviation: {bias_stats.get('std', 0):.6f}\n"
                    text_response += f"  - Min: {bias_stats.get('min', 0):.6f}\n"
                    text_response += f"  - Max: {bias_stats.get('max', 0):.6f}\n"
                    text_response += f"  - Median: {bias_stats.get('median', 0):.6f}\n\n"
                    
                    # Store for comparison
                    satellite_stats.append({
                        'satellite': sat,
                        'data_points': data_points,
                        'std': bias_stats.get('std', 0),
                        'range': bias_stats.get('max', 0) - bias_stats.get('min', 0)
                    })
            
            # Add comparison analysis
            if len(satellite_stats) >= 2:
                text_response += "🏆 **Performance Comparison:**\n\n"
                
                # Compare data coverage
                best_coverage = max(satellite_stats, key=lambda x: x['data_points'])
                text_response += f"**Best Data Coverage:** {best_coverage['satellite']} ({best_coverage['data_points']:,} data points)\n"
                
                # Compare stability (lower std is better)
                most_stable = min(satellite_stats, key=lambda x: x['std'])
                text_response += f"**Most Stable (lowest std dev):** {most_stable['satellite']} (std: {most_stable['std']:.6f})\n"
                
                # Compare consistency (smaller range is better)  
                most_consistent = min(satellite_stats, key=lambda x: x['range'])
                text_response += f"**Most Consistent (smallest range):** {most_consistent['satellite']} (range: {most_consistent['range']:.6f})\n"
                
                # Overall recommendation
                text_response += f"\n💡 **Overall Winner:** "
                if most_stable['satellite'] == most_consistent['satellite'] == best_coverage['satellite']:
                    text_response += f"**{most_stable['satellite']}** - Superior in all metrics (coverage, stability, and consistency)!"
                elif most_stable['satellite'] == most_consistent['satellite']:
                    text_response += f"**{most_stable['satellite']}** - Best choice for both stability and consistency."
                else:
                    # Count wins
                    wins = {}
                    for sat_info in [best_coverage, most_stable, most_consistent]:
                        sat = sat_info['satellite']
                        wins[sat] = wins.get(sat, 0) + 1
                    
                    winner = max(wins, key=wins.get)
                    text_response += f"**{winner}** - Performs best overall with {wins[winner]} out of 3 key metrics."
            
            return {
                "response_type": "text", 
                "text": text_response,
                "data": result
            }
    
    def _format_performance_response(self, result: Dict, format_type: str) -> Dict:
        """Format single satellite performance analysis"""
        satellite = result.get("satellite")
        metrics = result.get("metrics", {})
        prediction_horizon = result.get("prediction_horizon_seconds", 30)
        data_points = result.get("data_points", 0)
        error_percentiles = result.get("error_percentiles", {})
        
        if format_type == "json":
            return {
                "response_type": "performance",
                "data": result,
                "summary": f"Performance analysis for {satellite} at {prediction_horizon}s horizon"
            }
        
        text_response = f"🛰️ **Performance Analysis for Satellite {satellite}**\n\n"
        text_response += f"**Prediction Horizon:** {prediction_horizon} seconds ({prediction_horizon/60:.1f} minutes)\n"
        text_response += f"**Data Points Analyzed:** {data_points:,}\n\n"
        
        text_response += "**Error Metrics:**\n"
        text_response += f"  - RMSE: {metrics.get('rmse', 0):.6f}\n"
        text_response += f"  - MAE: {metrics.get('mae', 0):.6f}\n"
        text_response += f"  - Mean Bias Error: {metrics.get('mbe', 0):.6f}\n"
        text_response += f"  - Standard Deviation: {metrics.get('std_error', 0):.6f}\n"
        text_response += f"  - Max Absolute Error: {metrics.get('max_error', 0):.6f}\n"
        text_response += f"  - Min Absolute Error: {metrics.get('min_error', 0):.6f}\n\n"
        
        if error_percentiles:
            text_response += "**Error Distribution (Percentiles):**\n"
            text_response += f"  - 25th percentile: {error_percentiles.get('p25', 0):.6f}\n"
            text_response += f"  - 50th percentile (Median): {error_percentiles.get('p50', 0):.6f}\n"
            text_response += f"  - 75th percentile: {error_percentiles.get('p75', 0):.6f}\n"
            text_response += f"  - 90th percentile: {error_percentiles.get('p90', 0):.6f}\n"
            text_response += f"  - 95th percentile: {error_percentiles.get('p95', 0):.6f}\n"
        
        return {
            "response_type": "text",
            "text": text_response,
            "data": result
        }
    
    def _format_error_response(self, result: Dict, format_type: str) -> Dict:
        """Format error responses"""
        error_msg = result.get("error", "Unknown error occurred")
        details = result.get("details", "")
        available_satellites = result.get("available_satellites", [])
        total_available = result.get("total_available", 0)
        
        if format_type == "json":
            return {
                "response_type": "error",
                "data": result,
                "summary": error_msg
            }
        
        text_response = f"❌ **Error:** {error_msg}\n"
        
        if details:
            text_response += f"**Details:** {details}\n"
        
        if available_satellites:
            text_response += f"\n📋 **Available Satellites** ({total_available} total):\n"
            text_response += ", ".join(available_satellites)
            if len(available_satellites) < total_available:
                text_response += f" ... and {total_available - len(available_satellites)} more"
            text_response += "\n"
        
        return {
            "response_type": "error",
            "text": text_response,
            "data": result
        }
    
    def _format_generic_response(self, result: Dict, format_type: str) -> Dict:
        """Format generic responses"""
        if format_type == "json":
            return {
                "response_type": "generic",
                "data": result,
                "summary": "Analysis completed"
            }
        
        text_response = "📋 **Analysis Results:**\n\n"
        text_response += f"```json\n{json.dumps(result, indent=2)}\n```"
        
        return {
            "response_type": "text",
            "text": text_response,
            "data": result
        }

# Global instance
response_generator = ResponseGenerator()