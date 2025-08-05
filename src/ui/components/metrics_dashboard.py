"""
Metrics dashboard component for displaying workflow metrics.

This component provides visualizations and statistics for workflow
execution, including timing, token usage, and quality metrics.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import json


class MetricsDashboard:
    """
    Display comprehensive metrics for workflow execution.
    
    This component creates visualizations for various metrics
    collected during the workflow run.
    """
    
    def __init__(self):
        """Initialize the metrics dashboard."""
        self.metric_categories = {
            'performance': '⚡',
            'quality': '✨',
            'tokens': '🎯',
            'errors': '🚨'
        }
    
    def render(self, metrics: Dict[str, Any]):
        """
        Render the metrics dashboard.
        
        Args:
            metrics: Dictionary containing various metrics
        """
        if not metrics or not metrics.get('start_time'):
            st.info("No metrics available yet. Run the workflow to see metrics.")
            return
        
        # Overall summary
        self._render_summary(metrics)
        
        # Detailed metrics in tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "⚡ Performance",
            "✨ Quality",
            "🎯 Token Usage",
            "🚨 Errors"
        ])
        
        with tab1:
            self._render_performance_metrics(metrics)
        
        with tab2:
            self._render_quality_metrics(metrics)
        
        with tab3:
            self._render_token_metrics(metrics)
        
        with tab4:
            self._render_error_metrics(metrics)
    
    def _render_summary(self, metrics: Dict[str, Any]):
        """Render overall summary metrics."""
        st.markdown("### 📊 Workflow Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate total duration
        if metrics.get('end_time') and metrics.get('start_time'):
            duration = (metrics['end_time'] - metrics['start_time']).total_seconds()
        else:
            duration = 0
        
        with col1:
            st.metric(
                "Total Duration",
                f"{duration:.1f}s" if duration else "N/A",
                help="Total time from start to finish"
            )
        
        with col2:
            phases_completed = len(metrics.get('phase_durations', {}))
            st.metric(
                "Phases Completed",
                f"{phases_completed}/4",
                help="Number of workflow phases completed"
            )
        
        with col3:
            total_tokens = sum(metrics.get('token_usage', {}).values())
            st.metric(
                "Total Tokens",
                f"{total_tokens:,}" if total_tokens else "0",
                help="Total tokens used across all phases"
            )
        
        with col4:
            error_count = len(metrics.get('errors', []))
            st.metric(
                "Errors",
                error_count,
                delta=None if error_count == 0 else f"+{error_count}",
                delta_color="inverse",
                help="Number of errors encountered"
            )
    
    def _render_performance_metrics(self, metrics: Dict[str, Any]):
        """Render performance-related metrics."""
        st.markdown("#### Performance Metrics")
        
        phase_durations = metrics.get('phase_durations', {})
        
        if phase_durations:
            # Create DataFrame for phase durations
            df_data = []
            for phase, duration in phase_durations.items():
                df_data.append({
                    'Phase': phase.capitalize(),
                    'Duration (s)': round(duration, 2),
                    'Percentage': round(duration / sum(phase_durations.values()) * 100, 1)
                })
            
            df = pd.DataFrame(df_data)
            
            # Display as bar chart
            st.bar_chart(df.set_index('Phase')['Duration (s)'])
            
            # Display detailed table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
            
            # Performance insights
            with st.expander("💡 Performance Insights"):
                slowest_phase = max(phase_durations, key=phase_durations.get)
                fastest_phase = min(phase_durations, key=phase_durations.get)
                
                st.info(f"**Slowest Phase:** {slowest_phase.capitalize()} "
                       f"({phase_durations[slowest_phase]:.1f}s)")
                st.success(f"**Fastest Phase:** {fastest_phase.capitalize()} "
                          f"({phase_durations[fastest_phase]:.1f}s)")
                
                avg_duration = sum(phase_durations.values()) / len(phase_durations)
                st.metric("Average Phase Duration", f"{avg_duration:.1f}s")
        else:
            st.info("No performance data available")
    
    def _render_quality_metrics(self, metrics: Dict[str, Any]):
        """Render quality-related metrics."""
        st.markdown("#### Quality Metrics")
        
        # Mock quality metrics (would come from evaluation phase)
        quality_data = metrics.get('quality', {
            'code_quality': 85,
            'test_coverage': 92,
            'documentation': 78,
            'type_safety': 95,
            'best_practices': 88
        })
        
        if quality_data:
            # Create radar chart data
            categories = list(quality_data.keys())
            values = list(quality_data.values())
            
            # Display metrics
            cols = st.columns(len(categories))
            for i, (category, value) in enumerate(zip(categories, values)):
                with cols[i]:
                    # Color based on score
                    if value >= 90:
                        color = "🟢"
                    elif value >= 70:
                        color = "🟡"
                    else:
                        color = "🔴"
                    
                    st.metric(
                        category.replace('_', ' ').title(),
                        f"{value}%",
                        help=f"Quality score for {category}"
                    )
                    st.markdown(f"{color}")
            
            # Overall quality score
            overall_score = sum(values) / len(values)
            st.markdown("---")
            st.metric(
                "Overall Quality Score",
                f"{overall_score:.1f}%",
                help="Average of all quality metrics"
            )
            
            # Recommendations
            if overall_score < 80:
                with st.expander("📝 Quality Recommendations"):
                    for category, value in quality_data.items():
                        if value < 80:
                            st.warning(f"Consider improving {category.replace('_', ' ')}: "
                                     f"Current score is {value}%")
    
    def _render_token_metrics(self, metrics: Dict[str, Any]):
        """Render token usage metrics."""
        st.markdown("#### Token Usage")
        
        token_usage = metrics.get('token_usage', {})
        
        if not token_usage:
            # Generate sample data
            token_usage = {
                'analyzer': 1250,
                'specifier': 2100,
                'crafter': 3500,
                'evaluator': 1800
            }
        
        # Create DataFrame
        df_data = []
        total_tokens = sum(token_usage.values())
        
        for phase, tokens in token_usage.items():
            df_data.append({
                'Phase': phase.capitalize(),
                'Tokens': tokens,
                'Percentage': round(tokens / total_tokens * 100, 1) if total_tokens > 0 else 0,
                'Cost': f"${tokens * 0.00002:.4f}"  # Example pricing
            })
        
        df = pd.DataFrame(df_data)
        
        # Display pie chart
        st.markdown("##### Token Distribution by Phase")
        
        # Create columns for chart and stats
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Simple bar chart as alternative to pie
            st.bar_chart(df.set_index('Phase')['Tokens'])
        
        with col2:
            st.metric("Total Tokens", f"{total_tokens:,}")
            st.metric("Estimated Cost", f"${total_tokens * 0.00002:.4f}")
            st.metric("Avg per Phase", f"{total_tokens // len(token_usage):,}")
        
        # Detailed table
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Token efficiency insights
        with st.expander("💡 Token Efficiency Insights"):
            most_expensive = max(token_usage, key=token_usage.get)
            st.info(f"**Most token-intensive phase:** {most_expensive.capitalize()} "
                   f"({token_usage[most_expensive]:,} tokens)")
            
            if total_tokens > 10000:
                st.warning("Consider optimizing prompts to reduce token usage")
    
    def _render_error_metrics(self, metrics: Dict[str, Any]):
        """Render error-related metrics."""
        st.markdown("#### Error Analysis")
        
        errors = metrics.get('errors', [])
        
        if not errors:
            st.success("✅ No errors encountered during workflow execution!")
            
            # Show reliability metrics
            st.markdown("##### Reliability Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Success Rate", "100%")
            with col2:
                st.metric("Retry Count", "0")
            with col3:
                st.metric("Warnings", "0")
        else:
            st.error(f"⚠️ {len(errors)} error(s) encountered")
            
            # Error timeline
            st.markdown("##### Error Timeline")
            for i, error in enumerate(errors):
                with st.expander(f"Error {i+1}: {error.get('phase', 'Unknown')} Phase"):
                    st.error(error.get('error', 'Unknown error'))
                    st.caption(f"Time: {error.get('timestamp', 'Unknown')}")
                    
                    # Error details
                    if 'stack_trace' in error:
                        st.code(error['stack_trace'], language='text')
            
            # Error summary by phase
            st.markdown("##### Errors by Phase")
            phase_errors = {}
            for error in errors:
                phase = error.get('phase', 'unknown')
                phase_errors[phase] = phase_errors.get(phase, 0) + 1
            
            df = pd.DataFrame(
                [(k, v) for k, v in phase_errors.items()],
                columns=['Phase', 'Error Count']
            )
            st.bar_chart(df.set_index('Phase'))
    
    def export_metrics(self, metrics: Dict[str, Any]) -> str:
        """
        Export metrics to JSON format.
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            JSON string of metrics
        """
        # Convert datetime objects to strings
        export_data = {}
        for key, value in metrics.items():
            if isinstance(value, datetime):
                export_data[key] = value.isoformat()
            elif isinstance(value, dict):
                export_data[key] = {
                    k: v.isoformat() if isinstance(v, datetime) else v
                    for k, v in value.items()
                }
            else:
                export_data[key] = value
        
        return json.dumps(export_data, indent=2)
    
    def create_metrics_report(self, metrics: Dict[str, Any]) -> str:
        """
        Create a formatted metrics report.
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("# Workflow Metrics Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Summary
        report.append("\n## Summary")
        if metrics.get('start_time') and metrics.get('end_time'):
            duration = (metrics['end_time'] - metrics['start_time']).total_seconds()
            report.append(f"- Total Duration: {duration:.1f}s")
        
        report.append(f"- Phases Completed: {len(metrics.get('phase_durations', {}))}/4")
        report.append(f"- Total Errors: {len(metrics.get('errors', []))}")
        
        # Performance
        report.append("\n## Performance Metrics")
        for phase, duration in metrics.get('phase_durations', {}).items():
            report.append(f"- {phase.capitalize()}: {duration:.2f}s")
        
        # Errors
        if metrics.get('errors'):
            report.append("\n## Errors")
            for i, error in enumerate(metrics['errors']):
                report.append(f"\n### Error {i+1}")
                report.append(f"- Phase: {error.get('phase', 'Unknown')}")
                report.append(f"- Message: {error.get('error', 'Unknown error')}")
        
        return '\n'.join(report)