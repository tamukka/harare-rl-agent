import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import glob
from datetime import datetime

class TensorBoardAnalyzer:
    def __init__(self, logdir="./logs/"):
        self.logdir = logdir
        self.runs = self._get_available_runs()
        
    def _get_available_runs(self):
        """Get all available TensorBoard runs"""
        runs = {}
        for run_dir in glob.glob(os.path.join(self.logdir, "*")):
            if os.path.isdir(run_dir):
                run_name = os.path.basename(run_dir)
                runs[run_name] = run_dir
        return runs
    
    def load_run_data(self, run_name):
        """Load data from a specific TensorBoard run"""
        if run_name not in self.runs:
            return None
            
        run_path = self.runs[run_name]
        ea = EventAccumulator(run_path)
        ea.Reload()
        
        # Extract scalar data
        scalar_keys = ea.Tags()['scalars']
        data = {}
        
        for key in scalar_keys:
            scalar_events = ea.Scalars(key)
            data[key] = {
                'steps': [event.step for event in scalar_events],
                'values': [event.value for event in scalar_events],
                'timestamps': [event.wall_time for event in scalar_events]
            }
            
        return data
    
    def create_training_overview(self):
        """Create comprehensive training overview from all runs"""
        all_data = {}
        for run_name in self.runs:
            run_data = self.load_run_data(run_name)
            if run_data:
                all_data[run_name] = run_data
        
        if not all_data:
            return None, None, None
        
        # Create comparison charts
        fig_loss = go.Figure()
        fig_reward = go.Figure()
        fig_lr = go.Figure()
        
        for run_name, run_data in all_data.items():
            # Policy loss
            if 'train/loss' in run_data:
                fig_loss.add_trace(go.Scatter(
                    x=run_data['train/loss']['steps'],
                    y=run_data['train/loss']['values'],
                    mode='lines',
                    name=f'{run_name} - Loss',
                    line=dict(width=2)
                ))
            
            # Learning rate
            if 'train/learning_rate' in run_data:
                fig_lr.add_trace(go.Scatter(
                    x=run_data['train/learning_rate']['steps'],
                    y=run_data['train/learning_rate']['values'],
                    mode='lines',
                    name=f'{run_name} - LR',
                    line=dict(width=2)
                ))
        
        # Update layouts
        fig_loss.update_layout(
            title='Training Loss Comparison Across Runs',
            xaxis_title='Training Steps',
            yaxis_title='Loss',
            hovermode='x unified'
        )
        
        fig_lr.update_layout(
            title='Learning Rate Across Runs',
            xaxis_title='Training Steps',
            yaxis_title='Learning Rate',
            hovermode='x unified'
        )
        
        return fig_loss, fig_lr, all_data
    
    def get_latest_metrics(self, run_name=None):
        """Get latest metrics from the most recent run"""
        if run_name is None:
            # Get the most recent run
            if not self.runs:
                return None
            run_name = max(self.runs.keys())
        
        run_data = self.load_run_data(run_name)
        if not run_data:
            return None
        
        latest_metrics = {}
        for metric, data in run_data.items():
            if data['values']:
                latest_metrics[metric] = {
                    'value': data['values'][-1],
                    'step': data['steps'][-1],
                    'timestamp': datetime.fromtimestamp(data['timestamps'][-1])
                }
        
        return latest_metrics
    
    def create_detailed_training_analysis(self, run_name):
        """Create detailed analysis for a specific run"""
        run_data = self.load_run_data(run_name)
        if not run_data:
            return None
        
        # Create subplots for different metrics
        fig = go.Figure()
        
        # Add traces for different metrics
        metrics_to_plot = [
            ('train/loss', 'Loss', 'blue'),
            ('train/value_loss', 'Value Loss', 'red'),
            ('train/policy_gradient_loss', 'Policy Gradient Loss', 'green'),
            ('train/entropy_loss', 'Entropy Loss', 'orange'),
            ('train/approx_kl', 'Approximate KL', 'purple'),
            ('train/clip_fraction', 'Clip Fraction', 'brown'),
            ('train/explained_variance', 'Explained Variance', 'pink')
        ]
        
        for metric, name, color in metrics_to_plot:
            if metric in run_data:
                fig.add_trace(go.Scatter(
                    x=run_data[metric]['steps'],
                    y=run_data[metric]['values'],
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2),
                    visible='legendonly' if metric != 'train/loss' else True
                ))
        
        fig.update_layout(
            title=f'Detailed Training Metrics - {run_name}',
            xaxis_title='Training Steps',
            yaxis_title='Metric Value',
            hovermode='x unified',
            height=600
        )
        
        return fig
    
    def get_training_summary(self):
        """Get comprehensive training summary"""
        summary = {
            'total_runs': len(self.runs),
            'runs': list(self.runs.keys()),
            'latest_run': max(self.runs.keys()) if self.runs else None,
            'run_details': {}
        }
        
        for run_name in self.runs:
            run_data = self.load_run_data(run_name)
            if run_data:
                # Calculate run statistics
                total_steps = 0
                final_loss = None
                
                if 'train/loss' in run_data and run_data['train/loss']['steps']:
                    total_steps = max(run_data['train/loss']['steps'])
                    final_loss = run_data['train/loss']['values'][-1]
                
                summary['run_details'][run_name] = {
                    'total_steps': total_steps,
                    'final_loss': final_loss,
                    'metrics_count': len(run_data),
                    'start_time': min([min(data['timestamps']) for data in run_data.values()]) if run_data else None,
                    'end_time': max([max(data['timestamps']) for data in run_data.values()]) if run_data else None
                }
        
        return summary