"""
Enhanced Console Interface for EssentiaX EDA
==========================================
Phase 2: Enhanced Console Output with Interactive Features

Features:
- Interactive progress tracking
- Automatic plot generation and saving
- Enhanced user experience
- Real-time feedback
- Plot file management
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import warnings
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm
from rich.layout import Layout
from rich.live import Live
from rich.align import Align

warnings.filterwarnings("ignore")

# Initialize Rich Console
console = Console()

# Set modern styling for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


class EnhancedConsoleInterface:
    """Enhanced console interface for EDA with automatic plot generation"""
    
    def __init__(self, output_dir: str = "eda_plots"):
        self.console = Console()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.plot_counter = 0
        self.generated_plots = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create session directory
        self.session_dir = self.output_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Initialize plot settings
        self.plot_settings = {
            'save_format': 'png',
            'dpi': 300,
            'figsize': (12, 8),
            'style': 'seaborn-v0_8-darkgrid'
        }
    
    def display_welcome_banner(self, df_shape: Tuple[int, int], mode: str, target: str = None):
        """Display enhanced welcome banner"""
        banner_text = """
🧠 EssentiaX Enhanced Smart EDA Engine 🧠
        Advanced Console Interface v2.0
        """
        
        # Create info table
        info_table = Table(show_header=False, box=box.ROUNDED, border_style="blue")
        info_table.add_column("Metric", style="cyan", width=20)
        info_table.add_column("Value", style="bold green", width=30)
        
        info_table.add_row("📊 Dataset Shape", f"{df_shape[0]:,} × {df_shape[1]}")
        info_table.add_row("🎯 Target Variable", target if target else "Auto-Detect")
        info_table.add_row("📁 Output Directory", str(self.session_dir))
        info_table.add_row("🕒 Session ID", self.session_id)
        info_table.add_row("⚙️ Mode", mode.upper())
        
        # Display banner
        self.console.print("\n" + "="*80)
        self.console.print(Align.center(Text(banner_text, style="bold magenta")))
        self.console.print("="*80)
        self.console.print(Panel(info_table, title="📋 Session Configuration", border_style="blue"))
    
    def create_progress_tracker(self, total_steps: int = 10) -> Progress:
        """Create enhanced progress tracker"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            expand=True
        )
    
    def display_section_header(self, section_number: int, title: str, description: str = ""):
        """Display enhanced section headers"""
        header_text = f"{section_number}️⃣ {title}"
        
        if description:
            panel_content = f"[bold cyan]{title}[/bold cyan]\n[dim]{description}[/dim]"
        else:
            panel_content = f"[bold cyan]{title}[/bold cyan]"
        
        self.console.print(f"\n{header_text}")
        self.console.print("-" * 70)
    
    def save_plot_matplotlib(self, fig, plot_name: str, plot_type: str = "analysis") -> str:
        """Save matplotlib plot and return file path"""
        self.plot_counter += 1
        filename = f"{self.plot_counter:02d}_{plot_type}_{plot_name}.{self.plot_settings['save_format']}"
        filepath = self.session_dir / filename
        
        try:
            fig.savefig(
                filepath,
                dpi=self.plot_settings['dpi'],
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            plt.close(fig)
            
            self.generated_plots.append({
                'filename': filename,
                'filepath': str(filepath),
                'plot_type': plot_type,
                'plot_name': plot_name,
                'timestamp': datetime.now()
            })
            
            self.console.print(f"   📁 Plot saved: [green]{filename}[/green]")
            return str(filepath)
            
        except Exception as e:
            self.console.print(f"   ⚠️ Failed to save plot {filename}: {str(e)}")
            return None
    
    def save_plot_plotly(self, fig, plot_name: str, plot_type: str = "interactive") -> str:
        """Save plotly plot as HTML and PNG"""
        self.plot_counter += 1
        
        # Save as HTML (interactive)
        html_filename = f"{self.plot_counter:02d}_{plot_type}_{plot_name}.html"
        html_filepath = self.session_dir / html_filename
        
        # Save as PNG (static)
        png_filename = f"{self.plot_counter:02d}_{plot_type}_{plot_name}.png"
        png_filepath = self.session_dir / png_filename
        
        try:
            # Save HTML
            fig.write_html(html_filepath)
            
            # Save PNG (if kaleido is available)
            try:
                fig.write_image(png_filepath, width=1200, height=800, scale=2)
                png_saved = True
            except Exception:
                png_saved = False
            
            self.generated_plots.append({
                'filename': html_filename,
                'filepath': str(html_filepath),
                'plot_type': plot_type,
                'plot_name': plot_name,
                'timestamp': datetime.now(),
                'has_png': png_saved
            })
            
            if png_saved:
                self.console.print(f"   📁 Interactive plot saved: [green]{html_filename}[/green] & [green]{png_filename}[/green]")
            else:
                self.console.print(f"   📁 Interactive plot saved: [green]{html_filename}[/green]")
                self.console.print(f"   💡 Install kaleido for PNG export: pip install kaleido")
            
            return str(html_filepath)
            
        except Exception as e:
            self.console.print(f"   ⚠️ Failed to save plot {html_filename}: {str(e)}")
            return None
    def create_distribution_plot(self, df: pd.DataFrame, column: str, target: str = None) -> str:
        """Create and save distribution plot"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Distribution Analysis: {column}', fontsize=16, fontweight='bold')
            
            data = df[column].dropna()
            
            # Histogram
            axes[0, 0].hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Histogram')
            axes[0, 0].set_xlabel(column)
            axes[0, 0].set_ylabel('Frequency')
            
            # Box plot
            axes[0, 1].boxplot(data, vert=True, patch_artist=True,
                              boxprops=dict(facecolor='lightgreen', alpha=0.7))
            axes[0, 1].set_title('Box Plot')
            axes[0, 1].set_ylabel(column)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(data, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
            
            # Density plot
            axes[1, 1].hist(data, bins=30, density=True, alpha=0.7, color='orange', edgecolor='black')
            data_sorted = np.sort(data)
            kde = stats.gaussian_kde(data)
            axes[1, 1].plot(data_sorted, kde(data_sorted), 'r-', linewidth=2, label='KDE')
            axes[1, 1].set_title('Density Plot with KDE')
            axes[1, 1].set_xlabel(column)
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].legend()
            
            plt.tight_layout()
            return self.save_plot_matplotlib(fig, f"distribution_{column}", "distribution")
            
        except Exception as e:
            self.console.print(f"   ⚠️ Could not create distribution plot for {column}: {str(e)}")
            return None
    
    def create_categorical_plot(self, df: pd.DataFrame, column: str, target: str = None) -> str:
        """Create and save categorical analysis plot"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Categorical Analysis: {column}', fontsize=16, fontweight='bold')
            
            # Value counts
            value_counts = df[column].value_counts().head(15)
            axes[0, 0].bar(range(len(value_counts)), value_counts.values, color='lightcoral')
            axes[0, 0].set_title('Value Counts (Top 15)')
            axes[0, 0].set_xlabel('Categories')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_xticks(range(len(value_counts)))
            axes[0, 0].set_xticklabels(value_counts.index, rotation=45, ha='right')
            
            # Pie chart (top 10)
            top_10 = df[column].value_counts().head(10)
            axes[0, 1].pie(top_10.values, labels=top_10.index, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Distribution (Top 10)')
            
            # Percentage distribution
            pct_counts = df[column].value_counts(normalize=True).head(15) * 100
            axes[1, 0].barh(range(len(pct_counts)), pct_counts.values, color='lightgreen')
            axes[1, 0].set_title('Percentage Distribution (Top 15)')
            axes[1, 0].set_xlabel('Percentage (%)')
            axes[1, 0].set_ylabel('Categories')
            axes[1, 0].set_yticks(range(len(pct_counts)))
            axes[1, 0].set_yticklabels(pct_counts.index)
            
            # Target relationship (if target provided)
            if target and target in df.columns:
                try:
                    crosstab = pd.crosstab(df[column], df[target], normalize='index') * 100
                    crosstab.plot(kind='bar', ax=axes[1, 1], stac