"""
Smart Data Handling Utilities for EssentiaX Streamlit App
========================================================
Memory-efficient data processing for large datasets

Features:
- Smart sampling for Streamlit processing
- Chunked processing engine
- Progressive loading system
- Memory usage monitoring
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Callable, Dict, Any, Tuple
import psutil
import os
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import EssentiaX modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from essentiax.io.smart_read import smart_read
    from essentiax.visuals.big_data_plots import smart_sample_for_plots
    ESSENTIAX_AVAILABLE = True
except ImportError:
    ESSENTIAX_AVAILABLE = False


class SmartDataHandler:
    """Smart data handling for Streamlit applications"""
    
    def __init__(self, max_memory_mb: int = 500):
        self.max_memory_mb = max_memory_mb
        self.current_memory_usage = 0
        self.processing_stats = {}
    
    def create_smart_sample(
        self, 
        df: pd.DataFrame, 
        target_size: int = 50000, 
        preserve_distribution: bool = True,
        target_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Create representative sample for Streamlit processing
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        target_size : int
            Target sample size
        preserve_distribution : bool
            Whether to preserve statistical distributions
        target_col : str, optional
            Target column for stratified sampling
            
        Returns:
        --------
        tuple : (sampled_dataframe, sampling_stats)
        """
        original_size = len(df)
        
        # If dataset is already small enough, return as-is
        if original_size <= target_size:
            return df.copy(), {
                'original_size': original_size,
                'sample_size': original_size,
                'sampling_ratio': 1.0,
                'sampling_method': 'no_sampling',
                'preserved_distribution': True
            }
        
        sampling_stats = {
            'original_size': original_size,
            'target_size': target_size,
            'sampling_ratio': target_size / original_size,
            'sampling_method': 'random',
            'preserved_distribution': preserve_distribution
        }
        
        try:
            if preserve_distribution and target_col and target_col in df.columns:
                # Stratified sampling by target
                sampled_dfs = []
                target_counts = df[target_col].value_counts()
                
                for value in target_counts.index:
                    subset = df[df[target_col] == value]
                    subset_target_size = int(target_size * len(subset) / original_size)
                    
                    if subset_target_size > 0:
                        if len(subset) <= subset_target_size:
                            sampled_dfs.append(subset)
                        else:
                            sampled_subset = subset.sample(n=subset_target_size, random_state=42)
                            sampled_dfs.append(sampled_subset)
                
                sampled_df = pd.concat(sampled_dfs, ignore_index=True)
                sampling_stats['sampling_method'] = 'stratified'
                
            else:
                # Simple random sampling
                sampled_df = df.sample(n=target_size, random_state=42)
                sampling_stats['sampling_method'] = 'random'
            
            # Preserve outliers if requested
            if preserve_distribution:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                outlier_indices = set()
                
                for col in numeric_cols[:5]:  # Limit to 5 columns for performance
                    try:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        if IQR > 0:
                            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                            outlier_indices.update(outliers.index[:20])  # Limit outliers
                    except:
                        continue
                
                if outlier_indices:
                    outlier_df = df.loc[list(outlier_indices)]
                    sampled_df = pd.concat([sampled_df, outlier_df]).drop_duplicates().reset_index(drop=True)
                    sampling_stats['outliers_preserved'] = len(outlier_indices)
            
            sampling_stats['sample_size'] = len(sampled_df)
            sampling_stats['actual_ratio'] = len(sampled_df) / original_size
            
            return sampled_df, sampling_stats
            
        except Exception as e:
            st.error(f"Error in smart sampling: {str(e)}")
            # Fallback to simple random sampling
            sampled_df = df.sample(n=min(target_size, len(df)), random_state=42)
            sampling_stats.update({
                'sample_size': len(sampled_df),
                'sampling_method': 'fallback_random',
                'error': str(e)
            })
            return sampled_df, sampling_stats
    
    def process_large_dataset_chunks(
        self,
        file_path: str,
        chunk_size: int = 10000,
        analysis_func: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process large datasets in chunks for memory efficiency
        
        Parameters:
        -----------
        file_path : str
            Path to the dataset file
        chunk_size : int
            Size of each chunk
        analysis_func : callable, optional
            Function to apply to each chunk
        progress_callback : callable, optional
            Callback for progress updates
            
        Returns:
        --------
        dict : Aggregated results from all chunks
        """
        results = {
            'total_rows': 0,
            'chunks_processed': 0,
            'processing_time': 0,
            'memory_usage': [],
            'chunk_results': [],
            'aggregated_stats': {}
        }
        
        start_time = time.time()
        
        try:
            # Determine file type and create chunk iterator
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.csv':
                chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size)
            elif file_ext in ['.xlsx', '.xls']:
                # For Excel files, we need to load and then chunk
                df_full = pd.read_excel(file_path)
                chunk_iterator = [df_full[i:i+chunk_size] for i in range(0, len(df_full), chunk_size)]
                results['total_rows'] = len(df_full)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Process chunks
            for i, chunk in enumerate(chunk_iterator):
                # Memory monitoring
                memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
                results['memory_usage'].append(memory_usage)
                
                # Process chunk
                if analysis_func:
                    chunk_result = analysis_func(chunk)
                    results['chunk_results'].append(chunk_result)
                
                results['chunks_processed'] += 1
                results['total_rows'] += len(chunk)
                
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, results['total_rows'])
                
                # Memory check
                if memory_usage > self.max_memory_mb:
                    st.warning(f"Memory usage ({memory_usage:.1f} MB) exceeds limit ({self.max_memory_mb} MB)")
                    break
            
            results['processing_time'] = time.time() - start_time
            results['avg_memory_usage'] = np.mean(results['memory_usage'])
            results['max_memory_usage'] = np.max(results['memory_usage'])
            
            # Aggregate results if analysis function was provided
            if results['chunk_results'] and analysis_func:
                results['aggregated_stats'] = self._aggregate_chunk_results(results['chunk_results'])
            
            return results
            
        except Exception as e:
            results['error'] = str(e)
            results['processing_time'] = time.time() - start_time
            return results
    
    def progressive_data_loader(
        self,
        file_path: str,
        initial_rows: int = 1000
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load data progressively for better UX
        
        Parameters:
        -----------
        file_path : str
            Path to the dataset file
        initial_rows : int
            Number of rows to load initially
            
        Returns:
        --------
        tuple : (initial_sample, loader_info)
        """
        loader_info = {
            'file_path': file_path,
            'file_size_mb': 0,
            'total_rows': 0,
            'initial_rows': initial_rows,
            'load_time': 0,
            'can_load_full': False
        }
        
        start_time = time.time()
        
        try:
            # Get file size
            file_size = os.path.getsize(file_path)
            loader_info['file_size_mb'] = file_size / 1024 / 1024
            
            # Determine loading strategy based on file size
            if loader_info['file_size_mb'] < 50:  # Small file, load completely
                if ESSENTIAX_AVAILABLE:
                    df = smart_read(file_path)
                else:
                    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
                
                loader_info['total_rows'] = len(df)
                loader_info['can_load_full'] = True
                
                # Return initial sample
                initial_sample = df.head(initial_rows) if len(df) > initial_rows else df
                
            else:  # Large file, load sample first
                file_ext = Path(file_path).suffix.lower()
                
                if file_ext == '.csv':
                    initial_sample = pd.read_csv(file_path, nrows=initial_rows)
                    # Estimate total rows
                    with open(file_path, 'r') as f:
                        total_lines = sum(1 for _ in f)
                    loader_info['total_rows'] = total_lines - 1  # Subtract header
                else:
                    # For Excel, we need to load to get row count
                    df = pd.read_excel(file_path)
                    loader_info['total_rows'] = len(df)
                    initial_sample = df.head(initial_rows)
                    loader_info['can_load_full'] = loader_info['file_size_mb'] < 200
            
            loader_info['load_time'] = time.time() - start_time
            return initial_sample, loader_info
            
        except Exception as e:
            loader_info['error'] = str(e)
            loader_info['load_time'] = time.time() - start_time
            # Return empty dataframe on error
            return pd.DataFrame(), loader_info
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """Monitor current memory usage"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def _aggregate_chunk_results(self, chunk_results: list) -> Dict[str, Any]:
        """Aggregate results from multiple chunks"""
        if not chunk_results:
            return {}
        
        # This is a basic aggregation - can be extended based on specific needs
        aggregated = {
            'total_chunks': len(chunk_results),
            'chunk_sizes': [len(result) if isinstance(result, pd.DataFrame) else 1 for result in chunk_results]
        }
        
        # If results are dictionaries, try to aggregate numeric values
        if all(isinstance(result, dict) for result in chunk_results):
            all_keys = set()
            for result in chunk_results:
                all_keys.update(result.keys())
            
            for key in all_keys:
                values = [result.get(key, 0) for result in chunk_results if key in result]
                if values and all(isinstance(v, (int, float)) for v in values):
                    aggregated[f'{key}_sum'] = sum(values)
                    aggregated[f'{key}_mean'] = np.mean(values)
                    aggregated[f'{key}_max'] = max(values)
                    aggregated[f'{key}_min'] = min(values)
        
        return aggregated


# Convenience functions for direct use
def create_smart_sample(df: pd.DataFrame, target_size: int = 50000, target_col: str = None) -> Tuple[pd.DataFrame, Dict]:
    """Create smart sample for Streamlit processing"""
    handler = SmartDataHandler()
    return handler.create_smart_sample(df, target_size, target_col=target_col)

def load_data_progressively(file_path: str, initial_rows: int = 1000) -> Tuple[pd.DataFrame, Dict]:
    """Load data progressively for better UX"""
    handler = SmartDataHandler()
    return handler.progressive_data_loader(file_path, initial_rows)

def process_in_chunks(file_path: str, chunk_size: int = 10000, analysis_func: Callable = None) -> Dict:
    """Process large dataset in chunks"""
    handler = SmartDataHandler()
    return handler.process_large_dataset_chunks(file_path, chunk_size, analysis_func)

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    handler = SmartDataHandler()
    return handler.monitor_memory_usage()


# Example usage and testing
if __name__ == "__main__":
    # Create test data
    test_df = pd.DataFrame({
        'A': np.random.normal(0, 1, 100000),
        'B': np.random.exponential(2, 100000),
        'C': np.random.choice(['X', 'Y', 'Z'], 100000),
        'target': np.random.choice([0, 1], 100000)
    })
    
    print("=== Smart Data Handler Test ===")
    
    # Test smart sampling
    handler = SmartDataHandler()
    sampled_df, stats = handler.create_smart_sample(test_df, target_size=10000, target_col='target')
    
    print(f"Original size: {stats['original_size']:,}")
    print(f"Sample size: {stats['sample_size']:,}")
    print(f"Sampling method: {stats['sampling_method']}")
    print(f"Sampling ratio: {stats['sampling_ratio']:.3f}")
    
    # Test memory monitoring
    memory_stats = handler.monitor_memory_usage()
    print(f"\nMemory usage: {memory_stats['rss_mb']:.1f} MB ({memory_stats['percent']:.1f}%)")
    
    print("\n✅ Smart Data Handler test completed!")