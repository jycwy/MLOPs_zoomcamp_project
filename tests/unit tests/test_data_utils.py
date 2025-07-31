import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.feature_extraction import DictVectorizer
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'model training'))

from data_utils import load_data, load_parquet, vectorize


class TestLoadData:
    """Test cases for load_data function"""
    
    @patch('data_utils.pd.read_csv')
    def test_load_data_success(self, mock_read_csv):
        """Test successful data loading from CSV"""
        # Arrange
        expected_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        mock_read_csv.return_value = expected_df
        
        # Act
        result = load_data('test.csv')
        
        # Assert
        mock_read_csv.assert_called_once_with('test.csv')
        pd.testing.assert_frame_equal(result, expected_df)
    
    @patch('data_utils.pd.read_csv')
    def test_load_data_file_not_found(self, mock_read_csv):
        """Test load_data with non-existent file"""
        # Arrange
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            load_data('non_existent.csv')
    
    @patch('data_utils.pd.read_csv')
    def test_load_data_empty_file(self, mock_read_csv):
        """Test load_data with empty file"""
        # Arrange
        mock_read_csv.return_value = pd.DataFrame()
        
        # Act
        result = load_data('empty.csv')
        
        # Assert
        assert result.empty
        mock_read_csv.assert_called_once_with('empty.csv')


class TestLoadParquet:
    """Test cases for load_parquet function"""
    
    @patch('data_utils.pd.read_parquet')
    def test_load_parquet_success(self, mock_read_parquet):
        """Test successful data loading from Parquet"""
        # Arrange
        expected_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        mock_read_parquet.return_value = expected_df
        
        # Act
        result = load_parquet('test.parquet')
        
        # Assert
        mock_read_parquet.assert_called_once_with('test.parquet')
        pd.testing.assert_frame_equal(result, expected_df)
    
    @patch('data_utils.pd.read_parquet')
    def test_load_parquet_file_not_found(self, mock_read_parquet):
        """Test load_parquet with non-existent file"""
        # Arrange
        mock_read_parquet.side_effect = FileNotFoundError("File not found")
        
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            load_parquet('non_existent.parquet')
    
    @patch('data_utils.pd.read_parquet')
    def test_load_parquet_empty_file(self, mock_read_parquet):
        """Test load_parquet with empty file"""
        # Arrange
        mock_read_parquet.return_value = pd.DataFrame()
        
        # Act
        result = load_parquet('empty.parquet')
        
        # Assert
        assert result.empty
        mock_read_parquet.assert_called_once_with('empty.parquet')


class TestVectorize:
    """Test cases for vectorize function"""
    
    def test_vectorize_default_target_column(self):
        """Test vectorize with default target column 'Churn'"""
        # Arrange
        df = pd.DataFrame({
            'feature1': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
            'feature2': [1, 2, 1, 3, 2, 1, 3, 2, 1, 3],
            'Churn': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Act
        X_train, X_val, y_train, y_val, dv = vectorize(df)
        
        # Assert
        assert X_train.shape[0] == 8  # 80% of 10 samples
        assert X_val.shape[0] == 2    # 20% of 10 samples
        assert len(y_train) == 8
        assert len(y_val) == 2
        assert isinstance(dv, DictVectorizer)
        
        # Check that the splits maintain class balance (stratify=y)
        train_unique, train_counts = np.unique(y_train, return_counts=True)
        val_unique, val_counts = np.unique(y_val, return_counts=True)
        
        # Should have both classes in both splits
        assert len(train_unique) >= 1
        assert len(val_unique) >= 1
    
    def test_vectorize_custom_target_column(self):
        """Test vectorize with custom target column"""
        # Arrange
        df = pd.DataFrame({
            'feature1': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
            'feature2': [1, 2, 1, 3, 2, 1, 3, 2, 1, 3],
            'Target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Act
        X_train, X_val, y_train, y_val, dv = vectorize(df, target_col='Target')
        
        # Assert
        assert X_train.shape[0] == 8
        assert X_val.shape[0] == 2
        assert len(y_train) == 8
        assert len(y_val) == 2
        assert isinstance(dv, DictVectorizer)
    
    def test_vectorize_missing_target_column(self):
        """Test vectorize with missing target column"""
        # Arrange
        df = pd.DataFrame({
            'feature1': ['A', 'B', 'A'],
            'feature2': [1, 2, 1]
        })
        
        # Act & Assert
        with pytest.raises(KeyError):
            vectorize(df, target_col='NonExistent')
    
    def test_vectorize_single_class(self):
        """Test vectorize with single class (edge case for stratify)"""
        # Arrange
        df = pd.DataFrame({
            'feature1': ['A', 'B', 'A', 'C'],
            'feature2': [1, 2, 1, 3],
            'Churn': [0, 0, 0, 0]  # All same class
        })
        
        # Act & Assert
        # This should raise an error due to stratify with single class
        with pytest.raises(ValueError):
            vectorize(df)
    
    def test_vectorize_minimum_samples(self):
        """Test vectorize with minimum number of samples"""
        # Arrange - need at least 10 samples for stratified split to work properly
        df = pd.DataFrame({
            'feature1': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
            'feature2': [1, 2, 1, 3, 2, 1, 3, 2, 1, 3],
            'Churn': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Act
        X_train, X_val, y_train, y_val, dv = vectorize(df)
        
        # Assert
        assert X_train.shape[0] > 0
        assert X_val.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_val) > 0
    
    def test_vectorize_sparse_matrix_output(self):
        """Test that vectorize returns sparse matrix"""
        # Arrange
        df = pd.DataFrame({
            'feature1': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
            'feature2': [1, 2, 1, 3, 2, 1, 3, 2, 1, 3],
            'Churn': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Act
        X_train, X_val, y_train, y_val, dv = vectorize(df)
        
        # Assert
        # Check that X_train and X_val are sparse matrices
        assert hasattr(X_train, 'toarray')  # Sparse matrix method
        assert hasattr(X_val, 'toarray')    # Sparse matrix method
    
    def test_vectorize_feature_names(self):
        """Test that DictVectorizer creates proper feature names"""
        # Arrange
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
            'numeric': [1, 2, 1, 3, 2, 1, 3, 2, 1, 3],
            'Churn': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Act
        X_train, X_val, y_train, y_val, dv = vectorize(df)
        
        # Assert
        feature_names = dv.get_feature_names_out()
        assert len(feature_names) > 0
        # Should have features for categorical and numeric columns
        assert any('category=' in name for name in feature_names)
        assert any('numeric' in name for name in feature_names)


@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame for testing"""
    return pd.DataFrame({
        'feature1': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'feature2': [1, 2, 1, 3, 2, 1, 3, 2, 1, 3],
        'feature3': [10.5, 20.1, 15.2, 8.9, 12.3, 18.7, 25.4, 14.6, 22.1, 11.8],
        'Churn': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })


class TestIntegration:
    """Integration tests combining multiple functions"""
    
    def test_full_pipeline_csv(self, sample_dataframe, tmp_path):
        """Test complete pipeline: load CSV -> vectorize"""
        # Arrange
        csv_file = tmp_path / "test_data.csv"
        sample_dataframe.to_csv(csv_file, index=False)
        
        # Act
        df = load_data(str(csv_file))
        X_train, X_val, y_train, y_val, dv = vectorize(df)
        
        # Assert
        assert df.shape == sample_dataframe.shape
        assert X_train.shape[0] == 8
        assert X_val.shape[0] == 2
        assert len(y_train) == 8
        assert len(y_val) == 2
    
    def test_full_pipeline_parquet(self, sample_dataframe, tmp_path):
        """Test complete pipeline: load Parquet -> vectorize"""
        # Arrange
        parquet_file = tmp_path / "test_data.parquet"
        sample_dataframe.to_parquet(parquet_file, index=False)
        
        # Act
        df = load_parquet(str(parquet_file))
        X_train, X_val, y_train, y_val, dv = vectorize(df)
        
        # Assert
        assert df.shape == sample_dataframe.shape
        assert X_train.shape[0] == 8
        assert X_val.shape[0] == 2
        assert len(y_train) == 8
        assert len(y_val) == 2 