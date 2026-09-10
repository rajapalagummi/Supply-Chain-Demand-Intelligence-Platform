import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.forecast import mape, rmse
from models.inventory import compute_safety_stock, compute_reorder_point, compute_eoq


class TestMetrics:
    def test_mape_perfect(self):
        y = np.array([10.0, 20.0, 30.0])
        assert mape(y, y) == 0.0

    def test_mape_zeros_excluded(self):
        y_true = np.array([0.0, 10.0, 20.0])
        y_pred = np.array([5.0, 10.0, 20.0])
        result = mape(y_true, y_pred)
        assert result == 0.0

    def test_mape_positive(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([12.0, 18.0])
        result = mape(y_true, y_pred)
        assert result > 0

    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_rmse_positive(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        assert rmse(y_true, y_pred) > 0


class TestInventory:
    def setup_method(self):
        np.random.seed(42)
        self.demand = pd.Series(np.random.poisson(5, 365))

    def test_safety_stock_positive(self):
        ss = compute_safety_stock(self.demand)
        assert ss >= 0

    def test_safety_stock_higher_service_level(self):
        ss_95 = compute_safety_stock(self.demand, service_level=0.95)
        ss_99 = compute_safety_stock(self.demand, service_level=0.99)
        assert ss_99 >= ss_95

    def test_reorder_point_exceeds_safety_stock(self):
        ss = compute_safety_stock(self.demand)
        rop = compute_reorder_point(self.demand)
        assert rop >= ss

    def test_eoq_positive(self):
        eoq = compute_eoq(annual_demand=1000)
        assert eoq > 0

    def test_eoq_zero_demand(self):
        eoq = compute_eoq(annual_demand=0)
        assert eoq == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
