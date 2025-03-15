#!/usr/bin/env python3
"""
Script to run all unit tests for the WishartEM package.
"""
import unittest

if __name__ == "__main__":
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("wishartem/tests", pattern="test_*.py")
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_runner.run(test_suite)