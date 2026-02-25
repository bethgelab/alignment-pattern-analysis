"""Tests for intersubject consistency functions."""
import numpy as np
import pytest

from multitasking.intersubject_consistency import (
    compute_leave_one_out_mean,
    compute_mean,
    compute_mean_over_subs,
)


class TestComputeMeanFunctions:
    """Test suite for compute_mean_over_subs and compute_leave_one_out_mean."""

    def test_mean_over_subs_equals_loo_mean(self):
        """See name.

        Test that compute_mean_over_subs with all-but-one subjects equals
        compute_leave_one_out_mean for the excluded subject.
        """
        # Create synthetic data with 4 subjects and 2 ROIs
        np.random.seed(42)
        data = {
            "sub01": {
                "roi1": np.random.randn(10, 5),
                "roi2": np.random.randn(8, 3),
            },
            "sub02": {
                "roi1": np.random.randn(10, 5),
                "roi2": np.random.randn(8, 3),
            },
            "sub03": {
                "roi1": np.random.randn(10, 5),
                "roi2": np.random.randn(8, 3),
            },
            "sub04": {
                "roi1": np.random.randn(10, 5),
                "roi2": np.random.randn(8, 3),
            },
        }

        # Pick a subject to leave out
        subject_to_exclude = "sub02"
        subs_to_average = [s for s in data.keys() if s != subject_to_exclude]

        # Compute using both methods
        result_mean_over_subs = compute_mean_over_subs(data, subs_to_average)
        result_loo = compute_leave_one_out_mean(data, subject_to_exclude)

        # They should be equal
        assert result_mean_over_subs.keys() == result_loo.keys()
        for roi in result_mean_over_subs.keys():
            np.testing.assert_array_almost_equal(
                result_mean_over_subs[roi],
                result_loo[roi],
                decimal=10,
                err_msg=f"Mismatch in ROI {roi}"
            )

    def test_mean_over_subs_with_two_subjects_returns_specified_subject(self):
        """See name.

        Test that with only two subjects, compute_mean_over_subs with one
        subject ID returns the feature maps of that subject ID.
        """
        # Create synthetic data with 2 subjects and 2 ROIs
        np.random.seed(123)
        data = {
            "sub01": {
                "roi1": np.random.randn(10, 5),
                "roi2": np.random.randn(8, 3),
            },
            "sub02": {
                "roi1": np.random.randn(10, 5),
                "roi2": np.random.randn(8, 3),
            },
        }

        # Compute mean over only sub01
        result = compute_mean_over_subs(data, ["sub01"])

        # Result should be exactly sub01's data
        assert result.keys() == data["sub01"].keys()
        for roi in result.keys():
            np.testing.assert_array_equal(
                result[roi],
                data["sub01"][roi],
                err_msg=f"Mismatch in ROI {roi}"
            )

    def test_loo_mean_with_two_subjects_returns_other_subject(self):
        """See name.

        Test that with only two subjects, compute_leave_one_out_mean with
        one subject returns the feature map of the other subject.
        """
        # Create synthetic data with 2 subjects and 2 ROIs
        np.random.seed(456)
        data = {
            "sub01": {
                "roi1": np.random.randn(10, 5),
                "roi2": np.random.randn(8, 3),
            },
            "sub02": {
                "roi1": np.random.randn(10, 5),
                "roi2": np.random.randn(8, 3),
            },
        }

        # Compute leave-one-out mean, leaving out sub01
        result = compute_leave_one_out_mean(data, "sub01")

        # Result should be exactly sub02's data
        assert result.keys() == data["sub02"].keys()
        for roi in result.keys():
            np.testing.assert_array_equal(
                result[roi],
                data["sub02"][roi],
                err_msg=f"Mismatch in ROI {roi}"
            )

    def test_compute_mean_wrapper_loo_mode(self):
        """Test that compute_mean works correctly in leave-one-out mode."""
        np.random.seed(789)
        data = {
            "sub01": {"roi1": np.random.randn(5, 3)},
            "sub02": {"roi1": np.random.randn(5, 3)},
            "sub03": {"roi1": np.random.randn(5, 3)},
        }

        # When subs_to_average is None, should use leave-one-out
        result = compute_mean(data, "sub01", subs_to_average=None)
        expected = compute_leave_one_out_mean(data, "sub01")

        for roi in result.keys():
            np.testing.assert_array_equal(result[roi], expected[roi])


    def test_compute_mean_wrapper_specified_subs_mode(self):
        """Test that compute_mean works correctly with specified subjects."""
        np.random.seed(101112)
        data = {
            "sub01": {"roi1": np.random.randn(5, 3)},
            "sub02": {"roi1": np.random.randn(5, 3)},
            "sub03": {"roi1": np.random.randn(5, 3)},
        }

        # When subs_to_average is provided
        result = compute_mean(data, "sub01", subs_to_average=["sub02", "sub03"])
        expected = compute_mean_over_subs(data, ["sub02", "sub03"])

        for roi in result.keys():
            np.testing.assert_array_equal(result[roi], expected[roi])

    def test_compute_mean_raises_error_when_target_in_subs_to_average(self):
        """Test error at subject overlap."""
        data = {
            "sub01": {"roi1": np.random.randn(5, 3)},
            "sub02": {"roi1": np.random.randn(5, 3)},
        }

        with pytest.raises(ValueError, match="should not be in subs_to_average"):
            compute_mean(data, "sub01", subs_to_average=["sub01", "sub02"])

    def test_mean_over_subs_raises_error_when_empty_list(self):
        """Test that compute_mean_over_subs raises an error with empty list."""
        data = {
            "sub01": {"roi1": np.random.randn(5, 3)},
            "sub02": {"roi1": np.random.randn(5, 3)},
        }

        with pytest.raises(ValueError, match="subs_to_average is empty"):
            compute_mean_over_subs(data, [])

    def test_loo_mean_computes_correct_average(self):
        """Test that leave-one-out mean computes the correct average.

        (Oh please. But why not)
        """
        # Create simple data where we can manually verify the average
        data = {
            "sub01": {
                "roi1": np.array([[1.0, 2.0], [3.0, 4.0]]),
            },
            "sub02": {
                "roi1": np.array([[2.0, 4.0], [6.0, 8.0]]),
            },
            "sub03": {
                "roi1": np.array([[3.0, 6.0], [9.0, 12.0]]),
            },
        }

        # Leave out sub01, should average sub02 and sub03
        result = compute_leave_one_out_mean(data, "sub01")
        expected = np.array([[2.5, 5.0], [7.5, 10.0]])

        np.testing.assert_array_equal(result["roi1"], expected)

    def test_mean_over_subs_computes_correct_average(self):
        """Test that mean_over_subs computes the correct average."""
        # Create simple data where we can manually verify the average
        data = {
            "sub01": {
                "roi1": np.array([[1.0, 2.0], [3.0, 4.0]]),
            },
            "sub02": {
                "roi1": np.array([[2.0, 4.0], [6.0, 8.0]]),
            },
            "sub03": {
                "roi1": np.array([[3.0, 6.0], [9.0, 12.0]]),
            },
        }

        # Average sub02 and sub03
        result = compute_mean_over_subs(data, ["sub02", "sub03"])
        expected = np.array([[2.5, 5.0], [7.5, 10.0]])

        np.testing.assert_array_equal(result["roi1"], expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

