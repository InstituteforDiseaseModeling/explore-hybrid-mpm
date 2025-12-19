"""Quick validation test - just test correctness on small problem."""

from test_taichi_validation import test_correctness_small, test_cpu_backend_exact_match

print("Running quick validation tests...\n")

# Test 1: Small problem correctness
test1_pass = test_correctness_small(verbose=True)

# Test 2: CPU backend exact match
test2_pass = test_cpu_backend_exact_match(verbose=True)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
if test1_pass and test2_pass:
    print("✓ ALL TESTS PASSED - GPU implementation is correct!")
else:
    print("✗ Some tests failed")
    if not test1_pass:
        print("  - Test 1 (GPU Metal correctness) FAILED")
    if not test2_pass:
        print("  - Test 2 (CPU backend exact match) FAILED")
