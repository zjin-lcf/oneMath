// Standalone reproducer mirroring the geqrf+orgqr diagonal regression test
// added for https://github.com/uxlfoundation/oneMath/issues/626.
// Runs geqrf + orgqr repeatedly on a diagonal matrix (whose Q is identity)
// and verifies Q's diagonal is 1 on every run.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math.hpp"

template <typename fp>
bool run_case(sycl::queue& queue, std::int64_t n, std::int64_t lda, std::int64_t num_runs) {
    const std::int64_t m = n;
    const std::int64_t k = n;

    const fp diag_value = fp(2.0);
    std::vector<fp> A_initial(static_cast<size_t>(lda) * n, fp(0.0));
    for (std::int64_t j = 0; j < n; ++j)
        A_initial[j * lda + j] = diag_value;

    const fp tol = fp(10.0) * n * std::numeric_limits<fp>::epsilon();

    auto* A_dev = sycl::malloc_device<fp>(A_initial.size(), queue);
    auto* tau_dev = sycl::malloc_device<fp>(k, queue);

    const std::int64_t geqrf_ss = oneapi::math::lapack::geqrf_scratchpad_size<fp>(queue, m, n, lda);
    const std::int64_t orgqr_ss =
        oneapi::math::lapack::orgqr_scratchpad_size<fp>(queue, m, n, k, lda);
    const std::int64_t ss = std::max(geqrf_ss, orgqr_ss);
    auto* scratch_dev = sycl::malloc_device<fp>(ss, queue);

    std::vector<fp> Q(A_initial.size());
    bool ok = true;
    for (std::int64_t run = 0; run < num_runs && ok; ++run) {
        queue.memcpy(A_dev, A_initial.data(), A_initial.size() * sizeof(fp)).wait();

        oneapi::math::lapack::geqrf(queue, m, n, A_dev, lda, tau_dev, scratch_dev, geqrf_ss);
        oneapi::math::lapack::orgqr(queue, m, n, k, A_dev, lda, tau_dev, scratch_dev, orgqr_ss);
        queue.wait_and_throw();

        queue.memcpy(Q.data(), A_dev, Q.size() * sizeof(fp)).wait();

        int bad = 0;
        for (std::int64_t j = 0; j < n && bad < 5; ++j) {
            for (std::int64_t i = 0; i < m && bad < 5; ++i) {
                const fp expected = (i == j) ? fp(1.0) : fp(0.0);
                const fp got = std::abs(Q[j * lda + i]);
                if (std::abs(got - expected) > tol) {
                    std::cout << "    run " << run << ": Q(" << i << "," << j << ")=" << got
                              << " expected " << expected << "\n";
                    ++bad;
                    ok = false;
                }
            }
        }
    }

    sycl::free(A_dev, queue);
    sycl::free(tau_dev, queue);
    sycl::free(scratch_dev, queue);
    return ok;
}

int main() {
    sycl::queue queue{ sycl::gpu_selector_v };
    std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";

    struct Case {
        std::int64_t n, lda, runs;
    };
    const std::vector<Case> cases = { { 256, 256, 6 }, { 512, 512, 6 }, { 1024, 1024, 6 } };

    bool all_ok = true;
    for (auto c : cases) {
        std::cout << "[float ] n=" << c.n << " runs=" << c.runs << ": " << std::flush;
        bool ok = run_case<float>(queue, c.n, c.lda, c.runs);
        std::cout << (ok ? "PASS" : "FAIL") << "\n";
        all_ok &= ok;

        std::cout << "[double] n=" << c.n << " runs=" << c.runs << ": " << std::flush;
        ok = run_case<double>(queue, c.n, c.lda, c.runs);
        std::cout << (ok ? "PASS" : "FAIL") << "\n";
        all_ok &= ok;
    }

    std::cout << (all_ok ? "ALL PASS" : "SOME FAILED") << "\n";
    return all_ok ? 0 : 1;
}
