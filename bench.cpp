#include <benchmark/benchmark.h>
#include <array>
#include <random>
#include <iostream>
#include <algorithm>
#include <boost/math/tools/univariate_statistics.hpp>
#include <boost/math/tools/bivariate_statistics.hpp>
#include <boost/math/tools/signal_statistics.hpp>
#include <boost/math/tools/norms.hpp>
#include <boost/math/interpolators/barycentric_rational.hpp>
#include <boost/math/interpolators/cubic_b_spline.hpp>
#include <boost/math/differentiation/lanczos_smoothing.hpp>
#include <boost/math/interpolators/whittaker_shannon.hpp>
#include <boost/math/special_functions/daubechies_scaling.hpp>


template<class Real>
void BM_WhittakerShannon(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));

    std::mt19937 gen(323723);
    std::uniform_real_distribution<Real> dis(-0.95, 0.95);

    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = dis(gen);
    }

    auto ws = boost::math::interpolators::whittaker_shannon(std::move(v), Real(0), Real(1)/Real(32));
    Real arg = dis(gen);
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(ws(arg));
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_WhittakerShannon, double)->RangeMultiplier(2)->Range(1<<8, 1<<15)->Complexity(benchmark::oN);

static void UnitStep(benchmark::internal::Benchmark* b) {
  for (int i = 8; i <= 20; ++i)
      b->Args({i});
}

template<class Real>
void BM_RandGen(benchmark::State& state)
{

    std::default_random_engine gen;
    std::uniform_real_distribution<Real> dis(0, 3);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(dis(gen));
    }
}

BENCHMARK_TEMPLATE(BM_RandGen, float);
BENCHMARK_TEMPLATE(BM_RandGen, double);

template<class Real, int p>
void BM_DaubechiesScalingSingleCrankLinear(benchmark::State& state)
{
    using boost::math::daubechies_scaling;

    auto daub = daubechies_scaling<Real, p>(state.range(0));

    std::default_random_engine gen;
    std::uniform_real_distribution<Real> dis(daub.support().first, daub.support().second);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(daub.single_crank_linear(dis(gen)));
    }
}

//BENCHMARK_TEMPLATE(BM_DaubechiesScalingSingleCrankLinear, float, 3)->Apply(UnitStep);
//BENCHMARK_TEMPLATE(BM_DaubechiesScalingSingleCrankLinear, float, 4)->Apply(UnitStep);
//BENCHMARK_TEMPLATE(BM_DaubechiesScalingSingleCrankLinear, double, 3)->Apply(UnitStep);
BENCHMARK_TEMPLATE(BM_DaubechiesScalingSingleCrankLinear, double, 4)->Apply(UnitStep);


template<class Real, int p>
void BM_DaubechiesScalingLinearInterpolation(benchmark::State& state)
{
    using boost::math::daubechies_scaling;

    auto daub = daubechies_scaling<Real, p>(state.range(0));

    std::default_random_engine gen;
    std::uniform_real_distribution<Real> dis(daub.support().first, daub.support().second);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(daub.linear_interpolation(dis(gen)));
    }
}

//BENCHMARK_TEMPLATE(BM_DaubechiesScalingLinearInterpolation, float, 3)->Apply(UnitStep);
//BENCHMARK_TEMPLATE(BM_DaubechiesScalingLinearInterpolation, float, 4)->Apply(UnitStep);
//BENCHMARK_TEMPLATE(BM_DaubechiesScalingLinearInterpolation, double, 3)->Apply(UnitStep);
BENCHMARK_TEMPLATE(BM_DaubechiesScalingLinearInterpolation, double, 4)->Apply(UnitStep);


template<class Real, int p>
void BM_DaubechiesScalingConstantInterpolation(benchmark::State& state)
{
    using boost::math::daubechies_scaling;

    auto daub = daubechies_scaling<Real, p>(state.range(0));

    std::default_random_engine gen;
    std::uniform_real_distribution<Real> dis(daub.support().first, daub.support().second);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(daub.constant_interpolation(dis(gen)));
    }
}

//BENCHMARK_TEMPLATE(BM_DaubechiesScalingConstantInterpolation, float, 3)->Apply(UnitStep);
//BENCHMARK_TEMPLATE(BM_DaubechiesScalingConstantInterpolation, float, 4)->Apply(UnitStep);
//BENCHMARK_TEMPLATE(BM_DaubechiesScalingConstantInterpolation, double, 3)->Apply(UnitStep);
BENCHMARK_TEMPLATE(BM_DaubechiesScalingConstantInterpolation, double, 4)->Apply(UnitStep);


template<class Real>
void BM_BarycentricRationalConstructor(benchmark::State& state)
{
    std::vector<Real> x(state.range(0));
    std::vector<Real> y(state.range(0));
    for (size_t i = 0; i < x.size(); ++i)
    {
        x[i] = i;
        y[i] = i;
    }

    using boost::math::barycentric_rational;

    for (auto _ : state)
    {
        auto b = barycentric_rational(x.data(), y.data(), y.size());
        benchmark::DoNotOptimize(b);
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_BarycentricRationalConstructor, double)->RangeMultiplier(2)->Range(1<<6, 1<<22)->Complexity(benchmark::oN);

template<class Real>
void BM_CubicBSplineConstructor(benchmark::State& state)
{
    std::vector<Real> y(state.range(0));
    Real t0 = 0;
    Real h = Real(1)/Real(64);
    for (size_t i = 0; i < y.size(); ++i)
    {
        y[i] = std::sin(t0 + i*h);
    }

    using boost::math::cubic_b_spline;

    for (auto _ : state)
    {
        auto b = cubic_b_spline(y.data(), y.size(), t0, h);
        benchmark::DoNotOptimize(b);
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_CubicBSplineConstructor, double)->RangeMultiplier(2)->Range(1<<6, 1<<22)->Complexity(benchmark::oN);

template<class Real>
void BM_CubicBSplineEvaluation(benchmark::State& state)
{
    std::vector<Real> y(state.range(0));
    Real t0 = 0;
    Real h = Real(1)/Real(64);
    for (size_t i = 0; i < y.size(); ++i)
    {
        y[i] = std::sin(t0 + i*h);
    }

    using boost::math::cubic_b_spline;

    auto b = cubic_b_spline(y.data(), y.size(), t0, h);
    Real t = t0 + y.size()*h/2;
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(b(t));
    }
}

BENCHMARK_TEMPLATE(BM_CubicBSplineEvaluation, double)->RangeMultiplier(2)->Range(1<<6, 1<<22);

template<class Real>
void BM_SineEvaluation(benchmark::State& state)
{
    std::default_random_engine gen;
    std::uniform_real_distribution<Real> x_dis(0, 3.14159);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(std::sin(x_dis(gen)));
    }
}

BENCHMARK_TEMPLATE(BM_SineEvaluation, double);

template<class Real>
void BM_Sqrt(benchmark::State& state)
{
    std::default_random_engine gen;
    std::uniform_real_distribution<Real> x_dis(0, 3.14159);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(std::sqrt(x_dis(gen)));
    }
}

BENCHMARK_TEMPLATE(BM_Sqrt, double);


template<class Real>
void BM_Pow(benchmark::State& state)
{
    std::default_random_engine gen;
    std::uniform_real_distribution<Real> x_dis(0, 3.14159);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(std::pow(x_dis(gen), x_dis(gen)));
    }
}

BENCHMARK_TEMPLATE(BM_Pow, double);

template<class Real>
void BM_UniformRealDistribution(benchmark::State& state)
{
    std::default_random_engine gen;
    std::uniform_real_distribution<Real> x_dis(0, 3.14159);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(x_dis(gen));
    }
}

BENCHMARK_TEMPLATE(BM_UniformRealDistribution, double);


template<class Real>
void BM_LanczosVelocity(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    std::vector<Real> w(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    using boost::math::differentiation::discrete_lanczos_derivative;
    auto lanczos = discrete_lanczos_derivative(Real(1),4, 2);
    for (auto _ : state)
    {
        lanczos(v, w);
        benchmark::DoNotOptimize(w[0]);
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_LanczosVelocity, double)->RangeMultiplier(2)->Range(1<<6, 1<<22)->Complexity(benchmark::oN);


template<class Real>
void BM_LanczosVelocityPointEvaluation(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    using boost::math::differentiation::discrete_lanczos_derivative;
    auto lanczos = discrete_lanczos_derivative(Real(1));
    for (auto _ : state)
    {
        auto w = lanczos(v, v.size()/2);
        benchmark::DoNotOptimize(w);
    }
    state.SetComplexityN(state.range(0));
}


BENCHMARK_TEMPLATE(BM_LanczosVelocityPointEvaluation, double)->RangeMultiplier(2)->Range(1<<6, 1<<22)->Complexity();


template<class Real>
void BM_LanczosVelocityConstructor(benchmark::State& state)
{
    using boost::math::differentiation::discrete_lanczos_derivative;
    for (auto _ : state)
    {
        auto lanczos = discrete_lanczos_derivative(Real(1), state.range(0));
        benchmark::DoNotOptimize(lanczos.get_spacing());
    }
    state.SetComplexityN(state.range(0));
}


BENCHMARK_TEMPLATE(BM_LanczosVelocityConstructor, double)->Arg(3)->Arg(6)->Arg(12)->Arg(18)->Arg(24)->Arg(30)->Arg(36)->Arg(42)->Arg(48)->Arg(96)->Arg(200)->Arg(400)->Arg(800)->Arg(1600)->Complexity();


template<class Real>
void BM_LanczosAcceleration(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    std::vector<Real> w(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    using boost::math::differentiation::discrete_lanczos_derivative;
    auto lanczos = discrete_lanczos_derivative<Real, 2>(Real(1));
    for (auto _ : state)
    {
        lanczos(v, w);
        benchmark::DoNotOptimize(w[0]);
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_LanczosAcceleration, double)->RangeMultiplier(2)->Range(1<<6, 1<<22)->Complexity(benchmark::oN);

template<class Real>
void BM_LanczosAccelerationPointEvaluation(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    using boost::math::differentiation::discrete_lanczos_derivative;
    auto lanczos = discrete_lanczos_derivative<Real, 2>(Real(1));
    for (auto _ : state)
    {
        auto w = lanczos(v, v.size()/2);
        benchmark::DoNotOptimize(w);
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_LanczosAccelerationPointEvaluation, double)->RangeMultiplier(2)->Range(1<<6, 1<<22)->Complexity();


template<class Real>
void BM_LanczosAccelerationConstructor(benchmark::State& state)
{
    using boost::math::differentiation::discrete_lanczos_derivative;
    for (auto _ : state)
    {
        auto lanczos = discrete_lanczos_derivative<Real, 2>(Real(1), state.range(0));
        benchmark::DoNotOptimize(lanczos.get_spacing());
    }
    state.SetComplexityN(state.range(0));
}


BENCHMARK_TEMPLATE(BM_LanczosAccelerationConstructor, double)->Arg(3)->Arg(6)->Arg(12)->Arg(18)->Arg(24)->Arg(30)->Arg(36)->Arg(42)->Arg(48)->Arg(96)->Arg(200)->Arg(400)->Arg(800)->Arg(1600)->Complexity();


template<class Real>
void BM_HoyerSparsity(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    for (auto _ : state)
    {
        auto hs = boost::math::tools::hoyer_sparsity(v);
        benchmark::DoNotOptimize(hs);
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_HoyerSparsity, double)->RangeMultiplier(2)->Range(1<<2, 1<<15)->Complexity(benchmark::oN);

template<class Real>
void BM_Median(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    for (auto _ : state)
    {
        auto med = boost::math::tools::median(v);
        benchmark::DoNotOptimize(med);
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_Median, double)->RangeMultiplier(2)->Range(1<<2, 1<<15)->Complexity();


template<class Real>
void BM_MedianWithShuffle(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    std::random_device rd;
    std::mt19937 g(rd());

    for (auto _ : state)
    {
        state.PauseTiming();
        std::shuffle(v.begin(), v.end(), g);
        state.ResumeTiming();
        auto med = boost::math::tools::median(v);
        benchmark::DoNotOptimize(med);
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_MedianWithShuffle, double)->RangeMultiplier(2)->Range(1<<2, 1<<15)->Complexity();


template<class Real>
void BM_Covariance(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    std::vector<Real> u(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
        u[i] = -i;
    }

    for (auto _ : state)
    {
        auto cov = boost::math::tools::covariance(u, v);
        benchmark::DoNotOptimize(cov);
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_Covariance, double)->RangeMultiplier(2)->Range(1<<2, 1<<15)->Complexity(benchmark::oN);


template<class Real>
void BM_Mean(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::tools::mean(v));
    }
    state.SetComplexityN(state.range(0));
    state.SetBytesProcessed(state.iterations()*state.range(0)*sizeof(Real));
}

BENCHMARK_TEMPLATE(BM_Mean, double)->RangeMultiplier(2)->Range(1<<3, 1<<30)->Complexity(benchmark::oN);

template<class Real>
void BM_MeanNaive(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    for (auto _ : state)
    {
        Real sum = 0;
        for (auto & x : v) {
            sum += x;
        }
        benchmark::DoNotOptimize(sum/v.size());
    }
    state.SetComplexityN(state.range(0));
    state.SetBytesProcessed(state.iterations()*state.range(0)*sizeof(Real));
}

BENCHMARK_TEMPLATE(BM_MeanNaive, double)->RangeMultiplier(2)->Range(1<<3, 1<<30)->Complexity(benchmark::oN);


template<class Real>
void BM_Variance(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::tools::variance(v.begin(), v.end()));
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_Variance, double)->RangeMultiplier(2)->Range(1<<2, 1<<28)->Complexity(benchmark::oN);


template<class Real>
void BM_Skewness(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::tools::skewness(v.begin(), v.end()));
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_Skewness, double)->RangeMultiplier(2)->Range(1<<2, 1<<20)->Complexity(benchmark::oN);

template<class Real>
void BM_Kurtosis(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::tools::kurtosis(v.begin(), v.end()));
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_Kurtosis, double)->RangeMultiplier(2)->Range(1<<2, 1<<20)->Complexity(benchmark::oN);


template<class Real>
void BM_L1(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::tools::l1_norm(v.begin(), v.end()));
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_L1, double)->RangeMultiplier(2)->Range(1<<2, 1<<15)->Complexity(benchmark::oN);

template<class Real>
void BM_L2(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::tools::l2_norm(v.begin(), v.end()));
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_L2, double)->RangeMultiplier(2)->Range(1<<2, 1<<15)->Complexity(benchmark::oN);

template<class Real>
void BM_L3(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::tools::lp_norm(v.begin(), v.end(), 3));
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_L3, double)->RangeMultiplier(2)->Range(1<<2, 1<<15)->Complexity(benchmark::oN);

template<class Real>
void BM_TV(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::tools::total_variation(v.begin(), v.end()));
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_TV, double)->RangeMultiplier(2)->Range(1<<2, 1<<15)->Complexity(benchmark::oN);

template<class Real>
void BM_SupNorm(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::tools::sup_norm(v.begin(), v.end()));
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_SupNorm, double)->RangeMultiplier(2)->Range(1<<2, 1<<15)->Complexity(benchmark::oN);

template<class Real>
void BM_Gini(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(boost::math::tools::gini_coefficient(v.begin(), v.end()));
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(BM_Gini, double)->RangeMultiplier(2)->Range(1<<2, 1<<15)->Complexity();


template<class Real>
void BM_GiniUnsorted(benchmark::State& state)
{
    std::vector<Real> v(state.range(0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());

    for (auto _ : state)
    {
        state.PauseTiming();
        std::shuffle(v.begin(), v.end(), g);
        state.ResumeTiming();
        benchmark::DoNotOptimize(boost::math::tools::gini_coefficient(v.begin(), v.end()));
    }
    state.SetComplexityN(state.range(0));
}


BENCHMARK_TEMPLATE(BM_GiniUnsorted, double)->RangeMultiplier(2)->Range(1<<2, 1<<15)->Complexity();


BENCHMARK_MAIN();
